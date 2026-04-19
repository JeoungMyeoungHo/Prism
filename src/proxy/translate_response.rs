//! OpenAI Chat Completions → Anthropic Messages / OpenAI Responses response
//! translation.
//!
//! This module also owns the generic text-ish value extractor used by both
//! non-streaming and streaming translators (re-exported from the parent for
//! the SSE path).

use super::{current_timestamp_seconds, ApiError};
use crate::types::{OpenAiChatCompletionResponse, OpenAiUsage};
use axum::http::StatusCode;
use serde_json::{json, Value};

pub(super) fn openai_response_to_anthropic(
    response: OpenAiChatCompletionResponse,
    requested_model: &str,
) -> Result<Value, ApiError> {
    let choice = response.choices.into_iter().next().ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            "upstream response contained no choices",
        )
    })?;

    let mut content = Vec::new();

    if let Some(text) = extract_openai_text(choice.message.content.as_ref())? {
        if !text.is_empty() {
            content.push(json!({
                "type": "text",
                "text": text,
            }));
        }
    }

    if let Some(tool_calls) = choice.message.tool_calls {
        for tool_call in tool_calls {
            content.push(json!({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": decode_tool_arguments(&tool_call.function.arguments),
            }));
        }
    }

    let usage = response.usage.as_ref();
    let mut usage_obj = json!({
        "input_tokens": usage.map(|u| u.prompt_tokens).unwrap_or_default(),
        "output_tokens": usage.map(|u| u.completion_tokens).unwrap_or_default(),
    });
    if let Some(cache_read) = usage.and_then(OpenAiUsage::cache_read_tokens) {
        usage_obj["cache_read_input_tokens"] = json!(cache_read);
    }
    if let Some(cache_creation) = usage.and_then(|u| u.cache_creation_input_tokens) {
        usage_obj["cache_creation_input_tokens"] = json!(cache_creation);
    }

    Ok(json!({
        "id": response.id,
        "type": "message",
        "role": choice.message.role.unwrap_or_else(|| "assistant".into()),
        "model": response.model.unwrap_or_else(|| requested_model.to_string()),
        "content": content,
        "stop_reason": map_finish_reason(choice.finish_reason.as_deref()),
        "stop_sequence": Value::Null,
        "usage": usage_obj,
    }))
}

pub(super) fn openai_chat_response_to_responses(
    response: OpenAiChatCompletionResponse,
    requested_model: &str,
    client_parallel_tool_calls: Option<bool>,
) -> Result<Value, ApiError> {
    let created_at = current_timestamp_seconds();
    let choice = response.choices.into_iter().next().ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            "upstream response contained no choices",
        )
    })?;

    let mut output = Vec::new();

    if let Some(text) = extract_openai_text(choice.message.content.as_ref())? {
        if !text.is_empty() {
            output.push(json!({
                "id": format!("msg_{}", response.id),
                "type": "message",
                "status": "completed",
                "role": choice.message.role.clone().unwrap_or_else(|| "assistant".into()),
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": []
                    }
                ]
            }));
        }
    }

    if let Some(tool_calls) = choice.message.tool_calls {
        for tool_call in tool_calls {
            let call_id = tool_call.id.clone();
            output.push(json!({
                "id": tool_call.id,
                "type": "function_call",
                "call_id": call_id,
                "name": tool_call.function.name,
                "arguments": encode_tool_arguments(tool_call.function.arguments),
                "status": "completed"
            }));
        }
    }

    let usage = response.usage.as_ref().cloned().unwrap_or_default();
    let status = match choice.finish_reason.as_deref() {
        Some("length") => "incomplete",
        Some("content_filter") => "failed",
        _ => "completed",
    };

    // Prefer the client's declared `parallel_tool_calls` setting; fall back
    // to whether upstream actually emitted more than one function_call.
    let parallel_tool_calls = client_parallel_tool_calls.unwrap_or_else(|| {
        output
            .iter()
            .filter(|item| item.get("type") == Some(&json!("function_call")))
            .count()
            > 1
    });

    let mut usage_obj = json!({
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
    });
    if let Some(cache_read) = usage.cache_read_tokens() {
        usage_obj["input_tokens_details"] = json!({ "cached_tokens": cache_read });
    }

    let mut translated = json!({
        "id": response.id,
        "object": "response",
        "created_at": created_at,
        "completed_at": created_at,
        "status": status,
        "model": response.model.unwrap_or_else(|| requested_model.to_string()),
        "output": output,
        "parallel_tool_calls": parallel_tool_calls,
        "usage": usage_obj,
    });

    // `translated` just built via `json!({ ... })` — always Value::Object.
    let translated_object = translated
        .as_object_mut()
        .expect("json! literal always yields Value::Object");
    match choice.finish_reason.as_deref() {
        Some("length") => {
            translated_object.insert(
                "incomplete_details".into(),
                json!({ "reason": "max_output_tokens" }),
            );
        }
        Some("content_filter") => {
            translated_object.insert(
                "error".into(),
                json!({
                    "type": "content_filter",
                    "code": "content_filter",
                    "message": "upstream content filter blocked the response",
                }),
            );
        }
        _ => {}
    }

    Ok(translated)
}

fn extract_openai_text(content: Option<&Value>) -> Result<Option<String>, ApiError> {
    let Some(content) = content else {
        return Ok(None);
    };

    match content {
        Value::Null => Ok(None),
        other => Ok(extract_textish_value(other)),
    }
}

pub(super) fn extract_textish_value(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let mut text = String::new();
            for item in items {
                if let Some(part) = extract_textish_value(item) {
                    text.push_str(&part);
                }
            }
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        Value::Object(object) => {
            for key in [
                "text",
                "content",
                "reasoning_content",
                "reasoning",
                "reasoning_details",
                "thinking",
                "summary",
            ] {
                if let Some(candidate) = object.get(key) {
                    if let Some(text) = extract_textish_value(candidate) {
                        return Some(text);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn decode_tool_arguments(arguments: &Value) -> Value {
    match arguments {
        Value::Null => json!({}),
        Value::String(text) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                json!({})
            } else {
                serde_json::from_str(trimmed).unwrap_or_else(|_| Value::String(text.to_string()))
            }
        }
        other => other.clone(),
    }
}

pub(super) fn encode_tool_arguments(arguments: Value) -> String {
    match arguments {
        Value::String(text) => text,
        Value::Null => "{}".into(),
        other => serde_json::to_string(&other).unwrap_or_else(|_| "{}".into()),
    }
}

pub(super) fn map_finish_reason(reason: Option<&str>) -> Value {
    match reason {
        Some("stop") => Value::String("end_turn".into()),
        Some("length") => Value::String("max_tokens".into()),
        Some("tool_calls") => Value::String("tool_use".into()),
        Some("content_filter") => Value::String("refusal".into()),
        Some(other) => Value::String(other.to_string()),
        None => Value::Null,
    }
}
