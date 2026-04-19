//! Anthropic / OpenAI Responses → OpenAI Chat Completions request translation.
//!
//! Public entry points are [`anthropic_request_to_openai`] (consumes a typed
//! `AnthropicMessagesRequest`) and [`responses_request_to_openai_chat`]
//! (operates on raw JSON because the Responses schema is intentionally
//! flexible). Both produce a [`PreparedUpstreamRequest`] containing the
//! OpenAI body and any adapter notes generated while normalizing for the
//! target provider.
//!
//! The file is intentionally kept as one unit — the block/content helpers
//! (`append_anthropic_message`, `extract_system_prompt`, `document_block_*`,
//! `fallback_*_text`, …) are all called from the two top-level translators
//! and from each other. Splitting them further would just fragment the
//! conversion rules without sharpening module boundaries.

use super::translate_response::encode_tool_arguments;
use super::{convert_tool_choice, convert_tools, normalize_schema, ApiError};
use crate::types::{
    AnthropicBlock, AnthropicMessage, AnthropicMessagesRequest, AnthropicSystemPrompt, Backend,
};
use axum::http::StatusCode;
use serde::Serialize;
use serde_json::{json, Map, Value};

pub(super) struct PreparedUpstreamRequest {
    pub(super) body: Value,
    pub(super) adapter_notes: Vec<String>,
}

pub(super) fn anthropic_request_to_openai(
    request: AnthropicMessagesRequest,
    backend: &Backend,
) -> Result<PreparedUpstreamRequest, ApiError> {
    let mut messages = Vec::new();

    if let Some(system) = request.system {
        let system_text = extract_system_prompt(system)?;
        if !system_text.is_empty() {
            messages.push(json!({
                "role": "system",
                "content": system_text,
            }));
        }
    }

    for message in request.messages {
        append_anthropic_message(message, &mut messages)?;
    }

    let mut openai_request = json!({
        "model": request.model,
        "messages": messages,
    });

    // `openai_request` was just built with `json!({ ... })`, which always
    // yields `Value::Object`. The expect here is a non-panicking invariant.
    let object = openai_request
        .as_object_mut()
        .expect("json! literal always yields Value::Object");

    if let Some(max_tokens) = request.max_tokens {
        // Canonical OpenAI field for modern models; adapters downgrade to
        // `max_tokens` for upstreams that don't accept the newer form.
        object.insert("max_completion_tokens".into(), json!(max_tokens));
    }

    if let Some(stream) = request.stream {
        object.insert("stream".into(), json!(stream));
        if stream {
            let stream_options = request
                .stream_options
                .clone()
                .unwrap_or_else(|| json!({ "include_usage": true }));
            object.insert("stream_options".into(), stream_options);
        }
    }

    if let Some(temperature) = request.temperature {
        object.insert("temperature".into(), json!(temperature));
    }

    if let Some(top_p) = request.top_p {
        object.insert("top_p".into(), json!(top_p));
    }

    if let Some(top_k) = request.top_k {
        // OpenAI Chat Completions ignores top_k; many OpenAI-compatible
        // upstreams (Fireworks, Z.AI, vLLM) honor it. Forward verbatim.
        object.insert("top_k".into(), json!(top_k));
    }

    if let Some(stop_sequences) = request.stop_sequences {
        object.insert("stop".into(), json!(stop_sequences));
    }

    if let Some(metadata) = request.metadata {
        object.insert("metadata".into(), metadata);
    }

    if let Some(service_tier) = request.service_tier {
        object.insert("service_tier".into(), json!(service_tier));
    }

    if let Some(user) = request.user {
        object.insert("user".into(), json!(user));
    }

    if let Some(tools) = request.tools {
        object.insert("tools".into(), convert_tools(tools)?);
    }

    if let Some(tool_choice) = request.tool_choice {
        object.insert("tool_choice".into(), convert_tool_choice(tool_choice)?);
    }

    // Anthropic extended-thinking config has no OpenAI analog — record a
    // visible note rather than silently dropping.
    let mut adapter_notes = Vec::new();
    if request.thinking.is_some() {
        adapter_notes.push(
            "Prism dropped the Anthropic `thinking` parameter: no OpenAI-compatible equivalent."
                .into(),
        );
    }

    adapter_notes.extend(backend.provider.adapter().adapt_request(object));

    Ok(PreparedUpstreamRequest {
        body: openai_request,
        adapter_notes,
    })
}

pub(super) fn responses_request_to_openai_chat(
    request: Value,
    backend: &Backend,
) -> Result<PreparedUpstreamRequest, ApiError> {
    let object = request.as_object().ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Responses request body must be a JSON object",
        )
    })?;

    let model = object.get("model").and_then(Value::as_str).ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Responses request requires a string `model` field",
        )
    })?;

    let mut messages = Vec::new();

    if let Some(instructions) = object.get("instructions") {
        let system_text = extract_responses_text(instructions)?;
        if !system_text.is_empty() {
            messages.push(json!({
                "role": "system",
                "content": system_text,
            }));
        }
    }

    if let Some(input) = object.get("input") {
        append_responses_input(input, &mut messages)?;
    }

    let mut openai_request = json!({
        "model": model,
        "messages": messages,
    });

    // `openai_request` just built via `json!({ ... })` — always Value::Object.
    let request_object = openai_request
        .as_object_mut()
        .expect("json! literal always yields Value::Object");

    if let Some(max_output_tokens) = object.get("max_output_tokens").and_then(Value::as_u64) {
        // Canonical OpenAI field; adapters downgrade to `max_tokens` for
        // upstreams that don't accept the newer form.
        request_object.insert("max_completion_tokens".into(), json!(max_output_tokens));
    }

    if let Some(stream) = object.get("stream").and_then(Value::as_bool) {
        request_object.insert("stream".into(), json!(stream));
        if stream {
            let stream_options = object
                .get("stream_options")
                .cloned()
                .unwrap_or_else(|| json!({ "include_usage": true }));
            request_object.insert("stream_options".into(), stream_options);
        }
    }

    if let Some(temperature) = object.get("temperature").and_then(Value::as_f64) {
        request_object.insert("temperature".into(), json!(temperature));
    }

    if let Some(top_p) = object.get("top_p").and_then(Value::as_f64) {
        request_object.insert("top_p".into(), json!(top_p));
    }

    if let Some(parallel_tool_calls) = object.get("parallel_tool_calls").and_then(Value::as_bool) {
        request_object.insert("parallel_tool_calls".into(), json!(parallel_tool_calls));
    }

    if let Some(text) = object.get("text") {
        if let Some(response_format) = convert_responses_text_format(text)? {
            request_object.insert("response_format".into(), response_format);
        }
    }

    if let Some(tools) = object.get("tools") {
        request_object.insert("tools".into(), convert_responses_tools(tools)?);
    }

    if let Some(tool_choice) = object.get("tool_choice") {
        request_object.insert(
            "tool_choice".into(),
            convert_responses_tool_choice(tool_choice)?,
        );
    }

    let adapter_notes = backend.provider.adapter().adapt_request(request_object);

    Ok(PreparedUpstreamRequest {
        body: openai_request,
        adapter_notes,
    })
}

fn append_responses_input(input: &Value, output: &mut Vec<Value>) -> Result<(), ApiError> {
    match input {
        Value::Null => Ok(()),
        Value::String(text) => {
            if !text.is_empty() {
                output.push(json!({
                    "role": "user",
                    "content": text,
                }));
            }
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                append_responses_input_item(item, output)?;
            }
            Ok(())
        }
        Value::Object(_) => append_responses_input_item(input, output),
        other => Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("unsupported Responses `input` shape: {other}"),
        )),
    }
}

fn append_responses_input_item(item: &Value, output: &mut Vec<Value>) -> Result<(), ApiError> {
    let object = item.as_object().ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Responses input items must be objects",
        )
    })?;

    let item_type = object.get("type").and_then(Value::as_str);
    if matches!(
        item_type,
        Some("function_call_output" | "custom_tool_call_output")
    ) {
        let call_id = object
            .get("call_id")
            .or_else(|| object.get("tool_call_id"))
            .or_else(|| object.get("id"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "Responses function_call_output requires `call_id`",
                )
            })?;
        let content = object
            .get("output")
            .or_else(|| object.get("content"))
            .map(extract_responses_text)
            .transpose()?
            .unwrap_or_default();

        output.push(json!({
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }));
        return Ok(());
    }

    if matches!(item_type, Some("function_call" | "custom_tool_call")) {
        let call_id = object
            .get("call_id")
            .or_else(|| object.get("id"))
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "Responses function_call item requires `call_id` or `id`",
                )
            })?;
        let name = object.get("name").and_then(Value::as_str).ok_or_else(|| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "Responses function_call item requires `name`",
            )
        })?;
        let arguments = encode_tool_arguments(
            object
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| Value::String("{}".into())),
        );

        output.push(json!({
            "role": "assistant",
            "content": Value::Null,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    }
                }
            ]
        }));
        return Ok(());
    }

    let role = object
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or(match item_type {
            Some("message") => "user",
            _ => "user",
        });

    match role {
        "system" | "developer" => {
            let text = object
                .get("content")
                .map(extract_responses_text)
                .transpose()?
                .unwrap_or_default();

            if !text.is_empty() {
                output.push(json!({
                    "role": "system",
                    "content": text,
                }));
            }
        }
        "assistant" => {
            let content = object
                .get("content")
                .map(extract_responses_assistant_content)
                .transpose()?
                .unwrap_or(Value::Null);
            output.push(json!({
                "role": "assistant",
                "content": content,
            }));
        }
        "user" => {
            let content = object
                .get("content")
                .map(convert_responses_user_content)
                .transpose()?
                .unwrap_or_else(|| Value::String(String::new()));
            output.push(json!({
                "role": "user",
                "content": content,
            }));
        }
        unsupported => {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("unsupported Responses input role `{unsupported}`"),
            ))
        }
    }

    Ok(())
}

fn convert_responses_user_content(content: &Value) -> Result<Value, ApiError> {
    match content {
        Value::Null => Ok(Value::String(String::new())),
        Value::String(text) => Ok(Value::String(text.clone())),
        Value::Array(items) => {
            let mut parts = Vec::new();
            for item in items {
                match item {
                    Value::String(text) => push_text_content_part(text, &mut parts),
                    Value::Object(object) => match object.get("type").and_then(Value::as_str) {
                        Some("input_text" | "output_text" | "text") => {
                            if let Some(text) = object
                                .get("text")
                                .or_else(|| object.get("content"))
                                .and_then(Value::as_str)
                            {
                                push_text_content_part(text, &mut parts);
                            }
                        }
                        Some("input_image" | "image_url") => {
                            if let Some(part) = convert_responses_image_part(object) {
                                parts.push(part);
                            } else {
                                push_text_content_part(
                                    "[responses input_image omitted by Prism]",
                                    &mut parts,
                                );
                            }
                        }
                        Some("input_file" | "file") => push_text_content_part(
                            "[responses file input omitted by Prism]",
                            &mut parts,
                        ),
                        Some(other) => push_text_content_part(
                            &format!("[responses {other} item forwarded as note]"),
                            &mut parts,
                        ),
                        None => {
                            if let Some(text) = object
                                .get("text")
                                .or_else(|| object.get("content"))
                                .and_then(Value::as_str)
                            {
                                push_text_content_part(text, &mut parts);
                            }
                        }
                    },
                    _ => {}
                }
            }

            if parts
                .iter()
                .all(|part| matches!(part.get("type"), Some(Value::String(kind)) if kind == "text"))
            {
                Ok(Value::String(
                    parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n\n"),
                ))
            } else {
                Ok(Value::Array(parts))
            }
        }
        other => Ok(Value::String(other.to_string())),
    }
}

fn extract_responses_assistant_content(content: &Value) -> Result<Value, ApiError> {
    match content {
        Value::Null => Ok(Value::Null),
        Value::String(text) => Ok(Value::String(text.clone())),
        Value::Array(items) => {
            let mut text = String::new();
            for item in items {
                match item {
                    Value::String(value) => text.push_str(value),
                    Value::Object(object) => {
                        if let Some(value) = object
                            .get("text")
                            .or_else(|| object.get("content"))
                            .and_then(Value::as_str)
                        {
                            text.push_str(value);
                        }
                    }
                    _ => {}
                }
            }

            if text.is_empty() {
                Ok(Value::Null)
            } else {
                Ok(Value::String(text))
            }
        }
        other => Ok(Value::String(other.to_string())),
    }
}

fn extract_responses_text(value: &Value) -> Result<String, ApiError> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(items) => {
            let mut parts = Vec::new();
            for item in items {
                match item {
                    Value::String(text) => parts.push(text.clone()),
                    Value::Object(object) => match object.get("type").and_then(Value::as_str) {
                        Some("input_text" | "output_text" | "text") => {
                            if let Some(text) = object
                                .get("text")
                                .or_else(|| object.get("content"))
                                .and_then(Value::as_str)
                            {
                                parts.push(text.to_string());
                            }
                        }
                        Some("message") => {
                            if let Some(content) = object.get("content") {
                                let nested = extract_responses_text(content)?;
                                if !nested.is_empty() {
                                    parts.push(nested);
                                }
                            }
                        }
                        Some("item_reference") => {
                            return Err(ApiError::new(
                                StatusCode::BAD_REQUEST,
                                "invalid_request_error",
                                "Responses `item_reference` is not supported by Prism's chat-completions bridge yet",
                            ))
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            Ok(parts.join("\n\n"))
        }
        Value::Object(object) => {
            if let Some(text) = object
                .get("text")
                .or_else(|| object.get("content"))
                .and_then(Value::as_str)
            {
                Ok(text.to_string())
            } else if let Some(content) = object.get("content") {
                extract_responses_text(content)
            } else {
                Ok(value.to_string())
            }
        }
        other => Ok(other.to_string()),
    }
}

fn convert_responses_image_part(object: &Map<String, Value>) -> Option<Value> {
    if let Some(image_url) = object.get("image_url") {
        if let Some(url) = image_url.as_str() {
            return Some(json!({
                "type": "image_url",
                "image_url": {
                    "url": url
                }
            }));
        }

        if let Some(url) = image_url.get("url").and_then(Value::as_str) {
            return Some(json!({
                "type": "image_url",
                "image_url": {
                    "url": url
                }
            }));
        }
    }

    if let Some(url) = object.get("url").and_then(Value::as_str) {
        return Some(json!({
            "type": "image_url",
            "image_url": {
                "url": url
            }
        }));
    }

    None
}

fn convert_responses_tools(tools: &Value) -> Result<Value, ApiError> {
    let items = tools.as_array().ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Responses `tools` must be an array",
        )
    })?;

    let converted = items
        .iter()
        .map(|tool| {
            let object = tool.as_object().ok_or_else(|| {
                ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "Responses tool entries must be objects",
                )
            })?;

            let tool_type = object
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("function");

            if tool_type != "function" {
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    format!(
                        "Responses built-in tool `{tool_type}` is not supported by Prism's chat-completions bridge yet"
                    ),
                ));
            }

            let name = object
                .get("name")
                .or_else(|| object.get("function").and_then(|value| value.get("name")))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ApiError::new(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        "Responses function tool requires `name`",
                    )
                })?;

            let description = object
                .get("description")
                .or_else(|| object.get("function").and_then(|value| value.get("description")))
                .cloned()
                .unwrap_or(Value::Null);

            let parameters = object
                .get("parameters")
                .or_else(|| object.get("input_schema"))
                .or_else(|| object.get("function").and_then(|value| value.get("parameters")))
                .cloned()
                .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));

            Ok(json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": normalize_schema(parameters)?,
                }
            }))
        })
        .collect::<Result<Vec<_>, ApiError>>()?;

    Ok(Value::Array(converted))
}

fn convert_responses_tool_choice(tool_choice: &Value) -> Result<Value, ApiError> {
    match tool_choice {
        Value::String(value) => match value.as_str() {
            "auto" | "none" | "required" => Ok(Value::String(value.clone())),
            other => Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("unsupported Responses tool_choice `{other}`"),
            )),
        },
        Value::Object(object) => {
            let kind = object
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("function");
            if kind != "function" {
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    format!(
                        "Responses tool_choice type `{kind}` is not supported by Prism's chat-completions bridge yet"
                    ),
                ));
            }

            let name = object.get("name").and_then(Value::as_str).ok_or_else(|| {
                ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "Responses function tool_choice requires `name`",
                )
            })?;

            Ok(json!({
                "type": "function",
                "function": {
                    "name": name,
                }
            }))
        }
        _ => Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "unsupported Responses `tool_choice` shape",
        )),
    }
}

fn convert_responses_text_format(text: &Value) -> Result<Option<Value>, ApiError> {
    let Some(format) = text.get("format") else {
        return Ok(None);
    };

    let object = format.as_object().ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Responses `text.format` must be an object",
        )
    })?;

    let kind = object.get("type").and_then(Value::as_str).ok_or_else(|| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Responses `text.format` requires `type`",
        )
    })?;

    match kind {
        "json_object" => Ok(Some(json!({ "type": "json_object" }))),
        "json_schema" => {
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("prism_schema");
            let schema = object.get("schema").cloned().unwrap_or_else(|| json!({}));
            let strict = object
                .get("strict")
                .cloned()
                .unwrap_or_else(|| json!(false));

            Ok(Some(json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                    "strict": strict,
                }
            })))
        }
        other => Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("unsupported Responses text format `{other}`"),
        )),
    }
}

fn append_anthropic_message(
    message: AnthropicMessage,
    output: &mut Vec<Value>,
) -> Result<(), ApiError> {
    let blocks = message.content.into_blocks();

    match message.role.as_str() {
        "user" => {
            let mut content_parts = Vec::new();

            for block in blocks {
                match block.kind.as_str() {
                    "text" => {
                        let text = block.field_str("text").ok_or_else(|| {
                            ApiError::new(
                                StatusCode::BAD_REQUEST,
                                "invalid_request_error",
                                "text block is missing `text`",
                            )
                        })?;
                        push_text_content_part(text, &mut content_parts);
                    }
                    "image" => {
                        if let Some(part) = convert_image_block_to_openai_part(&block) {
                            content_parts.push(part);
                        } else if let Some(fallback) = fallback_user_block_text(&block) {
                            push_text_content_part(&fallback, &mut content_parts);
                        }
                    }
                    "document" => {
                        if let Some(parts) = expand_document_block_user_parts(&block) {
                            for part in parts {
                                match part.get("type").and_then(Value::as_str) {
                                    Some("text") => {
                                        if let Some(text) = part.get("text").and_then(Value::as_str)
                                        {
                                            push_text_content_part(text, &mut content_parts);
                                        }
                                    }
                                    _ => content_parts.push(part),
                                }
                            }
                        } else if let Some(fallback) = fallback_user_block_text(&block) {
                            push_text_content_part(&fallback, &mut content_parts);
                        }
                    }
                    "tool_result" => {
                        flush_user_content(&mut content_parts, output);
                        let tool_use_id = block.field_str("tool_use_id").ok_or_else(|| {
                            ApiError::new(
                                StatusCode::BAD_REQUEST,
                                "invalid_request_error",
                                "tool_result block is missing `tool_use_id`",
                            )
                        })?;
                        let content = extract_tool_result_text(block.field_value("content"))?;
                        output.push(json!({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": content,
                        }));
                    }
                    "thinking" | "redacted_thinking" => {}
                    _ => {
                        if let Some(fallback) = fallback_user_block_text(&block) {
                            push_text_content_part(&fallback, &mut content_parts);
                        }
                    }
                }
            }

            flush_user_content(&mut content_parts, output);
        }
        "assistant" => {
            let mut text_parts = Vec::new();
            let mut reasoning_parts = Vec::new();
            let mut tool_calls = Vec::new();

            for block in blocks {
                match block.kind.as_str() {
                    "text" => {
                        let text = block.field_str("text").ok_or_else(|| {
                            ApiError::new(
                                StatusCode::BAD_REQUEST,
                                "invalid_request_error",
                                "text block is missing `text`",
                            )
                        })?;
                        text_parts.push(text.to_string());
                    }
                    "tool_use" => {
                        let id = block.field_str("id").ok_or_else(|| {
                            ApiError::new(
                                StatusCode::BAD_REQUEST,
                                "invalid_request_error",
                                "tool_use block is missing `id`",
                            )
                        })?;
                        let name = block.field_str("name").ok_or_else(|| {
                            ApiError::new(
                                StatusCode::BAD_REQUEST,
                                "invalid_request_error",
                                "tool_use block is missing `name`",
                            )
                        })?;
                        let arguments =
                            serde_json::to_string(block.field_value("input").unwrap_or(&json!({})))
                                .map_err(|error| {
                                    ApiError::new(
                                        StatusCode::BAD_REQUEST,
                                        "invalid_request_error",
                                        format!("failed to encode tool input: {error}"),
                                    )
                                })?;

                        tool_calls.push(json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            }
                        }));
                    }
                    "thinking" => {
                        if let Some(thinking) = extract_reasoning_text(&block) {
                            reasoning_parts.push(thinking);
                        }
                    }
                    "redacted_thinking" => {
                        // Signal to downstream that a reasoning block was
                        // present but redacted — better than a silent drop.
                        reasoning_parts.push("[redacted thinking]".to_string());
                    }
                    _ => {
                        if let Some(fallback) = fallback_assistant_block_text(&block) {
                            text_parts.push(fallback);
                        }
                    }
                }
            }

            if text_parts.is_empty() && tool_calls.is_empty() && reasoning_parts.is_empty() {
                return Ok(());
            }

            let mut assistant_message = serde_json::Map::new();
            assistant_message.insert("role".into(), Value::String("assistant".into()));
            assistant_message.insert(
                "content".into(),
                if text_parts.is_empty() {
                    Value::Null
                } else {
                    Value::String(text_parts.join("\n\n"))
                },
            );
            if !reasoning_parts.is_empty() {
                assistant_message.insert(
                    "reasoning_content".into(),
                    Value::String(reasoning_parts.join("\n\n")),
                );
            }
            if !tool_calls.is_empty() {
                assistant_message.insert("tool_calls".into(), Value::Array(tool_calls));
            }

            output.push(Value::Object(assistant_message));
        }
        unsupported => {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("unsupported Anthropic role `{unsupported}`"),
            ))
        }
    }

    Ok(())
}

fn push_text_content_part(text: &str, content_parts: &mut Vec<Value>) {
    if text.is_empty() {
        return;
    }

    content_parts.push(json!({
        "type": "text",
        "text": text,
    }));
}

fn flush_user_content(content_parts: &mut Vec<Value>, output: &mut Vec<Value>) {
    if content_parts.is_empty() {
        return;
    }

    let content = if content_parts
        .iter()
        .all(|part| matches!(part.get("type"), Some(Value::String(kind)) if kind == "text"))
    {
        Value::String(
            content_parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n\n"),
        )
    } else {
        Value::Array(content_parts.clone())
    };

    output.push(json!({
        "role": "user",
        "content": content,
    }));
    content_parts.clear();
}

fn extract_system_prompt(prompt: AnthropicSystemPrompt) -> Result<String, ApiError> {
    match prompt {
        AnthropicSystemPrompt::Text(text) => Ok(text),
        AnthropicSystemPrompt::Blocks(blocks) => join_text_blocks(&blocks),
    }
}

fn extract_tool_result_text(content: Option<&Value>) -> Result<String, ApiError> {
    let Some(content) = content else {
        return Ok(String::new());
    };

    match content {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(items) => {
            let mut parts = Vec::new();
            for item in items {
                let block: AnthropicBlock =
                    serde_json::from_value(item.clone()).map_err(|error| {
                        ApiError::new(
                            StatusCode::BAD_REQUEST,
                            "invalid_request_error",
                            format!("invalid tool_result content block: {error}"),
                        )
                    })?;

                if let Some(text) = tool_result_block_to_text(&block) {
                    parts.push(text);
                }
            }
            Ok(parts.join("\n\n"))
        }
        other => Ok(other.to_string()),
    }
}

fn join_text_blocks(blocks: &[AnthropicBlock]) -> Result<String, ApiError> {
    let mut parts = Vec::new();

    for block in blocks {
        match block.kind.as_str() {
            "text" => {
                let text = block.field_str("text").ok_or_else(|| {
                    ApiError::new(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        "system text block is missing `text`",
                    )
                })?;
                parts.push(text.to_string());
            }
            "thinking" | "redacted_thinking" => {}
            _ => {
                if let Some(fallback) = fallback_user_block_text(block) {
                    parts.push(fallback);
                }
            }
        }
    }

    Ok(parts.join("\n\n"))
}

fn convert_image_block_to_openai_part(block: &AnthropicBlock) -> Option<Value> {
    let source = block.field_value("source")?.as_object()?;
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("image/png");
            let data = source.get("data").and_then(Value::as_str)?;
            Some(json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{media_type};base64,{data}")
                }
            }))
        }
        Some("url") => {
            let url = source.get("url").and_then(Value::as_str)?;
            Some(json!({
                "type": "image_url",
                "image_url": {
                    "url": url
                }
            }))
        }
        _ => None,
    }
}

fn fallback_user_block_text(block: &AnthropicBlock) -> Option<String> {
    match block.kind.as_str() {
        "thinking" | "redacted_thinking" => None,
        "document" => {
            if let Some(text) = document_block_to_text(block) {
                return Some(text);
            }
            Some(format!(
                "[anthropic document block omitted by Prism: {}]",
                compact_json_preview(&block.fields)
            ))
        }
        "audio" | "video" | "file" => Some(format!(
            "[anthropic {} block omitted by Prism: {}]",
            block.kind,
            compact_json_preview(&block.fields)
        )),
        _ => fallback_block_text(block),
    }
}

/// Extract a human-readable preamble (title / context) for a `document` block.
/// Returns `None` when there's nothing worth prepending.
fn document_block_preamble(block: &AnthropicBlock) -> Option<String> {
    let title = block
        .field_str("title")
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let context = block
        .field_str("context")
        .map(str::trim)
        .filter(|s| !s.is_empty());

    match (title, context) {
        (None, None) => None,
        (Some(t), None) => Some(format!("[document: {t}]")),
        (None, Some(c)) => Some(format!("[document context: {c}]")),
        (Some(t), Some(c)) => Some(format!("[document: {t}]\ncontext: {c}")),
    }
}

/// Expand an Anthropic `document` block into OpenAI content parts when the
/// source type is `text` or `content`. Returns `None` for binary sources
/// (`base64`, `url`, `file`) — those fall back to the omitted-note path.
fn expand_document_block_user_parts(block: &AnthropicBlock) -> Option<Vec<Value>> {
    let source = block.field_value("source")?.as_object()?;
    let source_type = source.get("type").and_then(Value::as_str)?;
    let preamble = document_block_preamble(block);

    match source_type {
        "text" => {
            let data = source.get("data").and_then(Value::as_str)?;
            let text = match preamble {
                Some(p) => format!("{p}\n{data}"),
                None => data.to_string(),
            };
            Some(vec![json!({ "type": "text", "text": text })])
        }
        "content" => {
            let inner = source.get("content")?;
            let mut parts: Vec<Value> = Vec::new();
            if let Some(p) = preamble {
                parts.push(json!({ "type": "text", "text": p }));
            }
            match inner {
                Value::String(text) => {
                    parts.push(json!({ "type": "text", "text": text }));
                }
                Value::Array(items) => {
                    for item in items {
                        let Ok(inner_block) =
                            serde_json::from_value::<AnthropicBlock>(item.clone())
                        else {
                            continue;
                        };
                        match inner_block.kind.as_str() {
                            "text" => {
                                if let Some(t) = inner_block.field_str("text") {
                                    parts.push(json!({ "type": "text", "text": t }));
                                }
                            }
                            "image" => {
                                if let Some(p) = convert_image_block_to_openai_part(&inner_block) {
                                    parts.push(p);
                                } else if let Some(fb) = fallback_user_block_text(&inner_block) {
                                    parts.push(json!({ "type": "text", "text": fb }));
                                }
                            }
                            _ => {
                                if let Some(fb) = fallback_user_block_text(&inner_block) {
                                    parts.push(json!({ "type": "text", "text": fb }));
                                }
                            }
                        }
                    }
                }
                _ => return None,
            }
            if parts.is_empty() {
                None
            } else {
                Some(parts)
            }
        }
        _ => None,
    }
}

/// Flatten a `document` block to text, for paths that don't accept multi-modal
/// parts (system prompt, assistant fallback, tool_result). Returns `None` for
/// sources we can't meaningfully render as text.
fn document_block_to_text(block: &AnthropicBlock) -> Option<String> {
    let parts = expand_document_block_user_parts(block)?;
    let mut out = Vec::new();
    for part in parts {
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            out.push(text.to_string());
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out.join("\n"))
    }
}

fn fallback_assistant_block_text(block: &AnthropicBlock) -> Option<String> {
    match block.kind.as_str() {
        "thinking" | "redacted_thinking" => None,
        _ => fallback_block_text(block),
    }
}

fn fallback_block_text(block: &AnthropicBlock) -> Option<String> {
    if let Some(text) = block.field_str("text") {
        return Some(text.to_string());
    }

    if let Some(thinking) = extract_reasoning_text(block) {
        return Some(format!(
            "[anthropic {} block forwarded as note]\n{}",
            block.kind, thinking
        ));
    }

    Some(format!(
        "[anthropic {} block forwarded as note: {}]",
        block.kind,
        compact_json_preview(&block.fields)
    ))
}

fn extract_reasoning_text(block: &AnthropicBlock) -> Option<String> {
    block
        .field_str("thinking")
        .or_else(|| block.field_str("text"))
        .map(ToString::to_string)
}

fn tool_result_block_to_text(block: &AnthropicBlock) -> Option<String> {
    match block.kind.as_str() {
        "text" => block.field_str("text").map(ToString::to_string),
        "thinking" | "redacted_thinking" => None,
        _ => fallback_user_block_text(block),
    }
}

fn compact_json_preview(value: &impl Serialize) -> String {
    let json = serde_json::to_string(value).unwrap_or_else(|_| "{}".into());
    if json.chars().count() > 240 {
        let mut preview = json.chars().take(240).collect::<String>();
        preview.push_str("...");
        preview
    } else {
        json
    }
}
