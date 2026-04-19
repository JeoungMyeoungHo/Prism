//! Anthropic → OpenAI tool schema conversion.

use super::ApiError;
use crate::types::{AnthropicTool, AnthropicToolChoice};
use axum::http::StatusCode;
use serde_json::{json, Value};

pub(super) fn convert_tools(tools: Vec<AnthropicTool>) -> Result<Value, ApiError> {
    let tools = tools
        .into_iter()
        .map(|tool| {
            Ok(json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": normalize_schema(tool.input_schema)?,
                }
            }))
        })
        .collect::<Result<Vec<_>, ApiError>>()?;

    Ok(Value::Array(tools))
}

pub(super) fn convert_tool_choice(tool_choice: AnthropicToolChoice) -> Result<Value, ApiError> {
    match tool_choice.kind.as_str() {
        "auto" => Ok(Value::String("auto".into())),
        "any" => Ok(Value::String("required".into())),
        "none" => Ok(Value::String("none".into())),
        "tool" => Ok(json!({
            "type": "function",
            "function": {
                "name": tool_choice.name.ok_or_else(|| {
                    ApiError::new(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        "tool choice `tool` requires a `name`",
                    )
                })?
            }
        })),
        unsupported => Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("unsupported tool choice `{unsupported}`"),
        )),
    }
}

pub(super) fn normalize_schema(schema: Value) -> Result<Value, ApiError> {
    match schema {
        Value::Object(mut object) => {
            object
                .entry("type")
                .or_insert_with(|| Value::String("object".into()));
            Ok(Value::Object(object))
        }
        Value::Null => Ok(json!({ "type": "object", "properties": {} })),
        other => Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("tool input_schema must be a JSON object, got {other}"),
        )),
    }
}
