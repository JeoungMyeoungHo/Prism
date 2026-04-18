use crate::{
    router::ModelRouter,
    types::{
        AnthropicBlock, AnthropicMessage, AnthropicMessagesRequest, AnthropicSystemPrompt,
        AnthropicTool, AnthropicToolChoice, Backend, OpenAiChatCompletionChunk,
        OpenAiChatCompletionResponse, OpenAiUsage,
    },
};
use async_stream::stream;
use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{
        header::{CACHE_CONTROL, CONNECTION, CONTENT_TYPE},
        HeaderMap, HeaderName, HeaderValue, StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use futures_util::TryStreamExt;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::{
    collections::BTreeMap,
    convert::Infallible,
    io,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_util::io::StreamReader;
use tracing::info;

#[derive(Clone)]
pub struct AppState {
    client: reqwest::Client,
    router: ModelRouter,
}

impl AppState {
    pub fn new(router: ModelRouter) -> Self {
        Self {
            client: reqwest::Client::new(),
            router,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ApiError {
    status: StatusCode,
    kind: String,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status,
            kind: kind.into(),
            message: message.into(),
        }
    }

    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "type": "error",
                "error": {
                    "type": self.kind,
                    "message": self.message,
                }
            })),
        )
            .into_response()
    }
}

pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

fn backend_label(backend: &Backend) -> String {
    format!("prefix={}", backend.prefix)
}

pub async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let request: AnthropicMessagesRequest = serde_json::from_slice(&body).map_err(|error| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("invalid Anthropic request body: {error}"),
        )
    })?;

    let (backend, upstream_model, matched_by) = {
        let resolution = state.router.resolve(&request.model).ok_or_else(|| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!(
                    "no route matches model `{}`. catalog: {}",
                    request.model,
                    state.router.describe_catalog()
                ),
            )
        })?;
        (
            resolution.backend.clone(),
            resolution.upstream_model.clone(),
            resolution.matched_by,
        )
    };

    info!(
        "routing model `{}` -> upstream `{}` via {} [{}] ({} match, credential source: {})",
        request.model,
        upstream_model,
        backend_label(&backend),
        backend.provider.as_str(),
        matched_by.as_str(),
        backend.credential_label
    );

    let forwarded_headers = collect_upstream_headers(&headers);
    if backend.anthropic_format {
        // `anthropic_format = true` route — relay the original Messages
        // request byte-for-byte after rewriting `model`. No format
        // translation, no provider adapter body tweaks.
        forward_anthropic_passthrough(
            &state.client,
            &backend,
            &forwarded_headers,
            &body,
            upstream_model,
        )
        .await
    } else {
        forward_request_to_backend(
            &state.client,
            &backend,
            &forwarded_headers,
            request,
            upstream_model,
        )
        .await
    }
}

pub async fn openai_responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let request: Value = serde_json::from_slice(&body).map_err(|error| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("invalid Responses request body: {error}"),
        )
    })?;

    let model = request
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "Responses request requires a string `model` field",
            )
        })?
        .to_string();

    let (backend, upstream_model, matched_by) = {
        let resolution = state.router.resolve(&model).ok_or_else(|| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!(
                    "no route matches model `{model}`. catalog: {}",
                    state.router.describe_catalog()
                ),
            )
        })?;
        (
            resolution.backend.clone(),
            resolution.upstream_model.clone(),
            resolution.matched_by,
        )
    };

    info!(
        "routing Responses model `{}` -> upstream `{}` via {} [{}] ({} match, credential source: {})",
        model,
        upstream_model,
        backend_label(&backend),
        backend.provider.as_str(),
        matched_by.as_str(),
        backend.credential_label
    );

    let forwarded_headers = collect_upstream_headers(&headers);
    forward_responses_request_to_backend(
        &state.client,
        &backend,
        &forwarded_headers,
        request,
        upstream_model,
    )
    .await
}

/// Forward the inbound Anthropic Messages request to a route flagged with
/// `anthropic_format = true`. The only mutation Prism performs is rewriting
/// the `model` field; everything else — unknown fields, system blocks, tool
/// definitions, SSE events — is relayed verbatim. Auth uses `x-api-key` +
/// `anthropic-version` (the Anthropic Messages API convention) regardless of
/// the route's provider adapter.
async fn forward_anthropic_passthrough(
    client: &reqwest::Client,
    backend: &Backend,
    forwarded_headers: &[(HeaderName, HeaderValue)],
    raw_body: &Bytes,
    upstream_model: String,
) -> Result<Response, ApiError> {
    let mut value: Value = serde_json::from_slice(raw_body).map_err(|error| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("invalid Anthropic request body: {error}"),
        )
    })?;
    let stream = value
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if let Some(object) = value.as_object_mut() {
        object.insert("model".into(), Value::String(upstream_model));
    }

    let upstream_url = backend
        .base
        .join("messages")
        .expect("normalized backend base URL");

    let mut request_builder = client
        .post(upstream_url)
        .header("x-api-key", &backend.api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&value);
    if stream {
        request_builder = request_builder.header(reqwest::header::ACCEPT, "text/event-stream");
    }
    for (name, value) in forwarded_headers {
        request_builder = request_builder.header(name, value);
    }

    let response = request_builder.send().await.map_err(|error| {
        ApiError::new(
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            format!(
                "failed to reach upstream backend `{}`: {error}",
                backend_label(backend)
            ),
        )
    })?;

    if !response.status().is_success() {
        return Err(read_upstream_error(response).await);
    }

    if stream {
        // Anthropic-native SSE passes through as-is; the client already
        // speaks Anthropic's event shape.
        let byte_stream = response
            .bytes_stream()
            .map_err(|error| io::Error::new(io::ErrorKind::Other, error));
        let mut out = Response::new(Body::from_stream(byte_stream));
        out.headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        out.headers_mut()
            .insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        out.headers_mut()
            .insert(CONNECTION, HeaderValue::from_static("keep-alive"));
        Ok(out)
    } else {
        let bytes = response.bytes().await.map_err(|error| {
            ApiError::new(
                StatusCode::BAD_GATEWAY,
                "upstream_error",
                format!("failed to read upstream response: {error}"),
            )
        })?;
        let mut out = Response::new(Body::from(bytes));
        out.headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(out)
    }
}

pub async fn forward_request_to_backend(
    client: &reqwest::Client,
    backend: &Backend,
    forwarded_headers: &[(HeaderName, HeaderValue)],
    mut request: AnthropicMessagesRequest,
    upstream_model: String,
) -> Result<Response, ApiError> {
    let requested_model = request.model.clone();
    let stream = request.stream.unwrap_or(false);
    request.model = upstream_model;
    let prepared = anthropic_request_to_openai(request, backend)?;

    for note in &prepared.adapter_notes {
        info!(
            "provider adapter `{}` note for model `{}`: {}",
            backend.provider.as_str(),
            requested_model,
            note
        );
    }

    if stream {
        proxy_streaming(
            client,
            backend,
            forwarded_headers,
            prepared.body,
            requested_model,
        )
        .await
    } else {
        proxy_non_streaming(
            client,
            backend,
            forwarded_headers,
            prepared.body,
            requested_model,
        )
        .await
    }
}

async fn forward_responses_request_to_backend(
    client: &reqwest::Client,
    backend: &Backend,
    forwarded_headers: &[(HeaderName, HeaderValue)],
    mut request: Value,
    upstream_model: String,
) -> Result<Response, ApiError> {
    let requested_model = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let stream = request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if let Some(obj) = request.as_object_mut() {
        obj.insert("model".into(), Value::String(upstream_model));
    }
    let prepared = responses_request_to_openai_chat(request, backend)?;

    for note in &prepared.adapter_notes {
        info!(
            "provider adapter `{}` note for Responses model `{}`: {}",
            backend.provider.as_str(),
            requested_model,
            note
        );
    }

    if stream {
        Err(ApiError::new(
            StatusCode::NOT_IMPLEMENTED,
            "unsupported_feature",
            "streaming `/v1/responses` translation is not implemented yet; use non-streaming Responses or `/v1/messages` streaming for now",
        ))
    } else {
        proxy_responses_non_streaming(
            client,
            backend,
            forwarded_headers,
            prepared.body,
            requested_model,
        )
        .await
    }
}

async fn proxy_non_streaming(
    client: &reqwest::Client,
    backend: &Backend,
    forwarded_headers: &[(HeaderName, HeaderValue)],
    upstream_body: Value,
    requested_model: String,
) -> Result<Response, ApiError> {
    let adapter = backend.provider.adapter();
    let mut request_builder = adapter.apply_auth(
        client.post(adapter.chat_completions_url(&backend.base)),
        &backend.api_key,
    );
    request_builder = request_builder.json(&upstream_body);

    for (name, value) in forwarded_headers {
        request_builder = request_builder.header(name, value);
    }

    let response = request_builder.send().await.map_err(|error| {
        ApiError::new(
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            format!(
                "failed to reach upstream backend `{}`: {error}",
                backend_label(backend)
            ),
        )
    })?;

    if !response.status().is_success() {
        return Err(read_upstream_error(response).await);
    }

    let openai_response: OpenAiChatCompletionResponse = response.json().await.map_err(|error| {
        ApiError::new(
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            format!("failed to decode upstream OpenAI response: {error}"),
        )
    })?;

    let anthropic_response = openai_response_to_anthropic(openai_response, &requested_model)?;
    Ok(Json(anthropic_response).into_response())
}

async fn proxy_responses_non_streaming(
    client: &reqwest::Client,
    backend: &Backend,
    forwarded_headers: &[(HeaderName, HeaderValue)],
    upstream_body: Value,
    requested_model: String,
) -> Result<Response, ApiError> {
    let adapter = backend.provider.adapter();
    let mut request_builder = adapter.apply_auth(
        client.post(adapter.chat_completions_url(&backend.base)),
        &backend.api_key,
    );
    request_builder = request_builder.json(&upstream_body);

    for (name, value) in forwarded_headers {
        request_builder = request_builder.header(name, value);
    }

    let response = request_builder.send().await.map_err(|error| {
        ApiError::new(
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            format!(
                "failed to reach upstream backend `{}`: {error}",
                backend_label(backend)
            ),
        )
    })?;

    if !response.status().is_success() {
        return Err(read_upstream_error(response).await);
    }

    let openai_response: OpenAiChatCompletionResponse = response.json().await.map_err(|error| {
        ApiError::new(
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            format!("failed to decode upstream OpenAI response: {error}"),
        )
    })?;

    let response_object = openai_chat_response_to_responses(openai_response, &requested_model)?;
    Ok(Json(response_object).into_response())
}

async fn proxy_streaming(
    client: &reqwest::Client,
    backend: &Backend,
    forwarded_headers: &[(HeaderName, HeaderValue)],
    upstream_body: Value,
    requested_model: String,
) -> Result<Response, ApiError> {
    let adapter = backend.provider.adapter();
    let mut request_builder = adapter.apply_auth(
        client.post(adapter.chat_completions_url(&backend.base)),
        &backend.api_key,
    );
    request_builder = request_builder
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .json(&upstream_body);

    for (name, value) in forwarded_headers {
        request_builder = request_builder.header(name, value);
    }

    let response = request_builder.send().await.map_err(|error| {
        ApiError::new(
            StatusCode::BAD_GATEWAY,
            "upstream_error",
            format!("failed to open upstream stream: {error}"),
        )
    })?;

    if !response.status().is_success() {
        return Err(read_upstream_error(response).await);
    }

    let stream = stream! {
        let bytes_stream = response
            .bytes_stream()
            .map_err(|error| io::Error::new(io::ErrorKind::Other, error));
        let reader = StreamReader::new(bytes_stream);
        let mut lines = BufReader::new(reader).lines();
        let mut translator = AnthropicStreamTranslator::new(requested_model);
        let mut finished = false;

        loop {
            let line = match lines.next_line().await {
                Ok(Some(line)) => line,
                Ok(None) => break,
                Err(error) => {
                    yield Ok::<Bytes, Infallible>(sse_event(
                        "error",
                        &json!({
                            "type": "error",
                            "error": {
                                "type": "upstream_stream_error",
                                "message": error.to_string(),
                            }
                        }),
                    ));
                    finished = true;
                    break;
                }
            };

            if line.is_empty() || line.starts_with(':') || line.starts_with("event:") {
                continue;
            }

            let Some(raw_data) = line.strip_prefix("data:") else {
                continue;
            };

            let payload = raw_data.trim();
            if payload == "[DONE]" {
                for chunk in translator.finish() {
                    yield Ok::<Bytes, Infallible>(chunk);
                }
                finished = true;
                break;
            }

            let value: Value = match serde_json::from_str(payload) {
                Ok(value) => value,
                Err(error) => {
                    yield Ok::<Bytes, Infallible>(sse_event(
                        "error",
                        &json!({
                            "type": "error",
                            "error": {
                                "type": "invalid_upstream_payload",
                                "message": error.to_string(),
                            }
                        }),
                    ));
                    finished = true;
                    break;
                }
            };

            if let Some(error) = value.get("error") {
                yield Ok::<Bytes, Infallible>(sse_event(
                    "error",
                    &json!({
                        "type": "error",
                        "error": {
                            "type": "upstream_error",
                            "message": extract_error_message(error),
                        }
                    }),
                ));
                finished = true;
                break;
            }

            let chunk: OpenAiChatCompletionChunk = match serde_json::from_value(value) {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Ok::<Bytes, Infallible>(sse_event(
                        "error",
                        &json!({
                            "type": "error",
                            "error": {
                                "type": "invalid_upstream_payload",
                                "message": error.to_string(),
                            }
                        }),
                    ));
                    finished = true;
                    break;
                }
            };

            for event in translator.push(chunk) {
                yield Ok::<Bytes, Infallible>(event);
            }
        }

        if !finished {
            for chunk in translator.finish() {
                yield Ok::<Bytes, Infallible>(chunk);
            }
        }
    };

    let mut response = Response::new(Body::from_stream(stream));
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
        .headers_mut()
        .insert(CONNECTION, HeaderValue::from_static("keep-alive"));
    Ok(response)
}

struct PreparedUpstreamRequest {
    body: Value,
    adapter_notes: Vec<String>,
}

fn anthropic_request_to_openai(
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

    let object = openai_request
        .as_object_mut()
        .expect("request root should always be an object");

    if let Some(max_tokens) = request.max_tokens {
        object.insert("max_completion_tokens".into(), json!(max_tokens));
        object.insert("max_tokens".into(), json!(max_tokens));
    }

    if let Some(stream) = request.stream {
        object.insert("stream".into(), json!(stream));
        if stream {
            object.insert("stream_options".into(), json!({ "include_usage": true }));
        }
    }

    if let Some(temperature) = request.temperature {
        object.insert("temperature".into(), json!(temperature));
    }

    if let Some(top_p) = request.top_p {
        object.insert("top_p".into(), json!(top_p));
    }

    if let Some(stop_sequences) = request.stop_sequences {
        object.insert("stop".into(), json!(stop_sequences));
    }

    if let Some(tools) = request.tools {
        object.insert("tools".into(), convert_tools(tools)?);
    }

    if let Some(tool_choice) = request.tool_choice {
        object.insert("tool_choice".into(), convert_tool_choice(tool_choice)?);
    }

    let adapter_notes = backend.provider.adapter().adapt_request(object);

    Ok(PreparedUpstreamRequest {
        body: openai_request,
        adapter_notes,
    })
}

fn responses_request_to_openai_chat(
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

    let model = object
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| {
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

    let request_object = openai_request
        .as_object_mut()
        .expect("request root should always be an object");

    if let Some(max_output_tokens) = object.get("max_output_tokens").and_then(Value::as_u64) {
        request_object.insert("max_completion_tokens".into(), json!(max_output_tokens));
        request_object.insert("max_tokens".into(), json!(max_output_tokens));
    }

    if let Some(stream) = object.get("stream").and_then(Value::as_bool) {
        request_object.insert("stream".into(), json!(stream));
        if stream {
            request_object.insert("stream_options".into(), json!({ "include_usage": true }));
        }
    }

    if let Some(temperature) = object.get("temperature").and_then(Value::as_f64) {
        request_object.insert("temperature".into(), json!(temperature));
    }

    if let Some(top_p) = object.get("top_p").and_then(Value::as_f64) {
        request_object.insert("top_p".into(), json!(top_p));
    }

    if let Some(parallel_tool_calls) = object.get("parallel_tool_calls").and_then(Value::as_bool)
    {
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
    if matches!(item_type, Some("function_call_output" | "custom_tool_call_output")) {
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
        let name = object
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
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
                            if let Some(text) =
                                object.get("text").or_else(|| object.get("content")).and_then(Value::as_str)
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
                    parts.iter()
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

    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| {
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
            let strict = object.get("strict").cloned().unwrap_or_else(|| json!(false));

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
                    "redacted_thinking" => {}
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
        "document" | "audio" | "video" | "file" => Some(format!(
            "[anthropic {} block omitted by Prism: {}]",
            block.kind,
            compact_json_preview(&block.fields)
        )),
        _ => fallback_block_text(block),
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

fn convert_tools(tools: Vec<AnthropicTool>) -> Result<Value, ApiError> {
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

fn convert_tool_choice(tool_choice: AnthropicToolChoice) -> Result<Value, ApiError> {
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

fn normalize_schema(schema: Value) -> Result<Value, ApiError> {
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

fn openai_response_to_anthropic(
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

    Ok(json!({
        "id": response.id,
        "type": "message",
        "role": choice.message.role.unwrap_or_else(|| "assistant".into()),
        "model": response.model.unwrap_or_else(|| requested_model.to_string()),
        "content": content,
        "stop_reason": map_finish_reason(choice.finish_reason.as_deref()),
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": response.usage.as_ref().map(|usage| usage.prompt_tokens).unwrap_or_default(),
            "output_tokens": response.usage.as_ref().map(|usage| usage.completion_tokens).unwrap_or_default(),
        }
    }))
}

fn openai_chat_response_to_responses(
    response: OpenAiChatCompletionResponse,
    requested_model: &str,
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

    Ok(json!({
        "id": response.id,
        "object": "response",
        "created_at": created_at,
        "completed_at": created_at,
        "status": "completed",
        "model": response.model.unwrap_or_else(|| requested_model.to_string()),
        "output": output,
        "parallel_tool_calls": output.iter().filter(|item| item.get("type") == Some(&json!("function_call"))).count() > 1,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.prompt_tokens + usage.completion_tokens,
        }
    }))
}

fn extract_openai_text(content: Option<&Value>) -> Result<Option<String>, ApiError> {
    let Some(content) = content else {
        return Ok(None);
    };

    match content {
        Value::Null => Ok(None),
        Value::String(text) => Ok(Some(text.clone())),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                match part {
                    Value::String(value) => text.push_str(value),
                    Value::Object(object) => {
                        if let Some(value) = object.get("text").and_then(Value::as_str) {
                            text.push_str(value);
                        } else if let Some(value) = object.get("content").and_then(Value::as_str) {
                            text.push_str(value);
                        } else if let Some(value) =
                            object.get("reasoning_content").and_then(Value::as_str)
                        {
                            text.push_str(value);
                        }
                    }
                    _ => {}
                }
            }
            Ok(if text.is_empty() { None } else { Some(text) })
        }
        _ => Ok(None),
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

fn encode_tool_arguments(arguments: Value) -> String {
    match arguments {
        Value::String(text) => text,
        Value::Null => "{}".into(),
        other => serde_json::to_string(&other).unwrap_or_else(|_| "{}".into()),
    }
}

fn map_finish_reason(reason: Option<&str>) -> Value {
    match reason {
        Some("stop") => Value::String("end_turn".into()),
        Some("length") => Value::String("max_tokens".into()),
        Some("tool_calls") => Value::String("tool_use".into()),
        Some("content_filter") => Value::String("refusal".into()),
        Some(other) => Value::String(other.to_string()),
        None => Value::Null,
    }
}

async fn read_upstream_error(response: reqwest::Response) -> ApiError {
    let status =
        StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let body = response.text().await.unwrap_or_default();

    let message = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|value| value.get("error").cloned())
        .map(|error| extract_error_message(&error))
        .unwrap_or_else(|| {
            if body.is_empty() {
                format!("upstream request failed with HTTP {status}")
            } else {
                body
            }
        });

    ApiError::new(status, "upstream_error", message)
}

fn extract_error_message(error: &Value) -> String {
    error
        .get("message")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| error.to_string())
}

fn sse_event(event: &str, payload: &Value) -> Bytes {
    let json = serde_json::to_string(payload).unwrap_or_else(|_| "{}".into());
    Bytes::from(format!("event: {event}\ndata: {json}\n\n"))
}

fn collect_upstream_headers(headers: &HeaderMap) -> Vec<(HeaderName, HeaderValue)> {
    const PASSTHROUGH_HEADERS: [&str; 3] = [
        "anthropic-beta",
        "anthropic-version",
        "x-claude-code-session-id",
    ];

    PASSTHROUGH_HEADERS
        .into_iter()
        .filter_map(|name| {
            headers
                .get(name)
                .cloned()
                .map(|value| (HeaderName::from_static(name), value))
        })
        .collect()
}

#[derive(Default)]
struct ToolCallStreamState {
    id: String,
    name: String,
    started: bool,
}

struct AnthropicStreamTranslator {
    requested_model: String,
    response_id: Option<String>,
    model: Option<String>,
    emitted_message_start: bool,
    finished: bool,
    finish_reason: Option<String>,
    usage: OpenAiUsage,
    text_block_index: Option<usize>,
    text_block_open: bool,
    tool_blocks: BTreeMap<usize, ToolCallStreamState>,
}

impl AnthropicStreamTranslator {
    fn new(requested_model: String) -> Self {
        Self {
            requested_model,
            response_id: None,
            model: None,
            emitted_message_start: false,
            finished: false,
            finish_reason: None,
            usage: OpenAiUsage::default(),
            text_block_index: None,
            text_block_open: false,
            tool_blocks: BTreeMap::new(),
        }
    }

    fn push(&mut self, chunk: OpenAiChatCompletionChunk) -> Vec<Bytes> {
        let mut events = Vec::new();

        if let Some(id) = chunk.id {
            self.response_id = Some(id);
        }
        if let Some(model) = chunk.model {
            self.model = Some(model);
        }
        if let Some(usage) = chunk.usage {
            self.usage = usage;
        }

        if !self.emitted_message_start {
            events.push(sse_event(
                "message_start",
                &json!({
                    "type": "message_start",
                    "message": {
                        "id": self.response_id.clone().unwrap_or_else(fallback_message_id),
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model.clone().unwrap_or_else(|| self.requested_model.clone()),
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": {
                            "input_tokens": self.usage.prompt_tokens,
                            "output_tokens": 0,
                        }
                    }
                }),
            ));
            self.emitted_message_start = true;
        }

        for choice in chunk.choices {
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }

            if let Some(content) = choice.delta.content {
                if let Some(text) = extract_stream_text(&content) {
                    if !text.is_empty() {
                        let index = self.ensure_text_block(&mut events);
                        events.push(sse_event(
                            "content_block_delta",
                            &json!({
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "text_delta",
                                    "text": text,
                                }
                            }),
                        ));
                    }
                }
            }

            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    let tool_index = tool_call.index.unwrap_or(0);
                    let content_index =
                        compute_tool_content_index(self.text_block_index, tool_index);
                    let mut start_event = None;
                    let mut delta_event = None;

                    {
                        let state = self.tool_blocks.entry(tool_index).or_default();

                        if let Some(id) = tool_call.id {
                            state.id = id;
                        }

                        if let Some(function) = tool_call.function {
                            if let Some(name) = function.name {
                                state.name = name;
                            }

                            if !state.started {
                                if state.id.is_empty() {
                                    state.id = format!("toolu_{}", tool_index);
                                }
                                if state.name.is_empty() {
                                    state.name = format!("tool_{}", tool_index);
                                }

                                start_event = Some(json!({
                                    "type": "content_block_start",
                                    "index": content_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": state.id,
                                        "name": state.name,
                                        "input": {}
                                    }
                                }));
                                state.started = true;
                            }

                            if let Some(arguments) = function.arguments {
                                if !arguments.is_empty() {
                                    delta_event = Some(json!({
                                        "type": "content_block_delta",
                                        "index": content_index,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": arguments,
                                        }
                                    }));
                                }
                            }
                        }
                    }

                    if let Some(payload) = start_event {
                        events.push(sse_event("content_block_start", &payload));
                    }

                    if let Some(payload) = delta_event {
                        events.push(sse_event("content_block_delta", &payload));
                    }
                }
            }
        }

        events
    }

    fn finish(&mut self) -> Vec<Bytes> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;

        let mut events = Vec::new();

        if !self.emitted_message_start {
            events.push(sse_event(
                "message_start",
                &json!({
                    "type": "message_start",
                    "message": {
                        "id": self.response_id.clone().unwrap_or_else(fallback_message_id),
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model.clone().unwrap_or_else(|| self.requested_model.clone()),
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": {
                            "input_tokens": self.usage.prompt_tokens,
                            "output_tokens": 0,
                        }
                    }
                }),
            ));
            self.emitted_message_start = true;
        }

        if self.text_block_open {
            events.push(sse_event(
                "content_block_stop",
                &json!({
                    "type": "content_block_stop",
                    "index": self.text_block_index.unwrap_or(0),
                }),
            ));
            self.text_block_open = false;
        }

        let mut tool_indices = self.tool_blocks.keys().cloned().collect::<Vec<_>>();
        tool_indices.sort_unstable();
        for index in tool_indices {
            if self
                .tool_blocks
                .get(&index)
                .map(|state| state.started)
                .unwrap_or(false)
            {
                events.push(sse_event(
                    "content_block_stop",
                    &json!({
                        "type": "content_block_stop",
                        "index": self.tool_content_index(index),
                    }),
                ));
            }
        }

        events.push(sse_event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": map_finish_reason(self.finish_reason.as_deref()),
                    "stop_sequence": Value::Null,
                },
                "usage": {
                    "output_tokens": self.usage.completion_tokens,
                }
            }),
        ));
        events.push(sse_event(
            "message_stop",
            &json!({ "type": "message_stop" }),
        ));

        events
    }

    fn ensure_text_block(&mut self, events: &mut Vec<Bytes>) -> usize {
        let index = *self.text_block_index.get_or_insert_with(|| {
            if self.tool_blocks.is_empty() {
                0
            } else {
                self.tool_blocks.len()
            }
        });

        if !self.text_block_open {
            events.push(sse_event(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "text",
                        "text": "",
                    }
                }),
            ));
            self.text_block_open = true;
        }

        index
    }

    fn tool_content_index(&self, tool_index: usize) -> usize {
        match self.text_block_index {
            Some(0) => tool_index + 1,
            Some(text_index) if text_index <= tool_index => tool_index,
            _ => tool_index,
        }
    }
}

fn extract_stream_text(content: &Value) -> Option<String> {
    match content {
        Value::String(text) => Some(text.clone()),
        Value::Array(items) => {
            let mut text = String::new();
            for item in items {
                if let Some(part) = item.get("text").and_then(Value::as_str) {
                    text.push_str(part);
                }
            }
            if text.is_empty() {
                None
            } else {
                Some(text)
            }
        }
        _ => None,
    }
}

fn compute_tool_content_index(text_block_index: Option<usize>, tool_index: usize) -> usize {
    match text_block_index {
        Some(0) => tool_index + 1,
        Some(text_index) if text_index <= tool_index => tool_index,
        _ => tool_index,
    }
}

fn fallback_message_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("msg_prism_{millis}")
}

fn current_timestamp_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::{anthropic_request_to_openai, openai_response_to_anthropic};
    use crate::{
        provider::ProviderKind,
        types::{
            AnthropicBlock, AnthropicContent, AnthropicMessage, AnthropicMessagesRequest, Backend,
            OpenAiChatCompletionResponse,
        },
    };
    use serde_json::{json, Map, Value};
    use url::Url;

    #[test]
    fn request_translation_relaxes_image_and_thinking_blocks() {
        let mut thinking_fields = Map::new();
        thinking_fields.insert(
            "thinking".into(),
            Value::String("private scratchpad".into()),
        );

        let mut image_fields = Map::new();
        image_fields.insert(
            "source".into(),
            json!({
                "type": "url",
                "url": "https://example.com/cat.png"
            }),
        );

        let backend = Backend {
            prefix: "glm".into(),
            provider: ProviderKind::Zai,
            base: Url::parse("https://api.z.ai/api/paas/v4/").unwrap(),
            api_key: "test".into(),
            credential_label: "inline".into(),
        };

        let request = AnthropicMessagesRequest {
            model: "glm-4.5".into(),
            messages: vec![
                AnthropicMessage {
                    role: "assistant".into(),
                    content: AnthropicContent::Blocks(vec![
                        AnthropicBlock {
                            kind: "thinking".into(),
                            fields: thinking_fields,
                        },
                        AnthropicBlock::text("Visible answer".into()),
                    ]),
                },
                AnthropicMessage {
                    role: "user".into(),
                    content: AnthropicContent::Blocks(vec![
                        AnthropicBlock {
                            kind: "image".into(),
                            fields: image_fields,
                        },
                        AnthropicBlock::text("Describe the image.".into()),
                    ]),
                },
            ],
            system: None,
            max_tokens: Some(64),
            stream: Some(true),
            temperature: None,
            top_p: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
        };

        let prepared = anthropic_request_to_openai(request, &backend).unwrap();
        let messages = prepared
            .body
            .get("messages")
            .and_then(Value::as_array)
            .unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(
            messages[0].get("reasoning_content"),
            Some(&json!("private scratchpad"))
        );
        assert!(messages[1]
            .get("content")
            .and_then(Value::as_array)
            .is_some());
    }

    #[test]
    fn response_translation_accepts_object_tool_arguments() {
        let response: OpenAiChatCompletionResponse = serde_json::from_value(json!({
            "id": "chatcmpl_demo",
            "model": "glm-4.5",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "lookup_weather",
                                    "arguments": {
                                        "city": "Seoul"
                                    }
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4
            }
        }))
        .unwrap();

        let anthropic = openai_response_to_anthropic(response, "glm-4.5").unwrap();
        let content = anthropic.get("content").and_then(Value::as_array).unwrap();

        assert_eq!(content[0].get("type"), Some(&json!("tool_use")));
        assert_eq!(
            content[0]
                .get("input")
                .and_then(Value::as_object)
                .and_then(|input| input.get("city"))
                .and_then(Value::as_str),
            Some("Seoul")
        );
    }
}
