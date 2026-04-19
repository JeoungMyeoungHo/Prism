//! Request/response translation and SSE streaming.
//!
//! This module is the heart of the proxy. It converts between three protocol
//! shapes and drives the upstream HTTP call:
//!
//! - **Anthropic Messages** (client-facing, `/v1/messages`)
//! - **OpenAI Chat Completions** (upstream wire format)
//! - **OpenAI Responses** (client-facing, `/v1/responses`)
//!
//! High-level flow (non-streaming):
//!
//! ```text
//! client request → translate_request → upstream POST → translate_response → client
//! ```
//!
//! Streaming flow feeds upstream SSE chunks through a stateful translator
//! (see [`AnthropicStreamTranslator`], [`ResponsesStreamTranslator`]) that
//! re-emits SSE events in the client's expected shape.
//!
//! Cross-cutting pieces:
//! - [`AppState`] — shared router + HTTP client, cloned into each handler.
//! - [`ApiError`] — unified error surface for translation/upstream failures.
//! - [`read_upstream_error`] — normalizes non-2xx upstream bodies into [`ApiError`].
//! - [`anthropic_passthrough_url`] / [`forward_anthropic_passthrough`] —
//!   bypass translation when upstream is itself Anthropic-compatible.
//!
//! The file is large; until it is split into submodules, use a symbol search
//! rather than scrolling. Items are grouped roughly in this order: app state
//! and error, passthrough, request translators, content/tool helpers, response
//! translators, error helpers, SSE formatters, stream translators, HTTP
//! handlers, tests.

use crate::{
    router::ModelRouter,
    types::{
        AnthropicBlock, AnthropicMessage, AnthropicMessagesRequest, AnthropicSystemPrompt,
        AnthropicTool, AnthropicToolChoice, Backend, OpenAiChatCompletionChunk,
        OpenAiChatCompletionResponse, OpenAiToolCallDelta, OpenAiUsage,
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
use tracing::{info, warn};

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
/// definitions, SSE events — is relayed verbatim. Auth sends both the native
/// Anthropic `x-api-key` header and an `Authorization: Bearer` header so the
/// same passthrough route can target Anthropic itself or third-party gateways
/// that only accept bearer auth on their Anthropic-compatible endpoint.
///
/// The upstream Messages URL is derived from the configured base:
/// - bases that already end in `/v1/` use `{base}messages`
/// - all other bases use `{base}v1/messages`
///
/// This matches the two common Anthropic-compatible conventions:
/// - direct endpoint bases such as `https://api.anthropic.com/v1/`
/// - SDK-style bases such as `https://api.fireworks.ai/inference/`
fn anthropic_passthrough_url(base: &url::Url) -> url::Url {
    let suffix = if base.path().trim_end_matches('/').ends_with("/v1") {
        "messages"
    } else {
        "v1/messages"
    };
    // `base` is validated at config load; `suffix` is one of two string
    // literals selected above. `join` cannot fail.
    base.join(suffix)
        .expect("literal relative ref joined against validated base URL")
}

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

    let upstream_url = anthropic_passthrough_url(&backend.base);

    let bearer = format!("Bearer {}", backend.api_key);
    let mut request_builder = client
        .post(upstream_url)
        .header("x-api-key", &backend.api_key)
        .header(reqwest::header::AUTHORIZATION, bearer)
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
        proxy_responses_streaming(
            client,
            backend,
            forwarded_headers,
            prepared.body,
            requested_model,
        )
        .await
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

async fn proxy_responses_streaming(
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
        let mut translator = ResponsesStreamTranslator::new(requested_model);
        let mut finished = false;

        loop {
            let line = match lines.next_line().await {
                Ok(Some(line)) => line,
                Ok(None) => break,
                Err(error) => {
                    for event in translator.fail("upstream_stream_error", &error.to_string()) {
                        yield Ok::<Bytes, Infallible>(event);
                    }
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
                    for event in translator.fail("invalid_upstream_payload", &error.to_string()) {
                        yield Ok::<Bytes, Infallible>(event);
                    }
                    finished = true;
                    break;
                }
            };

            if let Some(error) = value.get("error") {
                for event in translator.fail("upstream_error", &extract_error_message(error)) {
                    yield Ok::<Bytes, Infallible>(event);
                }
                finished = true;
                break;
            }

            let chunk: OpenAiChatCompletionChunk = match serde_json::from_value(value) {
                Ok(chunk) => chunk,
                Err(error) => {
                    for event in translator.fail("invalid_upstream_payload", &error.to_string()) {
                        yield Ok::<Bytes, Infallible>(event);
                    }
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

    // `openai_request` was just built with `json!({ ... })`, which always
    // yields `Value::Object`. The expect here is a non-panicking invariant.
    let object = openai_request
        .as_object_mut()
        .expect("json! literal always yields Value::Object");

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
    let status = match choice.finish_reason.as_deref() {
        Some("length") => "incomplete",
        Some("content_filter") => "failed",
        _ => "completed",
    };

    let mut translated = json!({
        "id": response.id,
        "object": "response",
        "created_at": created_at,
        "completed_at": created_at,
        "status": status,
        "model": response.model.unwrap_or_else(|| requested_model.to_string()),
        "output": output,
        "parallel_tool_calls": output.iter().filter(|item| item.get("type") == Some(&json!("function_call"))).count() > 1,
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.prompt_tokens + usage.completion_tokens,
        }
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

fn extract_textish_value(value: &Value) -> Option<String> {
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

    // Preserve the full upstream body in logs for debugging — response to the
    // client is tagged but keeps only the extracted message.
    warn!(
        target: "prism::upstream",
        %status,
        body = %body,
        "upstream error"
    );

    let message = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|value| value.get("error").cloned())
        .map(|error| extract_error_message(&error))
        .unwrap_or_else(|| {
            if body.is_empty() {
                format!("request failed with HTTP {status}")
            } else {
                body
            }
        });

    ApiError::new(status, "upstream_error", format!("[upstream] {message}"))
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

/// Incrementally converts an OpenAI Chat Completion SSE stream into the
/// Anthropic Messages SSE shape.
///
/// Event sequence emitted to the client:
///
/// ```text
/// message_start
///   (content_block_start → content_block_delta* → content_block_stop)*
/// message_delta
/// message_stop
/// ```
///
/// One text block and zero-or-more tool blocks may interleave. Block indices
/// are assigned by this translator and are **not** taken from upstream.
///
/// Invariants:
/// - `message_start` is emitted exactly once, on the first chunk carrying a
///   response id or model (guarded by `emitted_message_start`).
/// - `text_block_*` track the single text block; it is opened lazily on the
///   first text delta and closed before `message_delta`.
/// - `tool_blocks` is indexed by **our** assigned index (see
///   [`AnthropicStreamTranslator::resolve_tool_index`]); upstream tool indices
///   are best-effort and may be absent, so we soft-match by id/name or fall
///   back to a synthetic index.
/// - `finished` guards against duplicate `message_stop` if `finish()` runs
///   after a terminal chunk.
///
/// Silent `unwrap_or_default()` / `unwrap_or(...)` inside [`push`] handle
/// legitimately-absent OpenAI delta fields (e.g. missing `content` on a
/// usage-only chunk) and are part of the normal path — do not promote to
/// WARN without evidence of real data loss.
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
    next_tool_index: usize,
    last_tool_index: Option<usize>,
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
            next_tool_index: 0,
            last_tool_index: None,
        }
    }

    /// Ingest one OpenAI chunk and emit zero or more Anthropic SSE events.
    /// First chunk with identity fields triggers `message_start`; subsequent
    /// chunks may open/append text or tool blocks. Terminal chunks (finish
    /// reason set) are coalesced with the final usage into `message_delta` +
    /// `message_stop` by [`finish`].
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
                    let tool_index = self.resolve_tool_index(&tool_call);
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

    /// Lazily open the text block on first text delta and return its index.
    /// Index is chosen to not collide with existing tool blocks.
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

    /// Map an upstream tool delta to our local tool block index.
    ///
    /// Resolution order:
    /// 1. Upstream-provided `index` (authoritative when present).
    /// 2. Match by tool call id or function name against already-open blocks.
    /// 3. Reuse the single open tool block when there is exactly one.
    /// 4. Allocate a synthetic index via [`allocate_synthetic_tool_index`].
    ///
    /// This is the only soft-match point in the translator — upstreams that
    /// omit `index` on tool deltas still produce coherent output.
    fn resolve_tool_index(&mut self, tool_call: &OpenAiToolCallDelta) -> usize {
        if let Some(index) = tool_call.index {
            self.last_tool_index = Some(index);
            self.note_explicit_tool_index(index);
            return index;
        }

        if let Some(id) = tool_call.id.as_deref().filter(|id| !id.is_empty()) {
            if let Some(index) = self
                .tool_blocks
                .iter()
                .find_map(|(index, state)| (state.id == id).then_some(*index))
            {
                self.last_tool_index = Some(index);
                return index;
            }
        }

        if let Some(function) = tool_call.function.as_ref() {
            if let Some(name) = function.name.as_deref().filter(|name| !name.is_empty()) {
                let matches = self
                    .tool_blocks
                    .iter()
                    .filter_map(|(index, state)| (state.name == name).then_some(*index))
                    .collect::<Vec<_>>();
                if matches.len() == 1 {
                    self.last_tool_index = Some(matches[0]);
                    return matches[0];
                }
            }

            if function
                .arguments
                .as_deref()
                .is_some_and(|arguments| !arguments.is_empty())
            {
                if let Some(index) = self
                    .last_tool_index
                    .filter(|index| self.tool_blocks.contains_key(index))
                {
                    return index;
                }
            }
        }

        // Only one tool block open → route deltas there. `if let Some`
        // expresses the invariant locally without an unreachable expect.
        if self.tool_blocks.len() == 1 {
            if let Some((&index, _)) = self.tool_blocks.iter().next() {
                self.last_tool_index = Some(index);
                return index;
            }
        }

        let index = self.allocate_synthetic_tool_index();
        self.last_tool_index = Some(index);
        index
    }

    fn note_explicit_tool_index(&mut self, index: usize) {
        self.next_tool_index = self.next_tool_index.max(index.saturating_add(1));
    }

    /// Return the lowest unused index, skipping any already occupied by
    /// existing tool blocks. Used when upstream gives us no identifying info.
    fn allocate_synthetic_tool_index(&mut self) -> usize {
        while self.tool_blocks.contains_key(&self.next_tool_index) {
            self.next_tool_index += 1;
        }
        let index = self.next_tool_index;
        self.next_tool_index += 1;
        index
    }
}

fn extract_stream_text(content: &Value) -> Option<String> {
    extract_textish_value(content)
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

#[derive(Default)]
struct ResponsesToolStreamState {
    item_id: String,
    call_id: String,
    name: String,
    arguments: String,
    output_index: usize,
    added: bool,
}

struct ResponsesClosedOutputItem {
    output_index: usize,
    item: Value,
}

/// Translates an upstream `chat/completions` SSE stream into OpenAI
/// `/v1/responses` SSE events. Mirrors what
/// [`openai_chat_response_to_responses`] produces for non-streaming, but
/// maintains incremental state for `response.output_text.delta`,
/// `response.function_call_arguments.delta`, and
/// `response.reasoning_summary_text.delta`.
///
/// Event sequence emitted to the client:
///
/// ```text
/// response.created → response.in_progress
///   (response.output_item.added
///      → response.(reasoning_summary_text|output_text|function_call_arguments).delta*
///      → response.output_item.done)*
/// response.completed  (or response.failed via `fail`)
/// ```
///
/// Each upstream chunk may touch three distinct output item kinds —
/// reasoning, message (text), and tool call. They are tracked independently
/// and closed in the order they were opened, with `sequence` incremented on
/// every emitted event.
///
/// Invariants:
/// - `response.created` + `response.in_progress` fire exactly once, guarded
///   by `emitted_created`.
/// - `reasoning_*` / `message_*` ids and output indices are assigned lazily
///   on the first relevant delta. Opening is idempotent.
/// - `tools` is indexed by our local index (see [`resolve_tool_index`]);
///   upstream may omit indices, in which case id/name soft-match is tried.
/// - `finished` prevents `finish()`/`fail()` from double-emitting terminal
///   events.
/// - `next_output_index` is bumped whenever a new output item is opened, so
///   all emitted `output_index` values are monotonically increasing.
struct ResponsesStreamTranslator {
    requested_model: String,
    created_at: u64,
    response_id: Option<String>,
    upstream_model: Option<String>,
    usage: OpenAiUsage,
    finish_reason: Option<String>,

    sequence: u64,
    finished: bool,
    emitted_created: bool,

    next_output_index: usize,

    reasoning_item_id: Option<String>,
    reasoning_output_index: Option<usize>,
    reasoning_part_added: bool,
    reasoning_text: String,

    message_item_id: Option<String>,
    message_output_index: Option<usize>,
    message_part_added: bool,
    message_text: String,

    tools: BTreeMap<usize, ResponsesToolStreamState>,
    next_tool_index: usize,
    last_tool_index: Option<usize>,
    output_items: Vec<ResponsesClosedOutputItem>,
}

impl ResponsesStreamTranslator {
    fn new(requested_model: String) -> Self {
        Self {
            requested_model,
            created_at: current_timestamp_seconds(),
            response_id: None,
            upstream_model: None,
            usage: OpenAiUsage::default(),
            finish_reason: None,
            sequence: 0,
            finished: false,
            emitted_created: false,
            next_output_index: 0,
            reasoning_item_id: None,
            reasoning_output_index: None,
            reasoning_part_added: false,
            reasoning_text: String::new(),
            message_item_id: None,
            message_output_index: None,
            message_part_added: false,
            message_text: String::new(),
            tools: BTreeMap::new(),
            next_tool_index: 0,
            last_tool_index: None,
            output_items: Vec::new(),
        }
    }

    fn next_seq(&mut self) -> u64 {
        let n = self.sequence;
        self.sequence += 1;
        n
    }

    fn model_label(&self) -> String {
        self.upstream_model
            .clone()
            .unwrap_or_else(|| self.requested_model.clone())
    }

    fn response_id_or_fallback(&self) -> String {
        self.response_id
            .clone()
            .unwrap_or_else(fallback_response_id)
    }

    fn terminal_event_name(&self) -> &'static str {
        match self.finish_reason.as_deref() {
            Some("length") => "response.incomplete",
            Some("content_filter") => "response.failed",
            _ => "response.completed",
        }
    }

    fn terminal_status(&self) -> &'static str {
        match self.finish_reason.as_deref() {
            Some("length") => "incomplete",
            Some("content_filter") => "failed",
            _ => "completed",
        }
    }

    /// Ingest one OpenAI chunk and emit zero or more Responses SSE events.
    /// Opens per-kind output items lazily and bumps the shared `sequence`
    /// counter on every event. Finish reason, usage, and stream termination
    /// are collected and flushed by [`finish`] / [`fail`].
    fn push(&mut self, chunk: OpenAiChatCompletionChunk) -> Vec<Bytes> {
        let mut events = Vec::new();

        if let Some(id) = chunk.id {
            self.response_id = Some(id);
        }
        if let Some(model) = chunk.model {
            self.upstream_model = Some(model);
        }
        if let Some(usage) = chunk.usage {
            self.usage = usage;
        }

        if !self.emitted_created {
            self.emit_created(&mut events);
        }

        for choice in chunk.choices {
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }

            if let Some(reasoning) = choice.delta.reasoning_content {
                if let Some(text) = extract_stream_text(&reasoning) {
                    if !text.is_empty() {
                        self.emit_reasoning_delta(&text, &mut events);
                    }
                }
            }

            if let Some(content) = choice.delta.content {
                if let Some(text) = extract_stream_text(&content) {
                    if !text.is_empty() {
                        self.emit_text_delta(&text, &mut events);
                    }
                }
            }

            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    self.absorb_tool_call_delta(tool_call, &mut events);
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

        if !self.emitted_created {
            self.emit_created(&mut events);
        }

        // Close reasoning item if open (reasoning comes before the message).
        if self.reasoning_part_added {
            self.close_reasoning(&mut events);
        }

        // Close message item if open.
        if self.message_item_id.is_some() {
            self.close_message(&mut events);
        }

        // Close any tool items.
        self.close_all_tools(&mut events);

        let response = self.final_response_object();
        let seq = self.next_seq();
        events.push(sse_named_event(
            self.terminal_event_name(),
            &json!({
                "type": self.terminal_event_name(),
                "sequence_number": seq,
                "response": response,
            }),
        ));

        events
    }

    /// Abort the stream with `response.failed`. No-op if already finished.
    /// Closes any open items in their current state before emitting the
    /// terminal event.
    fn fail(&mut self, code: &str, message: &str) -> Vec<Bytes> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;

        let mut events = Vec::new();

        if !self.emitted_created {
            self.emit_created(&mut events);
        }

        if self.reasoning_item_id.is_some() {
            self.close_reasoning(&mut events);
        }
        if self.message_item_id.is_some() {
            self.close_message(&mut events);
        }
        self.close_all_tools(&mut events);

        let response = self.failed_response_object(code, message);
        let seq = self.next_seq();
        events.push(sse_named_event(
            "response.failed",
            &json!({
                "type": "response.failed",
                "sequence_number": seq,
                "response": response,
            }),
        ));

        events
    }

    fn emit_created(&mut self, events: &mut Vec<Bytes>) {
        self.emitted_created = true;
        let envelope = self.in_progress_response_object();
        let seq_created = self.next_seq();
        events.push(sse_named_event(
            "response.created",
            &json!({
                "type": "response.created",
                "sequence_number": seq_created,
                "response": envelope,
            }),
        ));
        let seq_in_progress = self.next_seq();
        events.push(sse_named_event(
            "response.in_progress",
            &json!({
                "type": "response.in_progress",
                "sequence_number": seq_in_progress,
                "response": self.in_progress_response_object(),
            }),
        ));
    }

    /// Append `text` to the reasoning output item, opening it on first use.
    fn emit_reasoning_delta(&mut self, text: &str, events: &mut Vec<Bytes>) {
        if self.message_item_id.is_some() {
            self.close_message(events);
        }
        self.close_all_tools(events);

        if self.reasoning_item_id.is_none() {
            let response_id = self.response_id_or_fallback();
            let output_index = self.next_output_index;
            self.next_output_index += 1;
            let item_id = format!("rs_{response_id}_{output_index}");
            self.reasoning_item_id = Some(item_id.clone());
            self.reasoning_output_index = Some(output_index);
            let seq = self.next_seq();
            events.push(sse_named_event(
                "response.output_item.added",
                &json!({
                    "type": "response.output_item.added",
                    "sequence_number": seq,
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "reasoning",
                        "status": "in_progress",
                        "summary": []
                    }
                }),
            ));
        }

        if !self.reasoning_part_added {
            let item_id = self.reasoning_item_id.clone().unwrap();
            let output_index = self.reasoning_output_index.unwrap();
            let seq = self.next_seq();
            events.push(sse_named_event(
                "response.reasoning_summary_part.added",
                &json!({
                    "type": "response.reasoning_summary_part.added",
                    "sequence_number": seq,
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": { "type": "summary_text", "text": "" }
                }),
            ));
            self.reasoning_part_added = true;
        }

        self.reasoning_text.push_str(text);
        let item_id = self.reasoning_item_id.clone().unwrap();
        let output_index = self.reasoning_output_index.unwrap();
        let seq = self.next_seq();
        events.push(sse_named_event(
            "response.reasoning_summary_text.delta",
            &json!({
                "type": "response.reasoning_summary_text.delta",
                "sequence_number": seq,
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": 0,
                "delta": text,
            }),
        ));
    }

    /// Append `text` to the message output item, opening it on first use.
    fn emit_text_delta(&mut self, text: &str, events: &mut Vec<Bytes>) {
        // Seal non-message items before opening the message item so
        // `output_index` reflects the order each item started streaming.
        if self.reasoning_item_id.is_some() {
            self.close_reasoning(events);
        }
        self.close_all_tools(events);

        if self.message_item_id.is_none() {
            let response_id = self.response_id_or_fallback();
            let output_index = self.next_output_index;
            self.next_output_index += 1;
            let item_id = format!("msg_{response_id}_{output_index}");
            self.message_item_id = Some(item_id.clone());
            self.message_output_index = Some(output_index);
            let seq = self.next_seq();
            events.push(sse_named_event(
                "response.output_item.added",
                &json!({
                    "type": "response.output_item.added",
                    "sequence_number": seq,
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": []
                    }
                }),
            ));
        }

        if !self.message_part_added {
            let item_id = self.message_item_id.clone().unwrap();
            let output_index = self.message_output_index.unwrap();
            let seq = self.next_seq();
            events.push(sse_named_event(
                "response.content_part.added",
                &json!({
                    "type": "response.content_part.added",
                    "sequence_number": seq,
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": { "type": "output_text", "text": "", "annotations": [] }
                }),
            ));
            self.message_part_added = true;
        }

        self.message_text.push_str(text);
        let item_id = self.message_item_id.clone().unwrap();
        let output_index = self.message_output_index.unwrap();
        let seq = self.next_seq();
        events.push(sse_named_event(
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "sequence_number": seq,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
            }),
        ));
    }

    /// Fold a tool-call delta into the matching tool state, opening the
    /// output item on first sight and routing fragments (id/name/arguments)
    /// into their accumulators. Must run after any pending text/reasoning
    /// items are closed so output indices stay monotonic.
    fn absorb_tool_call_delta(
        &mut self,
        tool_call: crate::types::OpenAiToolCallDelta,
        events: &mut Vec<Bytes>,
    ) {
        // Text and reasoning items should be sealed before function_call items
        // so that `output_index` reflects the order the first piece of each
        // item arrived. We seal opportunistically once a tool delta is seen.
        if self.reasoning_item_id.is_some() {
            self.close_reasoning(events);
        }
        if self.message_item_id.is_some() {
            self.close_message(events);
        }

        let tool_index = self.resolve_tool_index(&tool_call);
        let needs_open = !self.tools.contains_key(&tool_index);
        if needs_open {
            let output_index = self.next_output_index;
            self.next_output_index += 1;
            self.tools.insert(
                tool_index,
                ResponsesToolStreamState {
                    output_index,
                    ..Default::default()
                },
            );
        }

        let state = self.tools.get_mut(&tool_index).unwrap();
        if let Some(id) = tool_call.id {
            if !id.is_empty() {
                state.call_id = id;
            }
        }
        if let Some(function) = tool_call.function {
            if let Some(name) = function.name {
                if !name.is_empty() {
                    state.name = name;
                }
            }
            if let Some(args) = function.arguments {
                if !state.added
                    && (!state.name.is_empty() || !state.call_id.is_empty() || !args.is_empty())
                {
                    Self::open_tool_item(state, events, &mut self.sequence);
                }
                if !args.is_empty() {
                    state.arguments.push_str(&args);
                    let seq = {
                        let n = self.sequence;
                        self.sequence += 1;
                        n
                    };
                    events.push(sse_named_event(
                        "response.function_call_arguments.delta",
                        &json!({
                            "type": "response.function_call_arguments.delta",
                            "sequence_number": seq,
                            "item_id": state.item_id,
                            "output_index": state.output_index,
                            "delta": args,
                        }),
                    ));
                }
            }
        }

        // Ensure the item is opened even when only id/name arrived without
        // argument text yet.
        if let Some(state) = self.tools.get_mut(&tool_index) {
            if !state.added && (!state.name.is_empty() || !state.call_id.is_empty()) {
                Self::open_tool_item(state, events, &mut self.sequence);
            }
        }
    }

    fn close_all_tools(&mut self, events: &mut Vec<Bytes>) {
        let tool_indices: Vec<usize> = self.tools.keys().copied().collect();
        for idx in tool_indices {
            self.close_tool(idx, events);
        }
    }

    /// Emit `response.output_item.added` for a tool call. Idempotent via
    /// `state.added`.
    fn open_tool_item(
        state: &mut ResponsesToolStreamState,
        events: &mut Vec<Bytes>,
        sequence: &mut u64,
    ) {
        if state.added {
            return;
        }
        if state.call_id.is_empty() {
            state.call_id = format!("call_prism_{}", state.output_index);
        }
        if state.item_id.is_empty() {
            state.item_id = format!("fc_{}", state.call_id);
        }
        if state.name.is_empty() {
            state.name = format!("tool_{}", state.output_index);
        }

        let seq = {
            let n = *sequence;
            *sequence += 1;
            n
        };
        events.push(sse_named_event(
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "sequence_number": seq,
                "output_index": state.output_index,
                "item": {
                    "id": state.item_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": state.call_id,
                    "name": state.name,
                    "arguments": ""
                }
            }),
        ));
        state.added = true;
    }

    /// Emit `part.done` + `output_item.done` for the reasoning item and
    /// reset the reasoning accumulators so a future delta would reopen.
    fn close_reasoning(&mut self, events: &mut Vec<Bytes>) {
        let Some(item_id) = self.reasoning_item_id.clone() else {
            return;
        };
        let output_index = self.reasoning_output_index.unwrap_or(0);
        let text = std::mem::take(&mut self.reasoning_text);

        if self.reasoning_part_added {
            let seq1 = self.next_seq();
            events.push(sse_named_event(
                "response.reasoning_summary_text.done",
                &json!({
                    "type": "response.reasoning_summary_text.done",
                    "sequence_number": seq1,
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "text": text,
                }),
            ));
            let seq2 = self.next_seq();
            events.push(sse_named_event(
                "response.reasoning_summary_part.done",
                &json!({
                    "type": "response.reasoning_summary_part.done",
                    "sequence_number": seq2,
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": {
                        "type": "summary_text",
                        "text": text,
                    }
                }),
            ));
            self.reasoning_part_added = false;
        }

        let item = json!({
            "id": item_id.clone(),
            "type": "reasoning",
            "status": "completed",
            "summary": [
                { "type": "summary_text", "text": text.clone() }
            ]
        });
        let seq3 = self.next_seq();
        events.push(sse_named_event(
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "sequence_number": seq3,
                "output_index": output_index,
                "item": item.clone(),
            }),
        ));
        self.output_items
            .push(ResponsesClosedOutputItem { output_index, item });

        self.reasoning_item_id = None;
        self.reasoning_output_index = None;
    }

    /// Emit `part.done` + `output_item.done` for the message (text) item
    /// and reset its accumulators.
    fn close_message(&mut self, events: &mut Vec<Bytes>) {
        let Some(item_id) = self.message_item_id.clone() else {
            return;
        };
        let output_index = self.message_output_index.unwrap_or(0);
        let text = std::mem::take(&mut self.message_text);

        let seq1 = self.next_seq();
        events.push(sse_named_event(
            "response.output_text.done",
            &json!({
                "type": "response.output_text.done",
                "sequence_number": seq1,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            }),
        ));
        let seq2 = self.next_seq();
        events.push(sse_named_event(
            "response.content_part.done",
            &json!({
                "type": "response.content_part.done",
                "sequence_number": seq2,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": text,
                    "annotations": []
                }
            }),
        ));
        let item = json!({
            "id": item_id.clone(),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": text.clone(),
                    "annotations": []
                }
            ]
        });
        let seq3 = self.next_seq();
        events.push(sse_named_event(
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "sequence_number": seq3,
                "output_index": output_index,
                "item": item.clone(),
            }),
        ));
        self.output_items
            .push(ResponsesClosedOutputItem { output_index, item });

        self.message_part_added = false;
        self.message_item_id = None;
        self.message_output_index = None;
    }

    /// Emit `output_item.done` for the tool call and remove its state.
    fn close_tool(&mut self, tool_index: usize, events: &mut Vec<Bytes>) {
        let Some(mut state) = self.tools.remove(&tool_index) else {
            return;
        };
        if self.last_tool_index == Some(tool_index) {
            self.last_tool_index = None;
        }
        if !state.added {
            Self::open_tool_item(&mut state, events, &mut self.sequence);
        }

        let seq1 = self.next_seq();
        events.push(sse_named_event(
            "response.function_call_arguments.done",
            &json!({
                "type": "response.function_call_arguments.done",
                "sequence_number": seq1,
                "item_id": state.item_id,
                "output_index": state.output_index,
                "arguments": state.arguments,
            }),
        ));
        let item = json!({
            "id": state.item_id.clone(),
            "type": "function_call",
            "status": "completed",
            "call_id": state.call_id.clone(),
            "name": state.name.clone(),
            "arguments": state.arguments.clone(),
        });
        let seq2 = self.next_seq();
        events.push(sse_named_event(
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "sequence_number": seq2,
                "output_index": state.output_index,
                "item": item.clone(),
            }),
        ));
        self.output_items.push(ResponsesClosedOutputItem {
            output_index: state.output_index,
            item,
        });
    }

    /// Map an upstream tool delta to our local tool index. Mirrors
    /// [`AnthropicStreamTranslator::resolve_tool_index`]: explicit upstream
    /// index wins, then id/name soft-match, then single-open reuse, then
    /// synthetic allocation.
    fn resolve_tool_index(&mut self, tool_call: &crate::types::OpenAiToolCallDelta) -> usize {
        if let Some(index) = tool_call.index {
            self.last_tool_index = Some(index);
            self.note_explicit_tool_index(index);
            return index;
        }

        if let Some(id) = tool_call.id.as_deref().filter(|id| !id.is_empty()) {
            if let Some(index) = self
                .tools
                .iter()
                .find_map(|(index, state)| (state.call_id == id).then_some(*index))
            {
                self.last_tool_index = Some(index);
                return index;
            }
        }

        if let Some(function) = tool_call.function.as_ref() {
            if let Some(name) = function.name.as_deref().filter(|name| !name.is_empty()) {
                let matches = self
                    .tools
                    .iter()
                    .filter_map(|(index, state)| (state.name == name).then_some(*index))
                    .collect::<Vec<_>>();
                if matches.len() == 1 {
                    self.last_tool_index = Some(matches[0]);
                    return matches[0];
                }
            }

            if function
                .arguments
                .as_deref()
                .is_some_and(|arguments| !arguments.is_empty())
            {
                if let Some(index) = self
                    .last_tool_index
                    .filter(|idx| self.tools.contains_key(idx))
                {
                    return index;
                }
            }
        }

        // Only one tool open → route deltas there. `if let Some`
        // expresses the invariant locally without an unreachable expect.
        if self.tools.len() == 1 {
            if let Some((&index, _)) = self.tools.iter().next() {
                self.last_tool_index = Some(index);
                return index;
            }
        }

        let index = self.allocate_synthetic_tool_index();
        self.last_tool_index = Some(index);
        index
    }

    fn note_explicit_tool_index(&mut self, index: usize) {
        self.next_tool_index = self.next_tool_index.max(index.saturating_add(1));
    }

    /// Lowest unused index, skipping any already occupied tool slot.
    fn allocate_synthetic_tool_index(&mut self) -> usize {
        while self.tools.contains_key(&self.next_tool_index) {
            self.next_tool_index += 1;
        }
        let index = self.next_tool_index;
        self.next_tool_index += 1;
        index
    }

    fn finalized_output(&self) -> Vec<Value> {
        let mut output_items: Vec<&ResponsesClosedOutputItem> = self.output_items.iter().collect();
        output_items.sort_by_key(|item| item.output_index);
        output_items.iter().map(|item| item.item.clone()).collect()
    }

    fn in_progress_response_object(&self) -> Value {
        json!({
            "id": self.response_id_or_fallback(),
            "object": "response",
            "created_at": self.created_at,
            "status": "in_progress",
            "model": self.model_label(),
            "output": [],
        })
    }

    fn final_response_object(&self) -> Value {
        let output = self.finalized_output();
        let tool_count = output
            .iter()
            .filter(|item| item.get("type") == Some(&json!("function_call")))
            .count();

        let mut response = json!({
            "id": self.response_id_or_fallback(),
            "object": "response",
            "created_at": self.created_at,
            "completed_at": current_timestamp_seconds(),
            "status": self.terminal_status(),
            "model": self.model_label(),
            "output": output,
            "parallel_tool_calls": tool_count > 1,
            "usage": {
                "input_tokens": self.usage.prompt_tokens,
                "output_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.prompt_tokens + self.usage.completion_tokens,
            }
        });

        // `response` just built via `json!({ ... })` — always Value::Object.
        let response_object = response
            .as_object_mut()
            .expect("json! literal always yields Value::Object");
        match self.finish_reason.as_deref() {
            Some("length") => {
                response_object.insert(
                    "incomplete_details".into(),
                    json!({ "reason": "max_output_tokens" }),
                );
            }
            Some("content_filter") => {
                response_object.insert(
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

        response
    }

    fn failed_response_object(&self, code: &str, message: &str) -> Value {
        let output = self.finalized_output();
        let tool_count = output
            .iter()
            .filter(|item| item.get("type") == Some(&json!("function_call")))
            .count();

        json!({
            "id": self.response_id_or_fallback(),
            "object": "response",
            "created_at": self.created_at,
            "completed_at": current_timestamp_seconds(),
            "status": "failed",
            "model": self.model_label(),
            "output": output,
            "parallel_tool_calls": tool_count > 1,
            "usage": {
                "input_tokens": self.usage.prompt_tokens,
                "output_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.prompt_tokens + self.usage.completion_tokens,
            },
            "error": {
                "type": code,
                "code": code,
                "message": message,
            }
        })
    }
}

fn sse_named_event(event: &str, payload: &Value) -> Bytes {
    let json = serde_json::to_string(payload).unwrap_or_else(|_| "{}".into());
    Bytes::from(format!("event: {event}\ndata: {json}\n\n"))
}

fn fallback_response_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("resp_prism_{millis}")
}

#[cfg(test)]
mod tests;
