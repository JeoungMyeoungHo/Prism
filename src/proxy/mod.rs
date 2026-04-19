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

mod handlers;
mod passthrough;
mod streaming;
mod tools;
mod translate_request;
mod translate_response;
mod upstream_error;

pub use handlers::forward_request_to_backend;

use handlers::forward_responses_request_to_backend;
use passthrough::forward_anthropic_passthrough;
use translate_request::{anthropic_request_to_openai, responses_request_to_openai_chat};
use translate_response::{
    extract_textish_value, openai_chat_response_to_responses, openai_response_to_anthropic,
};

use crate::{
    router::ModelRouter,
    types::{AnthropicMessagesRequest, Backend},
};
use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};
use streaming::{sse_event, AnthropicStreamTranslator, ResponsesStreamTranslator};
use tools::{convert_tool_choice, convert_tools, normalize_schema};
use tracing::info;
use upstream_error::{extract_error_message, read_upstream_error};

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

fn current_timestamp_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests;
