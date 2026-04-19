//! Upstream HTTP orchestration and SSE bridging.
//!
//! `forward_request_to_backend` / `forward_responses_request_to_backend` are
//! the entry points called from the top-level route handlers in
//! [`super`]; they invoke the translator and dispatch to a streaming or
//! non-streaming delegate. The `proxy_*` functions run the upstream request
//! and adapt the response body.

use super::{
    anthropic_request_to_openai, backend_label, extract_error_message,
    openai_chat_response_to_responses, openai_response_to_anthropic, read_upstream_error,
    responses_request_to_openai_chat, sse_event, AnthropicStreamTranslator, ApiError,
    ResponsesStreamTranslator,
};
use crate::types::{
    AnthropicMessagesRequest, Backend, OpenAiChatCompletionChunk, OpenAiChatCompletionResponse,
};
use async_stream::stream;
use axum::{
    body::{Body, Bytes},
    http::{
        header::{CACHE_CONTROL, CONNECTION, CONTENT_TYPE},
        HeaderName, HeaderValue, StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use futures_util::TryStreamExt;
use serde_json::{json, Value};
use std::{convert::Infallible, io};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_util::io::StreamReader;
use tracing::info;

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

pub(super) async fn forward_responses_request_to_backend(
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

    // Captured before the body is consumed by the upstream request.
    let client_parallel_tool_calls = upstream_body
        .get("parallel_tool_calls")
        .and_then(Value::as_bool);

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

    let response_object = openai_chat_response_to_responses(
        openai_response,
        &requested_model,
        client_parallel_tool_calls,
    )?;
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
