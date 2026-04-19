//! Anthropic-format passthrough — forwards the inbound Messages request to
//! an upstream that already speaks Anthropic's API, rewriting only the
//! `model` field and relaying auth + SSE verbatim.

use super::{backend_label, read_upstream_error, ApiError};
use crate::types::Backend;
use axum::{
    body::{Body, Bytes},
    http::{
        header::{CACHE_CONTROL, CONNECTION, CONTENT_TYPE},
        HeaderName, HeaderValue, StatusCode,
    },
    response::Response,
};
use futures_util::TryStreamExt;
use serde_json::Value;
use std::io;

/// Compute the upstream Messages URL from a configured base.
///
/// - bases that already end in `/v1/` use `{base}messages`
/// - all other bases use `{base}v1/messages`
///
/// This matches the two common Anthropic-compatible conventions:
/// - direct endpoint bases such as `https://api.anthropic.com/v1/`
/// - SDK-style bases such as `https://api.fireworks.ai/inference/`
pub(super) fn anthropic_passthrough_url(base: &url::Url) -> url::Url {
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

/// Forward the inbound Anthropic Messages request to a route flagged with
/// `anthropic_format = true`. The only mutation Prism performs is rewriting
/// the `model` field; everything else — unknown fields, system blocks, tool
/// definitions, SSE events — is relayed verbatim. Auth sends both the native
/// Anthropic `x-api-key` header and an `Authorization: Bearer` header so the
/// same passthrough route can target Anthropic itself or third-party gateways
/// that only accept bearer auth on their Anthropic-compatible endpoint.
pub(super) async fn forward_anthropic_passthrough(
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
