//! SSE translation pipelines.
//!
//! Each submodule owns a stateful translator that folds an OpenAI
//! Chat Completion chunk stream into the target protocol's SSE shape:
//!
//! - [`anthropic::AnthropicStreamTranslator`] — Anthropic Messages SSE.
//! - [`responses::ResponsesStreamTranslator`] — OpenAI Responses SSE.
//!
//! This module itself only hosts the two formatting helpers and the
//! upstream-text extractor both translators share.

mod anthropic;
mod responses;

pub(crate) use anthropic::AnthropicStreamTranslator;
pub(crate) use responses::ResponsesStreamTranslator;

use axum::body::Bytes;
use serde_json::Value;

pub(crate) fn sse_event(event: &str, payload: &Value) -> Bytes {
    let json = serde_json::to_string(payload).unwrap_or_else(|_| "{}".into());
    Bytes::from(format!("event: {event}\ndata: {json}\n\n"))
}

pub(crate) fn sse_named_event(event: &str, payload: &Value) -> Bytes {
    let json = serde_json::to_string(payload).unwrap_or_else(|_| "{}".into());
    Bytes::from(format!("event: {event}\ndata: {json}\n\n"))
}

pub(crate) fn extract_stream_text(content: &Value) -> Option<String> {
    super::extract_textish_value(content)
}
