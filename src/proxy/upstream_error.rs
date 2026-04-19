//! Upstream error body → [`ApiError`] normalization.

use super::ApiError;
use axum::http::StatusCode;
use serde_json::Value;
use tracing::warn;

pub(super) async fn read_upstream_error(response: reqwest::Response) -> ApiError {
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

pub(super) fn extract_error_message(error: &Value) -> String {
    error
        .get("message")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| error.to_string())
}
