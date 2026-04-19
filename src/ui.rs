//! Builder UI and diagnostic endpoints.
//!
//! Two responsibilities:
//! - Serve the static Builder (inlined [`BUILDER_HTML`] and presets) at `/`
//!   and `/builder`. The Builder runs entirely in the browser and writes TOML
//!   that the user pastes into `prism.toml`.
//! - Provide `/api/test-upstream`, `/api/test-stream`, `/api/resolve-preview`
//!   so the Builder can verify a backend's API key and URL without restarting
//!   the server.
//!
//! These endpoints run untranslated probe payloads against the provider
//! adapter; they do **not** go through the main proxy translators.

use crate::{
    provider::ProviderKind,
    proxy::{forward_request_to_backend, ApiError},
    router::ModelRouter,
    types::{AnthropicContent, AnthropicMessage, AnthropicMessagesRequest, AnthropicTool, Backend},
};
use axum::{
    extract::Json,
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashSet;
use tracing::warn;
use url::Url;

pub async fn builder() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

/// Serve `static/presets.js` so the Builder UI can load the preset list via a
/// plain `<script>` tag — works the same way when the HTML is opened via
/// `http://…:8088/` (this handler) or directly from disk (`file://`).
/// Content is embedded at compile time; edit `static/presets.js` and rebuild.
pub async fn presets_js() -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        include_str!("../static/presets.js"),
    )
}

#[derive(Debug, Deserialize)]
pub struct UpstreamTestRequest {
    pub base: String,
    pub api_key: String,
    pub model: String,
    #[serde(default)]
    pub provider: Option<ProviderKind>,
}

#[derive(Debug, Deserialize)]
pub struct StreamPlaygroundRequest {
    pub base: String,
    pub api_key: String,
    pub model: String,
    #[serde(default)]
    pub provider: Option<ProviderKind>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub scenario: Option<String>,
}

pub async fn test_upstream(
    Json(request): Json<UpstreamTestRequest>,
) -> Result<Json<Value>, UiError> {
    let backend = build_preview_backend(
        &request.base,
        &request.api_key,
        request.provider,
        &request.model,
    )?;

    let mut payload = json!({
        "model": request.model.trim(),
        "messages": [
            {
                "role": "user",
                "content": "Reply with the single word pong."
            }
        ],
        "max_completion_tokens": 24,
        "max_tokens": 24
    });

    // `payload` just built via `json!({ ... })` — always Value::Object.
    let object = payload
        .as_object_mut()
        .expect("json! literal always yields Value::Object");
    let adapter_notes = backend.provider.adapter().adapt_request(object);

    let adapter = backend.provider.adapter();
    let response = adapter
        .apply_auth(
            reqwest::Client::new().post(adapter.chat_completions_url(&backend.base)),
            &backend.api_key,
        )
        .json(&payload)
        .send()
        .await
        .map_err(|error| UiError::bad_gateway(format!("Failed to reach upstream API: {error}")))?;

    let status = response.status();
    let raw_body = response.text().await.map_err(|error| {
        UiError::bad_gateway(format!("Failed to read upstream response: {error}"))
    })?;

    let parsed = serde_json::from_str::<Value>(&raw_body).ok();
    let reply = parsed.as_ref().and_then(extract_reply_text);
    let upstream_error = parsed
        .as_ref()
        .and_then(|value| value.get("error"))
        .map(extract_message)
        .filter(|message| !message.is_empty());

    if !status.is_success() {
        warn!(
            target: "prism::upstream",
            %status,
            body = %raw_body,
            "test-upstream: upstream error"
        );
    }

    Ok(Json(json!({
        "ok": status.is_success(),
        "status": status.as_u16(),
        "provider": backend.provider.as_str(),
        "reply": reply,
        "error": upstream_error,
        "adapter_notes": adapter_notes,
        "body_preview": truncate_for_preview(&raw_body, 1200),
    })))
}

pub async fn test_stream(
    Json(request): Json<StreamPlaygroundRequest>,
) -> Result<Response, UiError> {
    let backend = build_preview_backend(
        &request.base,
        &request.api_key,
        request.provider,
        &request.model,
    )?;
    let anthropic_request = build_stream_request(
        request.model.trim(),
        request.prompt.as_deref(),
        request.scenario.as_deref(),
    )?;

    let upstream_model = anthropic_request.model.clone();
    forward_request_to_backend(
        &reqwest::Client::new(),
        &backend,
        &[],
        anthropic_request,
        upstream_model,
    )
    .await
    .map_err(UiError::from_api_error)
}

#[derive(Debug, Deserialize)]
pub struct ResolvePreviewRequest {
    #[serde(default)]
    pub routes: Vec<PreviewRoute>,
    pub model: String,
}

#[derive(Debug, Deserialize)]
pub struct PreviewRoute {
    pub prefix: String,
    #[serde(default)]
    pub provider: Option<ProviderKind>,
    #[serde(default)]
    pub base: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

pub async fn resolve_preview(Json(request): Json<ResolvePreviewRequest>) -> Json<Value> {
    let model = request.model.trim();
    if model.is_empty() {
        return Json(json!({
            "ok": false,
            "reason": "empty_input",
        }));
    }

    let mut backends: Vec<Backend> = Vec::new();
    let mut seen_prefixes: HashSet<String> = HashSet::new();
    let mut warnings: Vec<String> = Vec::new();

    for (idx, route) in request.routes.iter().enumerate() {
        let prefix = route.prefix.trim();
        if prefix.is_empty() {
            warnings.push(format!("route #{idx} has an empty prefix"));
            continue;
        }
        if !seen_prefixes.insert(prefix.to_string()) {
            warnings.push(format!("duplicate prefix `{prefix}`"));
            continue;
        }
        let base_raw = route
            .base
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .unwrap_or("https://example.invalid/");
        let base = match normalize_base_url(base_raw) {
            Ok(url) => url,
            // Literal well-formed URL — parse cannot fail.
            Err(_) => Url::parse("https://example.invalid/").expect("literal URL parses"),
        };
        let provider = ProviderKind::resolve(route.provider, &base);
        let default_model = route
            .model
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(ToString::to_string);
        backends.push(Backend {
            prefix: prefix.to_string(),
            provider,
            base,
            api_key: "preview".to_string(),
            credential_label: "preview".to_string(),
            default_model,
            anthropic_format: false, // irrelevant for resolver-only dry-run
        });
    }

    let router = ModelRouter::new(backends);
    let resolution = router.resolve(model);

    let payload = match resolution {
        Some(res) => json!({
            "ok": true,
            "matched_by": res.matched_by.as_str(),
            "upstream_model": res.upstream_model,
            "backend": {
                "prefix": res.backend.prefix,
                "provider": res.backend.provider.as_str(),
                "base": res.backend.base.to_string(),
                "default_model": res.backend.default_model,
            },
            "warnings": warnings,
            "catalog": router.describe_catalog(),
        }),
        None => json!({
            "ok": false,
            "reason": "no_match",
            "matched_by": Value::Null,
            "warnings": warnings,
            "catalog": router.describe_catalog(),
        }),
    };

    Json(payload)
}

#[derive(Debug)]
pub struct UiError {
    status: StatusCode,
    message: String,
}

impl UiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn bad_gateway(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: message.into(),
        }
    }

    fn from_api_error(error: ApiError) -> Self {
        Self {
            status: error.status(),
            message: error.message().to_string(),
        }
    }
}

impl IntoResponse for UiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "ok": false,
                "error": self.message,
            })),
        )
            .into_response()
    }
}

fn normalize_base_url(raw: &str) -> Result<Url, url::ParseError> {
    let mut url = Url::parse(raw)?;
    if !url.path().ends_with('/') {
        let current = url.path().trim_end_matches('/');
        let normalized = if current.is_empty() {
            "/".to_string()
        } else {
            format!("{current}/")
        };
        url.set_path(&normalized);
    }
    Ok(url)
}

fn extract_reply_text(value: &Value) -> Option<String> {
    let content = value
        .get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?;

    match content {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                if let Some(part) = part.get("text").and_then(Value::as_str) {
                    text.push_str(part);
                } else if let Some(part) = part.as_str() {
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

fn extract_message(value: &Value) -> String {
    value
        .get("message")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .unwrap_or_else(|| value.to_string())
}

fn truncate_for_preview(text: &str, limit: usize) -> String {
    let mut truncated = text.chars().take(limit).collect::<String>();
    if text.chars().count() > limit {
        truncated.push_str("\n...(truncated)");
    }
    truncated
}

fn build_preview_backend(
    base: &str,
    api_key: &str,
    provider: Option<ProviderKind>,
    model: &str,
) -> Result<Backend, UiError> {
    let base = base.trim();
    let api_key = api_key.trim();
    let model = model.trim();

    if base.is_empty() {
        return Err(UiError::bad_request("Base URL is required."));
    }
    if api_key.is_empty() {
        return Err(UiError::bad_request("API key is required."));
    }
    if model.is_empty() {
        return Err(UiError::bad_request("Test model is required."));
    }

    let base = normalize_base_url(base)
        .map_err(|error| UiError::bad_request(format!("Invalid base URL `{base}`: {error}")))?;
    let provider = ProviderKind::resolve(provider, &base);

    Ok(Backend {
        prefix: model.to_string(),
        provider,
        base,
        api_key: api_key.to_string(),
        credential_label: "builder api_key".into(),
        default_model: None,
        anthropic_format: false, // builder upstream-ping always uses OpenAI path
    })
}

fn build_stream_request(
    model: &str,
    prompt: Option<&str>,
    scenario: Option<&str>,
) -> Result<AnthropicMessagesRequest, UiError> {
    if model.trim().is_empty() {
        return Err(UiError::bad_request("Stream test model is required."));
    }

    let scenario = scenario.unwrap_or("text");
    let prompt = prompt
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(match scenario {
        "tool" => {
            "Use the provided tool to answer this request. What is the weather in Seoul right now?"
        }
        _ => "Reply with a short greeting from the streaming test.",
    });

    let tools = if scenario == "tool" {
        Some(vec![AnthropicTool {
            name: "lookup_weather".into(),
            description: Some("Return a mock weather lookup for a city.".into()),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City to look up."
                    }
                },
                "required": ["city"]
            }),
        }])
    } else {
        None
    };

    Ok(AnthropicMessagesRequest {
        model: model.to_string(),
        messages: vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Text(prompt.to_string()),
        }],
        system: None,
        max_tokens: Some(256),
        stream: Some(true),
        temperature: Some(0.2),
        top_p: None,
        top_k: None,
        stop_sequences: None,
        tools,
        tool_choice: None,
        metadata: None,
        service_tier: None,
        user: None,
        thinking: None,
        stream_options: None,
    })
}
