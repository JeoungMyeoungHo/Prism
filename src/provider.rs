//! Provider adapters for upstream-specific quirks.
//!
//! [`ProviderKind`] names the variants; [`ProviderAdapter`] is the trait that
//! each variant implements. Adapters decide:
//! - where to POST chat completions relative to the configured backend base,
//! - how to attach auth (bearer header vs. provider-specific header),
//! - request/response body mutations that paper over dialect differences
//!   (e.g. Fireworks drops `max_completion_tokens`, Z.AI renames tool fields).
//!
//! [`ProviderKind::Auto`] sniffs the backend URL host and dispatches to the
//! best-matching concrete adapter, falling back to OpenAI-compatible.

use crate::types::JsonMap;
use reqwest::RequestBuilder;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    #[default]
    Auto,
    // Accept all common spellings users / Builder UI may emit. Canonical serde
    // form is `open_ai_compatible` (from the snake_case auto-rename), but the
    // more natural names must also parse.
    #[serde(
        alias = "openai",
        alias = "openai_compatible",
        alias = "openai-compatible"
    )]
    OpenAiCompatible,
    Fireworks,
    #[serde(alias = "z.ai", alias = "z_ai")]
    Zai,
}

impl ProviderKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::OpenAiCompatible => "openai_compatible",
            Self::Fireworks => "fireworks",
            Self::Zai => "zai",
        }
    }

    pub fn resolve(explicit: Option<Self>, base: &Url) -> Self {
        match explicit.unwrap_or_default() {
            Self::Auto => Self::infer(base),
            provider => provider,
        }
    }

    pub fn infer(base: &Url) -> Self {
        let host = base.host_str().unwrap_or_default().to_ascii_lowercase();
        if host.contains("fireworks.ai") {
            Self::Fireworks
        } else if host.contains("z.ai") || host.contains("bigmodel.cn") || host.contains("zhipu") {
            Self::Zai
        } else {
            Self::OpenAiCompatible
        }
    }

    pub fn adapter(self) -> ProviderAdapter {
        ProviderAdapter { kind: self }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderAdapter {
    kind: ProviderKind,
}

impl ProviderAdapter {
    pub fn chat_completions_url(self, base: &Url) -> Url {
        // `base` is validated at config load and always parses as an absolute
        // URL; "chat/completions" is a literal relative ref, so `join` cannot
        // fail here.
        base.join("chat/completions")
            .expect("literal relative ref joined against validated base URL")
    }

    pub fn apply_auth(self, builder: RequestBuilder, api_key: &str) -> RequestBuilder {
        builder.bearer_auth(api_key)
    }

    pub fn adapt_request(self, object: &mut JsonMap) -> Vec<String> {
        let mut notes = Vec::new();

        self.normalize_reasoning_history(object, &mut notes);

        match self.kind {
            ProviderKind::Auto | ProviderKind::OpenAiCompatible => {}
            ProviderKind::Fireworks => {
                if object.contains_key("max_completion_tokens") && object.contains_key("max_tokens")
                {
                    object.remove("max_completion_tokens");
                    notes.push(
                        "Fireworks treats `max_completion_tokens` as an alias for `max_tokens`, so Prism only forwarded `max_tokens`."
                            .into(),
                    );
                }
            }
            ProviderKind::Zai => {
                if object.contains_key("max_completion_tokens") {
                    object.remove("max_completion_tokens");
                    notes.push(
                        "Z.AI expects `max_tokens`, so Prism removed `max_completion_tokens`."
                            .into(),
                    );
                }

                if let Some(tool_choice) = object.get("tool_choice").cloned() {
                    let needs_downgrade =
                        !matches!(tool_choice, Value::String(ref value) if value == "auto");
                    if needs_downgrade {
                        object.insert("tool_choice".into(), Value::String("auto".into()));
                        notes.push(
                            "Z.AI currently supports `tool_choice = auto` only, so Prism downgraded the requested tool choice."
                                .into(),
                        );
                    }
                }

                if let Some(Value::Array(stop_sequences)) = object.get_mut("stop") {
                    if stop_sequences.len() > 1 {
                        let first = stop_sequences.first().cloned().unwrap_or(Value::Null);
                        stop_sequences.clear();
                        stop_sequences.push(first);
                        notes.push(
                            "Z.AI currently supports a single stop sequence, so Prism kept only the first entry."
                                .into(),
                        );
                    }
                }

                if matches!(object.get("stream"), Some(Value::Bool(true)))
                    && matches!(object.get("tools"), Some(Value::Array(tools)) if !tools.is_empty())
                {
                    object.insert("tool_stream".into(), Value::Bool(true));
                }
            }
        }

        notes
    }

    fn normalize_reasoning_history(self, object: &mut JsonMap, notes: &mut Vec<String>) {
        let Some(Value::Array(messages)) = object.get_mut("messages") else {
            return;
        };

        let supports_reasoning = matches!(self.kind, ProviderKind::Fireworks | ProviderKind::Zai);
        let mut removed_any = false;

        for message in messages {
            let Some(message_object) = message.as_object_mut() else {
                continue;
            };

            if supports_reasoning {
                continue;
            }

            if message_object.remove("reasoning_content").is_some() {
                removed_any = true;
            }
        }

        if removed_any {
            notes.push(
                "This provider does not advertise `reasoning_content` in chat history, so Prism dropped Anthropic thinking blocks from prior turns."
                    .into(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ProviderKind;
    use serde_json::json;

    #[test]
    fn fireworks_keeps_only_one_max_tokens_field() {
        let mut request = json!({
            "model": "accounts/fireworks/models/demo",
            "messages": [],
            "max_tokens": 128,
            "max_completion_tokens": 128
        })
        .as_object()
        .cloned()
        .unwrap();

        let notes = ProviderKind::Fireworks
            .adapter()
            .adapt_request(&mut request);

        assert_eq!(request.get("max_tokens"), Some(&json!(128)));
        assert!(!request.contains_key("max_completion_tokens"));
        assert_eq!(notes.len(), 1);
    }

    #[test]
    fn zai_downgrades_tool_choice_and_enables_tool_stream() {
        let mut request = json!({
            "model": "glm-4.5",
            "messages": [],
            "stream": true,
            "tools": [{ "type": "function", "function": { "name": "lookup", "parameters": {} } }],
            "tool_choice": "required",
            "stop": ["one", "two"],
            "max_tokens": 64,
            "max_completion_tokens": 64
        })
        .as_object()
        .cloned()
        .unwrap();

        let notes = ProviderKind::Zai.adapter().adapt_request(&mut request);

        assert_eq!(request.get("tool_choice"), Some(&json!("auto")));
        assert_eq!(request.get("tool_stream"), Some(&json!(true)));
        assert_eq!(request.get("stop"), Some(&json!(["one"])));
        assert!(!request.contains_key("max_completion_tokens"));
        assert!(notes.len() >= 3);
    }

    #[test]
    fn auto_provider_infers_known_hosts() {
        let fireworks = url::Url::parse("https://api.fireworks.ai/inference/v1/").unwrap();
        let zai = url::Url::parse("https://api.z.ai/api/paas/v4/").unwrap();
        let generic = url::Url::parse("https://example.com/v1/").unwrap();

        assert_eq!(ProviderKind::infer(&fireworks), ProviderKind::Fireworks);
        assert_eq!(ProviderKind::infer(&zai), ProviderKind::Zai);
        assert_eq!(
            ProviderKind::infer(&generic),
            ProviderKind::OpenAiCompatible
        );
    }
}
