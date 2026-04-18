use crate::provider::ProviderKind;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use url::Url;

pub type JsonMap = Map<String, Value>;

#[derive(Debug, Clone, Deserialize)]
pub struct FileConfig {
    pub port: Option<u16>,
    #[serde(default)]
    pub routes: Vec<RouteConfigSource>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RouteConfigSource {
    pub prefix: String,
    #[serde(default)]
    pub provider: Option<ProviderKind>,
    #[serde(alias = "url")]
    pub base: String,
    #[serde(default, alias = "api_key_env")]
    pub key_env: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    /// Optional default upstream model. Used when the request model string
    /// exactly equals `prefix` — resolver swaps in this value before
    /// forwarding. Enables short names like `main` for long model IDs.
    #[serde(default)]
    pub model: Option<String>,
    /// When `true`, forward the request in Anthropic Messages native format
    /// (no OpenAI translation, Anthropic-style auth headers, upstream path
    /// auto-resolved as `{base}/messages` or `{base}/v1/messages`
    /// depending on whether `base` already ends with `/v1/`). Defaults to
    /// `false`, in which case Prism translates to OpenAI
    /// `chat/completions` as usual.
    #[serde(default, alias = "anthropic-format", alias = "anthropicFormat")]
    pub anthropic_format: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct Backend {
    pub prefix: String,
    pub provider: ProviderKind,
    pub base: Url,
    pub api_key: String,
    pub credential_label: String,
    pub default_model: Option<String>,
    pub anthropic_format: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<AnthropicSystemPrompt>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
}

impl AnthropicContent {
    pub fn into_blocks(self) -> Vec<AnthropicBlock> {
        match self {
            Self::Text(text) => vec![AnthropicBlock::text(text)],
            Self::Blocks(blocks) => blocks,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicBlock {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(flatten)]
    pub fields: JsonMap,
}

impl AnthropicBlock {
    pub fn text(text: String) -> Self {
        let mut fields = JsonMap::new();
        fields.insert("text".into(), Value::String(text));
        Self {
            kind: "text".into(),
            fields,
        }
    }

    pub fn field_str(&self, key: &str) -> Option<&str> {
        self.fields.get(key)?.as_str()
    }

    pub fn field_value(&self, key: &str) -> Option<&Value> {
        self.fields.get(key)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystemPrompt {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicToolChoice {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiChatCompletionResponse {
    pub id: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<OpenAiChoice>,
    #[serde(default)]
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiChoice {
    pub message: OpenAiResponseMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiResponseMessage {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(
        default,
        rename = "reasoning_content",
        alias = "reasoning",
        alias = "reasoning_details",
        alias = "thinking"
    )]
    pub _reasoning_content: Option<Value>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OpenAiFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunctionCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct OpenAiUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiChatCompletionChunk {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<OpenAiStreamChoice>,
    #[serde(default)]
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiStreamChoice {
    #[serde(default)]
    pub delta: OpenAiDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct OpenAiDelta {
    #[serde(default)]
    pub content: Option<Value>,
    /// OpenAI-compatible providers stream reasoning under a few different
    /// field names (`reasoning_content`, `reasoning`, `reasoning_details`,
    /// `thinking`). Capture the common aliases so the Responses translator can
    /// surface them as reasoning summary events.
    #[serde(
        default,
        alias = "reasoning",
        alias = "reasoning_details",
        alias = "thinking"
    )]
    pub reasoning_content: Option<Value>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiToolCallDelta {
    #[serde(default)]
    pub index: Option<usize>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<OpenAiFunctionDelta>,
}

#[derive(Debug, Default, Deserialize)]
pub struct OpenAiFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}
