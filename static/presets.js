// Prism Builder — Route preset list.
//
// Edit this file alone to manage the "Preset" dropdown on each route card in
// the Builder UI. Reload the page to see changes (the server embeds this at
// compile time, so rebuild with `cargo run` when served through Prism).
//
// Each entry fills these route fields when picked:
//   - base             (URL)
//   - provider         ("auto" | "openai" | "fireworks" | "zai")
//   - key_env          (optional; only filled if the route's env field is empty)
//   - anthropic_format (optional bool; when true the route bypasses the
//                       OpenAI translation and relays Anthropic Messages
//                       natively, using x-api-key + anthropic-version auth
//                       and the {base}/messages path)
//
// Presets NEVER touch prefix, default model, or the api key value itself.
//
// Base URLs and available models change over time. Verify against each
// provider's official docs before relying on any entry below. Prism has no
// affiliation with any of these providers.

window.PRISM_PRESETS = [
  // ── Anthropic-native (passthrough) ─────────────────────────────────────
  // `anthropic_format: true` bypasses Prism's OpenAI translation. Useful for
  // proxying Claude Code straight through Prism with logging/routing but no
  // reformatting. Provider stays "auto" — the flag, not the adapter, drives
  // the protocol.
  {
    name: "Anthropic (native)",
    base: "https://api.anthropic.com/v1/",
    provider: "auto",
    key_env: "ANTHROPIC_API_KEY",
    anthropic_format: true,
  },

  // ── Primary coding / Claude Code targets ───────────────────────────────
  // The main reason most people run Prism: swap Claude Code's upstream to
  // a non-Anthropic coding model without changing the client.
  {
    name: "Z.AI (coding)",
    base: "https://api.z.ai/api/coding/paas/v4/",
    provider: "zai",
    key_env: "ZAI_API_KEY",
  },
  {
    name: "Fireworks",
    base: "https://api.fireworks.ai/inference/v1/",
    provider: "fireworks",
    key_env: "FIREWORKS_API_KEY",
  },
  {
    name: "OpenRouter (aggregator)",
    base: "https://openrouter.ai/api/v1/",
    provider: "openai",
    key_env: "OPENROUTER_API_KEY",
  },

  // ── OpenAI family (and OpenAI-shaped frontier models) ──────────────────
  {
    name: "OpenAI",
    base: "https://api.openai.com/v1/",
    provider: "openai",
    key_env: "OPENAI_API_KEY",
  },
  {
    name: "Google Gemini (OAI-compat)",
    base: "https://generativelanguage.googleapis.com/v1beta/openai/",
    provider: "openai",
    key_env: "GEMINI_API_KEY",
  },
  {
    name: "xAI (Grok)",
    base: "https://api.x.ai/v1/",
    provider: "openai",
    key_env: "XAI_API_KEY",
  },
  {
    name: "Mistral",
    base: "https://api.mistral.ai/v1/",
    provider: "openai",
    key_env: "MISTRAL_API_KEY",
  },

  // ── Fast / specialty inference clouds ──────────────────────────────────
  // All OpenAI-shaped, typically hosting open-weight checkpoints behind
  // custom accelerators.
  {
    name: "Groq",
    base: "https://api.groq.com/openai/v1/",
    provider: "openai",
    key_env: "GROQ_API_KEY",
  },
  {
    name: "Cerebras",
    base: "https://api.cerebras.ai/v1/",
    provider: "openai",
    key_env: "CEREBRAS_API_KEY",
  },
  {
    name: "SambaNova",
    base: "https://api.sambanova.ai/v1/",
    provider: "openai",
    key_env: "SAMBANOVA_API_KEY",
  },
  {
    name: "Together AI",
    base: "https://api.together.xyz/v1/",
    provider: "openai",
    key_env: "TOGETHER_API_KEY",
  },
  {
    name: "Hyperbolic",
    base: "https://api.hyperbolic.xyz/v1/",
    provider: "openai",
    key_env: "HYPERBOLIC_API_KEY",
  },
  {
    name: "Nebius",
    base: "https://api.studio.nebius.ai/v1/",
    provider: "openai",
    key_env: "NEBIUS_API_KEY",
  },

  // ── Chinese model providers (own frontier models) ──────────────────────
  {
    name: "DeepSeek",
    base: "https://api.deepseek.com/v1/",
    provider: "openai",
    key_env: "DEEPSEEK_API_KEY",
  },
  {
    name: "Moonshot (Kimi)",
    base: "https://api.moonshot.ai/v1/",
    provider: "openai",
    key_env: "MOONSHOT_API_KEY",
  },
  {
    name: "Qwen (DashScope intl)",
    base: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/",
    provider: "openai",
    key_env: "DASHSCOPE_API_KEY",
  },
  {
    name: "MiniMax",
    base: "https://api.minimax.io/v1/",
    provider: "openai",
    key_env: "MINIMAX_API_KEY",
  },

  // ── Search-augmented / specialty ───────────────────────────────────────
  {
    name: "Perplexity",
    base: "https://api.perplexity.ai/",
    provider: "openai",
    key_env: "PERPLEXITY_API_KEY",
  },
  {
    name: "Cohere (OAI-compat)",
    base: "https://api.cohere.ai/compatibility/v1/",
    provider: "openai",
    key_env: "COHERE_API_KEY",
  },

  // ── Local / self-hosted (no real auth; key_env is a placeholder) ───────
  {
    name: "Ollama (local)",
    base: "http://localhost:11434/v1/",
    provider: "openai",
    key_env: "OLLAMA_API_KEY",
  },
  {
    name: "LM Studio (local)",
    base: "http://localhost:1234/v1/",
    provider: "openai",
    key_env: "LMSTUDIO_API_KEY",
  },
  {
    name: "vLLM (local, OpenAI-compat)",
    base: "http://localhost:8000/v1/",
    provider: "openai",
    key_env: "VLLM_API_KEY",
  },
];
