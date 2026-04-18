# Prism

**English** · [한국어](./README.ko.md)

![Prism Hero](./assets/prism-hero.png)

Prism is a Rust proxy that puts many OpenAI-compatible upstream providers behind a single Anthropic-compatible endpoint (`POST /v1/messages`) and routes each request to the right upstream based on the model string. The goal: let a client like Claude Code talk to one base URL while actually fanning out to different providers per model.

```text
Claude Code  →  Prism  →  Z.AI / Fireworks / Groq / OpenAI / ...
                  (3-rule resolver, Anthropic ↔ OpenAI translation)
```

**Non-affiliation.** Prism is an independent project. It is **not** affiliated with or operated by Anthropic, OpenAI, Google, Groq, Fireworks, Z.AI, MiniMax, or any other provider. Provider names, URLs, model IDs, and Claude Code integration notes in this document are for interoperability reference only.

---

## What it solves

- **One base URL, many providers.** Point `ANTHROPIC_BASE_URL` at Prism and each of Claude Code's tier overrides (`ANTHROPIC_DEFAULT_HAIKU_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL`) can resolve to a different provider.
- **Model-name disambiguation.** When the same model id (e.g. `deepseek-v3`) is served by several providers, `fw/deepseek-v3` vs `groq/deepseek-v3` disambiguates deterministically.
- **Short aliases for long model ids.** Expose `accounts/fireworks/models/kimi-k2p5` as a one-word name like `main`.
- **Format translation.** Converts Anthropic Messages requests into OpenAI `chat/completions`, and translates the response and SSE events back. Clients always see the Anthropic shape.

---

## Quick Start

```bash
cargo run
```

Starts on port 8088 by default:

- Builder UI: <http://127.0.0.1:8088/>
- Health:     <http://127.0.0.1:8088/healthz>
- Proxy:      <http://127.0.0.1:8088/v1/messages>

It's fine to run with zero routes — use the Builder UI to configure, download a TOML, then relaunch with:

```bash
PRISM_CONFIG=./prism.config.toml cargo run
```

---

## Routing Rules

The request `model` field is resolved once, in this order:

1. **Exact-default** — `model == route.prefix` AND the route has `model = "..."`. The request goes to that route and the upstream receives the route's `model`.
   - Example: route `prefix = "main", model = "accounts/.../kimi-k2p5"` → request `main` → upstream `accounts/.../kimi-k2p5`
2. **Namespace** — `model` contains `/` and the first segment equals some route's `prefix`. That route is used and the upstream receives everything after the first `/`.
   - Example: `fw/deepseek-v3` → route `fw`, upstream `deepseek-v3`
   - Example: `fw/accounts/fireworks/models/llama-v3p1-8b-instruct` → only the first `fw/` is stripped
3. **Raw longest-prefix** — otherwise, the route whose `prefix` is the longest string-prefix of `model` wins. The upstream receives the original string unchanged.
4. If none of the three match, Prism returns `400 invalid_request_error`. The response body includes the catalog of available prefixes / default models.

Boot-time validation:

- Missing / empty `prefix` → error
- Duplicate `prefix` across routes → error
- Missing both `api_key` and `key_env` → error

The Resolver Simulator in the Builder UI dry-runs the same three rules against any model string.

---

## Config File (TOML)

Each route needs `prefix` (required), `base` (required), and auth (`api_key` or `key_env`).
Leave `provider` unset or `auto` to infer the adapter from the base URL host.
An optional `model` becomes the upstream model for exact-default matches.

```toml
port = 8088

# 1) Short alias for a long model id — request `main` → Fireworks / accounts/fireworks/models/kimi-k2p5
[[routes]]
prefix   = "main"
provider = "fireworks"
base     = "https://api.fireworks.ai/inference/v1/"
key_env  = "FIREWORKS_API_KEY"
model    = "accounts/fireworks/models/kimi-k2p5"

# 2) Pass-through — `glm-4.5`, `glm-4-flash` all forwarded to Z.AI unchanged
[[routes]]
prefix   = "glm"
provider = "zai"
base     = "https://api.z.ai/api/coding/paas/v4/"
key_env  = "ZAI_API_KEY"

# 3) Disambiguate the same model across providers — `fw/deepseek-v3` vs `groq/deepseek-v3`
[[routes]]
prefix   = "fw"
provider = "fireworks"
base     = "https://api.fireworks.ai/inference/v1/"
key_env  = "FIREWORKS_API_KEY"

[[routes]]
prefix   = "groq"
provider = "openai"
base     = "https://api.groq.com/openai/v1/"
key_env  = "GROQ_API_KEY"
```

Or configure routes entirely from the environment:

```bash
PRISM_ROUTES="prefix=main,provider=fireworks,base=https://api.fireworks.ai/inference/v1,key_env=FIREWORKS_API_KEY,model=accounts/fireworks/models/kimi-k2p5;prefix=glm,provider=zai,base=https://api.z.ai/api/coding/paas/v4,key_env=ZAI_API_KEY" \
PRISM_PORT=8088 \
cargo run
```

Supported provider adapters (body-normalization layer):

| `provider`       | Behaviour                                                                                 |
| ---------------- | ----------------------------------------------------------------------------------------- |
| `auto` (default) | Inferred from the base URL host — picks zai / fireworks / openai                          |
| `openai`         | OpenAI-compatible baseline — Groq, DeepSeek, Together, OpenRouter, xAI, Mistral, …        |
| `fireworks`      | Resolves `max_tokens` / `max_completion_tokens` collisions                                |
| `zai`            | Forces `tool_choice = auto`, enables tool streaming, normalizes `max_tokens`              |

### `anthropic_format` — Anthropic Messages passthrough

Independently of `provider`, a route can set the boolean flag
`anthropic_format = true` to **bypass the OpenAI translation layer
entirely**. When set:

- The inbound Anthropic Messages body is forwarded verbatim (only the
  `model` field is rewritten per the routing rules above).
- Auth sends `x-api-key: <key>`, `Authorization: Bearer <key>`, and `anthropic-version: 2023-06-01`.
- The upstream endpoint is auto-resolved from `base`: `{base}/messages`
  when `base` already ends in `/v1/`, otherwise `{base}/v1/messages`.
- The response and SSE stream are relayed byte-for-byte — Prism does not
  parse or translate anything.
- Provider-specific body tweaks (`fireworks` / `zai` quirks) are skipped
  because the adapter never runs on this path.

Use this for routing Claude Code straight to `api.anthropic.com`, or to any
gateway that speaks the Anthropic Messages API natively. `provider` is
purely a label in this mode (you can leave it on `auto`).

Example TOML:

```toml
[[routes]]
prefix = "claude"
base = "https://api.anthropic.com/v1/"
key_env = "ANTHROPIC_API_KEY"
anthropic_format = true
```

Request `claude/claude-sonnet-4-5` will land on the Anthropic Messages API
with upstream model `claude-sonnet-4-5`, no translation in the way.

---

## Claude Code integration

Claude Code supports a gateway setup through `ANTHROPIC_BASE_URL`. Use the Builder UI's **Outputs** section to preview and copy a ready-to-use `settings.json` and drop it at:

- Project-local:               `.claude/settings.local.json`
- User (macOS / Linux / WSL):   `~/.claude/settings.json`
- User (Windows):               `%USERPROFILE%\.claude\settings.json`

Example:

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8088",
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL":  "glm-4-flash",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "main",
    "ANTHROPIC_DEFAULT_OPUS_MODEL":   "glm-4.6",
    "ANTHROPIC_SMALL_FAST_MODEL":     "glm-4-flash"
  },
  "model": "main"
}
```

What each slot is for:

| Slot                              | Purpose                                                                |
| --------------------------------- | ---------------------------------------------------------------------- |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL`   | Background tasks Claude Code requests at "haiku" tier                  |
| `ANTHROPIC_DEFAULT_SONNET_MODEL`  | Main coding model ("sonnet" tier) — the one you'll hit most            |
| `ANTHROPIC_DEFAULT_OPUS_MODEL`    | Heavy lifting Claude Code promotes to "opus" tier                      |
| `ANTHROPIC_SMALL_FAST_MODEL`      | Tool routing / classifier calls (the lightest tier). Typically mirrors Haiku |
| top-level `"model"`               | Initial `/model` selection when Claude Code starts. Typically mirrors Sonnet |

The Builder UI auto-mirrors Haiku into `ANTHROPIC_SMALL_FAST_MODEL` and Sonnet into the top-level `"model"`. You can of course edit the downloaded JSON by hand to set them independently.

The value you put into any of those slots follows the same three routing rules above:

| Value              | Matching route                                         | Upstream model sent                  |
| ------------------ | ------------------------------------------------------ | ------------------------------------ |
| `main`             | `prefix = "main", model = "accounts/.../kimi-k2p5"`    | `accounts/.../kimi-k2p5` (exact-default) |
| `fw/deepseek-v3`   | `prefix = "fw"`                                        | `deepseek-v3` (namespace)            |
| `glm-4.5`          | `prefix = "glm"`                                       | `glm-4.5` (raw prefix, unchanged)    |

Note: `ANTHROPIC_API_KEY` is a placeholder — Prism doesn't check inbound auth. Put anything non-empty.

---

## Builder UI

At <http://127.0.0.1:8088/>:

- Add / edit / remove routes — port, base URL, API key (inline ↔ env toggle), default model, each as a collapsible row
- **Resolver Simulator** — type any model string, see which rule matches, which route receives it, and the upstream model name
- TOML preview / copy / download; browser `localStorage` autosave
- One-shot upstream `chat/completions` test with the route's base + key
- Anthropic SSE streaming playground that validates the `message_start` / `content_block_*` / `message_delta` / `message_stop` sequence Prism emits
- Claude Code `settings.json` preview / copy / download
- English default, Korean toggle
- Provider reference links (non-affiliated, informational)

The Builder is a single `static/index.html` served by Prism at `/`. Route editing and TOML generation also work when you open the file directly with `file://` (the Resolver Simulator and upstream tests need the Prism server).

---

## Builder Presets (`static/presets.js`)

The **Preset** dropdown at the top-right of each route card is backed by a single file:

```js
window.PRISM_PRESETS = [
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
  // ... add more
];
```

- Applying a preset fills the route's **`base`, `provider`, and `key_env` (only if the route's env is empty)**. It never touches `prefix`, `default model`, or the api key value.
- 31 presets ship by default, grouped roughly as:
  - **Anthropic-native** (passthrough): Anthropic, Fireworks (Anthropic native), OpenRouter (Anthropic native), LMRouter (Anthropic native), OfoxAI (Anthropic native), Anannas (Anthropic native), LLMGateway (Anthropic native), NagaAI (Anthropic native), Shannon AI (Anthropic native).
  - **Primary coding / Claude Code targets**: Z.AI (coding), Fireworks, OpenRouter.
  - **OpenAI family**: OpenAI, Google Gemini (OAI-compat), xAI (Grok), Mistral.
  - **Fast inference clouds**: Groq, Cerebras, SambaNova, Together AI, Hyperbolic, Nebius.
  - **Chinese model providers**: DeepSeek, Moonshot (Kimi), Qwen (DashScope intl), MiniMax.
  - **Search / specialty**: Perplexity, Cohere.
  - **Local / self-hosted**: Ollama, LM Studio, vLLM.
- Prism serves the file at `/presets.js` (embedded at compile time). Edit `static/presets.js` and rebuild with `cargo run` to pick up changes.
- If `static/index.html` is opened through `file://`, the sibling `presets.js` is loaded by the `<script>` tag directly — no server required.

Provider URLs and model line-ups change over time. Double-check each provider's official docs before relying on an entry.

---

## API Endpoints

| Method | Path                    | Description                                                         |
| ------ | ----------------------- | ------------------------------------------------------------------- |
| GET    | `/`                     | Builder UI                                                          |
| GET    | `/healthz`              | `{"status":"ok"}`                                                   |
| GET    | `/presets.js`           | Builder presets (compile-time embedded)                             |
| POST   | `/v1/messages`          | Anthropic Messages — routing + translation + SSE relay              |
| POST   | `/v1/responses`         | OpenAI Responses → chat/completions translation                       |
| POST   | `/api/test-upstream`    | Builder one-shot upstream `chat/completions` ping                   |
| POST   | `/api/test-stream`      | Builder SSE streaming playground                                    |
| POST   | `/api/resolve-preview`  | Resolver Simulator dry-run                                          |

---

## Translation notes & limits

- Anthropic `thinking` blocks are dropped on the user path (carried as `reasoning_content` on the assistant path). `image` blocks translate natively. `document` blocks with `source.type = "text"` or `"content"` are expanded inline (text + nested images preserved); `base64` / `url` / `file` document sources, plus `audio` / `video` / `file` blocks, are still lowered to textual notes where no OpenAI-compatible equivalent exists.
- `tool_use` / `tool_result` translate in both directions. Some providers ship slightly different tool-streaming formats; the adapter layer smooths those out.
- `/v1/responses` supports both non-streaming and streaming. The streaming path emits full Responses-shaped SSE events (`response.created` / `response.output_item.added` / `response.output_text.delta` / `response.function_call_arguments.delta` / `response.reasoning_summary_text.delta` / `response.completed`). Upstream `reasoning_content` deltas are surfaced as `reasoning` items preceding the message item.
- Prism does **not** check inbound authentication. Run it locally only.

---

## Security / operational notes

- The Builder stores routes in `localStorage` and writes TOML downloads with **API keys in plaintext**. Use this on personal dev machines only.
- Don't commit TOML config files with inline keys; prefer `key_env` so secrets live in environment variables.
- Prism binds to `0.0.0.0:8088` by default. Don't expose this port to untrusted networks without an in-front reverse proxy / firewall.

---

## Development

```bash
cargo check                                  # type/lint
cargo run                                    # serve on 8088
PRISM_CONFIG=./prism.config.toml cargo run   # with a config file
RUST_LOG=prism=debug cargo run               # verbose routing logs
```

Source layout:

```
src/
  main.rs       — axum router + bootstrap
  config.rs     — TOML parsing, env routes, validation
  router.rs     — 3-rule resolver (exact-default / namespace / raw-prefix)
  proxy.rs      — Anthropic ↔ OpenAI translation, SSE relay
  provider.rs   — provider adapter (zai / fireworks / openai / auto)
  ui.rs         — Builder API handlers (tests, simulator, presets.js)
  types.rs      — domain types
static/
  index.html    — Builder UI (single file, i18n en/ko)
  presets.js    — route preset catalog (edit this)
```

---

## License

Licensed under the [Apache License 2.0](./LICENSE). Copyright belongs to the contributors. If the distribution includes a `NOTICE` file, keep it alongside the license.
