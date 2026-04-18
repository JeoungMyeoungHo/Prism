# Prism

[English](./README.md) · **한국어**

![Prism Hero](./assets/prism-hero.png)

Prism은 Anthropic 호환 엔드포인트(`POST /v1/messages`) 하나 뒤에 여러 OpenAI 호환 provider를 연결해두는 Rust 프록시입니다. 
들어오는 요청의 모델명을 보고 적절한 upstream으로 넘기는 역할을 합니다. Claude Code 같은 클라이언트 입장에서는 base URL 한 개만 설정하면 되지만, 실제로는 모델마다 다른 provider가 응답하게 됩니다.

```text
Claude Code  →  Prism  →  Z.AI / Fireworks / Groq / OpenAI / ...
                  (3단 리졸버, Anthropic ↔ OpenAI 포맷 변환)
```

**비제휴 고지.** Prism은 독립 프로젝트입니다. Anthropic, OpenAI, Google, Groq, Fireworks, Z.AI, MiniMax 등 어떤 provider와도 제휴하거나 공식 운영되지 않습니다. 이 문서에 등장하는 provider 이름, URL, 모델 ID, Claude Code 설정 가이드는 호환 목적의 참고 자료일 뿐입니다.

---

## 목적

- **Base URL 하나로 여러 provider 쓰기.** `ANTHROPIC_BASE_URL`을 Prism으로 맞춰두면 Claude Code의 티어별 모델 오버라이드(`ANTHROPIC_DEFAULT_HAIKU_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL`)를 각각 다른 provider로 보낼 수 있습니다.
- **모델명 충돌 정리.** `deepseek-v3`처럼 똑같은 이름을 여러 provider가 제공할 때, `fw/deepseek-v3`와 `groq/deepseek-v3`로 확실하게 구분해서 라우팅합니다.
- **긴 모델 ID 단축.** `accounts/fireworks/models/kimi-k2p5` 같은 긴 식별자를 `main` 한 단어로 대체 할 수 있습니다.
- **포맷 자동 변환.** 들어오는 Anthropic Messages 요청을 OpenAI `chat/completions`로 바꿔 upstream에 보내고, 응답과 SSE 이벤트는 다시 Anthropic 프로토콜로 돌려줍니다. 클라이언트는 Anthropic 프로토콜만 바라보면 됩니다.

---

## 빠른 시작

```bash
cargo run
```

기본적으로 8088 포트에서 실행.

- Builder UI: <http://127.0.0.1:8088/>
- Health:     <http://127.0.0.1:8088/healthz>
- Proxy:      <http://127.0.0.1:8088/v1/messages>

Builder UI에서 라우트 설정을 한 후 TOML 파일을 받아, 아래와 같이 실행 할 수 있습니다.
```bash
PRISM_CONFIG=./prism.config.toml cargo run
```

---

## 라우팅 규칙

들어온 요청의 `model` 값을 아래 세 규칙에 위에서부터 순서대로 대입해 매칭합니다.

| # | 규칙 | 조건 | upstream에 보내는 값 |
| --- | --- | --- | --- |
| 1 | **Exact-default** | `model`이 어떤 라우트의 `prefix`와 완전히 일치, 그리고 그 라우트에 `model = "..."` 필드가 있음 | 라우트의 `model` 값 |
| 2 | **Namespace** | `model`이 `prefix/무엇` 형태 | `prefix/`를 떼어낸 나머지 |
| 3 | **Raw longest-prefix** | `model`이 어떤 `prefix`로 시작 (여러 개 걸리면 가장 긴 prefix 우선) | 원본 문자열 그대로 |

어디에도 해당하지 않으면 `400 invalid_request_error`를 돌려줍니다. 응답 본문에 현재 등록된 prefix와 default model 목록이 함께 오므로 디버깅할 때 참고하세요.

### 예시

세 라우트가 아래와 같이 등록돼 있다고 가정합니다.

- `prefix = "main"`, `model = "accounts/fireworks/models/kimi-k2p5"`
- `prefix = "fw"`
- `prefix = "glm"`

| 들어온 `model`명 | 매칭 규칙 | upstream에 실제로 가는 값 |
| --- | --- | --- |
| `main` | Exact-default | `accounts/fireworks/models/kimi-k2p5` |
| `fw/deepseek-v3` | Namespace | `deepseek-v3` |
| `fw/accounts/fireworks/models/llama-v3p1-8b` | Namespace | `accounts/fireworks/models/llama-v3p1-8b` (맨 앞 `fw/`만 제거) |
| `glm-4.5` | Raw longest-prefix | `glm-4.5` (변경 없음) |
| `unknown-xyz` | — | 400 에러 |

### 서버가 뜰 때 검증되는 것들

- `prefix`가 비어 있거나 누락돼 있으면 에러
- 같은 `prefix`가 두 번 이상이면 에러
- `api_key`도 `key_env`도 없으면 에러

같은 매칭 로직을 Builder UI의 Resolver Simulator에서 원하는 모델명으로 직접 돌려볼 수 있습니다.

---

## 설정 파일 (TOML)

라우트마다 `prefix`(필수), `base`(필수), 그리고 인증 수단(`api_key` 아니면 `key_env`) 셋이 꼭 있어야 합니다.
`provider`는 비워두거나 `auto`로 두면 base URL의 호스트 이름을 보고 어댑터를 알아서 고릅니다.
선택 사항인 `model` 필드는 exact-default 매칭에서 upstream에 대신 보낼 모델명을 의미합니다.

```toml
port = 8088

# 1) 긴 모델 ID에 짧은 별명 붙이기 — 요청 `main` → Fireworks / accounts/fireworks/models/kimi-k2p5
[[routes]]
prefix   = "main"
provider = "fireworks"
base     = "https://api.fireworks.ai/inference/v1/"
key_env  = "FIREWORKS_API_KEY"
model    = "accounts/fireworks/models/kimi-k2p5"

# 2) prefix 매칭만, 이름 변환 없음 — glm-4.5, glm-4-flash 등을 그대로 Z.AI로 전송
[[routes]]
prefix   = "glm"
provider = "zai"
base     = "https://api.z.ai/api/coding/paas/v4/"
key_env  = "ZAI_API_KEY"

# 3) 같은 모델을 여러 provider가 제공할 때 구분 — `fw/deepseek-v3`는 Fireworks, `groq/deepseek-v3`는 Groq
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

환경변수 한 줄로 라우트를 전부 정의하고 싶다면 아래와 같이 작성.

```bash
PRISM_ROUTES="prefix=main,provider=fireworks,base=https://api.fireworks.ai/inference/v1,key_env=FIREWORKS_API_KEY,model=accounts/fireworks/models/kimi-k2p5;prefix=glm,provider=zai,base=https://api.z.ai/api/coding/paas/v4,key_env=ZAI_API_KEY" \
PRISM_PORT=8088 \
cargo run
```

## Provider 어댑터

기본적으로 Prism은 들어온 요청을 OpenAI `chat/completions` 포맷으로 변환해서 보냅니다. 이 경로에서 provider마다 필요한 파라미터 수정이 있어 어댑터 계층으로 분리했습니다. 대부분의 OpenAI 호환 provider는 `openai` 하나로 커버되고, 특수 동작이 필요할 때만 별도 어댑터가 있습니다.

| `provider`      | 담당하는 upstream 그룹                                                     |
| --------------- | -------------------------------------------------------------------------- |
| `auto` (기본값) | base URL 호스트를 보고 zai / fireworks / openai 중 자동 선택               |
| `openai`        | OpenAI 호환 대부분 — Groq, DeepSeek, Together, OpenRouter, xAI, Mistral 등 |
| `fireworks`     | Fireworks 전용 — `max_tokens` / `max_completion_tokens` 충돌 정리           |
| `zai`           | Z.AI 전용 — `tool_choice = auto` 강제, tool streaming 활성화, 토큰 필드 보정 |

### `anthropic_format` — Anthropic Messages passthrough

`provider`와는 별개로, 라우트에 `anthropic_format = true` 플래그를 붙이면 OpenAI 변환 계층을 **완전히 건너뜁니다**. 구체적으로는:

- 들어온 Anthropic Messages 본문을 그대로 릴레이합니다 (위의 라우팅 규칙에 따라 `model` 필드 한 개만 교체).
- 인증은 `x-api-key: <key>`, `Authorization: Bearer <key>`, `anthropic-version: 2023-06-01` 헤더를 함께 보냅니다.
- upstream 경로는 `base`에서 자동 결정됩니다. `base`가 `/v1/`로 끝나면 `{base}/messages`, 아니면 `{base}/v1/messages`를 사용합니다.
- 응답 JSON과 SSE 이벤트 모두 바이트 단위로 그대로 흘려보냅니다. Prism이 파싱하지 않습니다.
- provider별 본문 보정(`fireworks` / `zai`의 quirk 처리)은 적용되지 않습니다. 이 경로는 어댑터를 거치지 않기 때문입니다.

Claude Code를 Prism을 통해 `api.anthropic.com`에 직접 연결하거나, Anthropic Messages API를 네이티브로 지원하는 다른 게이트웨이에 붙일 때 사용하세요. 이 모드에서는 `provider`가 사실상 라벨 용도라 `auto`로 둬도 됩니다.

TOML 예시:

```toml
[[routes]]
prefix = "claude"
base = "https://api.anthropic.com/v1/"
key_env = "ANTHROPIC_API_KEY"
anthropic_format = true
```

`claude/claude-sonnet-4-5` 같은 요청은 Anthropic Messages API로 upstream 모델 `claude-sonnet-4-5`가 실려 전송되고, 도중에 어떤 변환도 거치지 않습니다.

## Claude Code와 연동하기

Claude Code는 `ANTHROPIC_BASE_URL`을 통한 게이트웨이 방식을 지원합니다. Builder UI의 **Outputs** 섹션에서 바로 쓸 수 있는 `settings.json`을 복사하거나 받아서 아래 위치에 두면 됩니다.

- 프로젝트 단위:                `.claude/settings.local.json`
- 유저 전역 (macOS / Linux / WSL): `~/.claude/settings.json`
- 유저 전역 (Windows):          `%USERPROFILE%\.claude\settings.json`

예시 `settings.json`:

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

각 자리의 용도:

| 자리                                | 용도                                                                       |
| ----------------------------------- | -------------------------------------------------------------------------- |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL`     | Claude Code가 "haiku 티어"로 보내는 백그라운드 작업용 모델                 |
| `ANTHROPIC_DEFAULT_SONNET_MODEL`    | "sonnet 티어" — 메인 코딩 모델, 가장 자주 쓰임                              |
| `ANTHROPIC_DEFAULT_OPUS_MODEL`      | "opus 티어" — 고난도 작업일 때 승격되는 모델                                |
| `ANTHROPIC_SMALL_FAST_MODEL`        | 툴 선택·분류용 초경량 호출. 보통 Haiku 값과 동일                            |
| 최상위 `"model"`                    | Claude Code 시작 시 `/model`에 선택돼 있는 기본 모델. 보통 Sonnet 값과 동일 |

Builder UI는 Haiku 값을 `ANTHROPIC_SMALL_FAST_MODEL`로, Sonnet 값을 최상위 `"model"`로 자동으로 똑같이 채워줍니다. 서로 다르게 쓰고 싶으면 다운로드한 JSON을 직접 고치면 됩니다.

각 자리에 넣을 값은 앞의 3단 라우팅 규칙을 그대로 따릅니다.

| 적는 값            | 매칭되는 라우트                                         | upstream에 실제로 가는 모델              |
| ------------------ | ------------------------------------------------------- | ---------------------------------------- |
| `main`             | `prefix = "main", model = "accounts/.../kimi-k2p5"`     | `accounts/.../kimi-k2p5` (exact-default) |
| `fw/deepseek-v3`   | `prefix = "fw"`                                         | `deepseek-v3` (namespace)                |
| `glm-4.5`          | `prefix = "glm"`                                        | `glm-4.5` 그대로 (raw prefix)            |

참고: `ANTHROPIC_API_KEY`는 placeholder입니다. Prism이 들어오는 요청의 인증을 검사하지 않아서, 아무 값이나 비어 있지 않기만 하면 됩니다.

---

## Builder UI

<http://127.0.0.1:8088/>을 브라우저로 열면 아래 기능을 사용할 수 있습니다.

- 라우트 추가·편집·삭제 — 포트, base, API key(inline ↔ 환경변수 토글), default model
- **Resolver Simulator** — 모델 문자열을 타이핑하면 어떤 규칙에 걸렸는지, 어느 라우트로 가는지, upstream에는 어떤 이름이 실리는지 바로 확인
- TOML 미리보기·복사·다운로드, 브라우저 `localStorage`에 자동 저장
- 입력한 base와 API key로 upstream `chat/completions` 요청 테스트
- `message_start` / `content_block_*` / `message_delta` / `message_stop` 순서가 제대로 나오는지 보는 Anthropic SSE 스트리밍 플레이그라운드
- Claude Code용 `settings.json` 미리보기·복사·다운로드
- 영어 기본, 한국어 토글
- provider 공식 문서 링크 카드 (비제휴 참고용)

Builder는 `static/index.html` 단일 파일이고, Prism 서버가 `/`로 서빙합니다. 서버 없이 파일을 브라우저로 직접 열어도 라우트 편집이나 TOML 생성은 잘 동작합니다. 다만 Resolver Simulator와 upstream 테스트는 Prism 서버가 켜져있어야 사용할 수 있습니다.

---

## Builder 프리셋 (`static/presets.js`)

각 라우트 카드 오른쪽 위에 있는 **Preset** 드롭다운은 `static/presets.js` 파일 에서 관리됩니다. 이 파일만 고치면 드롭다운 내용이 변경됩니다.

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
  // 원하는 만큼 추가
];
```

- 프리셋을 적용하면 라우트의 **`base`와 `provider`, 그리고 `key_env`(단, 현재 비어 있을 때만)** 세 가지가 채워집니다. `prefix`, default model, 실제 API key 값은 절대 건드리지 않아요.
- 기본으로 31개를 동봉해뒀고, 다음 그룹으로 분류돼 있습니다:
  - **Anthropic 네이티브** (passthrough): Anthropic, Fireworks (Anthropic native), OpenRouter (Anthropic native), LMRouter (Anthropic native), OfoxAI (Anthropic native), Anannas (Anthropic native), LLMGateway (Anthropic native), NagaAI (Anthropic native), Shannon AI (Anthropic native).
  - **주요 코딩 / Claude Code 타깃**: Z.AI (coding), Fireworks, OpenRouter.
  - **OpenAI 계열**: OpenAI, Google Gemini (OAI-compat), xAI (Grok), Mistral.
  - **고속 추론 클라우드**: Groq, Cerebras, SambaNova, Together AI, Hyperbolic, Nebius.
  - **중국계 provider**: DeepSeek, Moonshot (Kimi), Qwen (DashScope intl), MiniMax.
  - **검색·특수 목적**: Perplexity, Cohere.
  - **로컬 / 셀프호스팅**: Ollama, LM Studio, vLLM.
- Prism 서버는 이 파일을 컴파일 시점에 바이너리에 embed해 `/presets.js` 경로로 서빙합니다. 서버를 통해 쓸 때는 파일을 고친 뒤 `cargo run`으로 다시 빌드하세요.
- `static/index.html`을 브라우저에 `file://`로 직접 열 때는 같은 폴더의 `presets.js`를 `<script>` 태그가 바로 읽어갑니다. 서버가 필요 없어요.

provider의 URL이나 모델 라인업은 시간이 지나면 바뀔 수 있으니, 실제로 운영에 쓰기 전에는 각 provider 공식 문서를 한 번 더 확인하는 걸 권장합니다.

---

## API 엔드포인트

| 메서드 | 경로                    | 설명                                                           |
| ------ | ----------------------- | -------------------------------------------------------------- |
| GET    | `/`                     | Builder UI                                                     |
| GET    | `/healthz`              | `{"status":"ok"}`                                              |
| GET    | `/presets.js`           | Builder용 프리셋 (컴파일 시점에 embed)                         |
| POST   | `/v1/messages`          | Anthropic Messages — 라우팅 + 포맷 변환 + SSE 중계             |
| POST   | `/v1/responses`         | OpenAI Responses → chat/completions 변환                      |
| POST   | `/api/test-upstream`    | Builder의 ping 테스트 — upstream `chat/completions` 1회 호출   |
| POST   | `/api/test-stream`      | Builder의 SSE 스트리밍 플레이그라운드                          |
| POST   | `/api/resolve-preview`  | Resolver Simulator의 dry-run                                   |

---

## 포맷 변환 관련 사항과 한계

- Anthropic의 `thinking`, `image`, `document` 같은 블록은 OpenAI 호환 페이로드에 자연스럽게 담을 수 없는 경우가 있어서, 가능한 범위에서만 변환하고 나머지는 텍스트 노트로 낮춰서 내보냅니다.
- `tool_use`와 `tool_result`는 양방향 변환을 지원합니다. 일부 provider는 tool streaming 포맷이 살짝 다르지만 adapter 계층에서 흡수하고 있어요.
- `/v1/responses`는 스트리밍이 아직 구현돼 있지 않아서 호출하면 `501 unsupported_feature`를 반환합니다. 비스트리밍은 동작합니다.
- Prism 자체는 들어오는 요청의 인증을 검사하지 않습니다. **로컬 환경에서만** 쓰세요.

---

## 보안과 운영 시 주의

- Builder가 `localStorage`에 저장하는 값과 다운로드해 주는 TOML 파일에는 **API key가 평문 그대로** 들어갑니다. 개인 개발 환경에서만 다루세요.
- TOML 설정 파일을 git에 올리지 마세요. 운영에 쓰는 키는 `key_env`로 빼내서 환경변수로 공급하는 걸 권장합니다.
- Prism은 인증을 검사하지 않으니, 기본 바인딩인 `0.0.0.0:8088`을 외부망에 그대로 노출하면 안 됩니다. 바깥에서 접근해야 한다면 앞단에 리버스 프록시나 방화벽을 두세요.

---

## 개발

```bash
cargo check                                  # 타입·lint 체크
cargo run                                    # 8088 포트로 기동
PRISM_CONFIG=./prism.config.toml cargo run   # 설정 파일 지정
RUST_LOG=prism=debug cargo run               # 라우팅 로그 자세히 보기
```

소스 구조는 다음과 같습니다.

```
src/
  main.rs       — axum 라우터 설정과 부팅
  config.rs     — TOML 파싱, 환경변수 라우트, 검증
  router.rs     — 3단 리졸버 (exact-default / namespace / raw-prefix)
  proxy.rs      — Anthropic ↔ OpenAI 포맷 변환, SSE 중계
  provider.rs   — provider adapter (zai / fireworks / openai / auto)
  ui.rs         — Builder API 핸들러 (테스트, 시뮬레이터, presets.js)
  types.rs      — 도메인 타입
static/
  index.html    — Builder UI (단일 파일, 영/한 i18n 내장)
  presets.js    — 라우트 프리셋 목록 (직접 편집하는 파일)
```

---

## 라이선스

[Apache License 2.0](./LICENSE) 아래에서 공개됩니다. 저작권은 기여자들이 공동 보유합니다. 배포 결과물에 `NOTICE` 파일이 포함돼 있다면 함께 유지해 주세요.
