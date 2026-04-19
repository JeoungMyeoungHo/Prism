//! Anthropic Messages SSE translator.

use super::{extract_stream_text, sse_event};
use crate::types::{OpenAiChatCompletionChunk, OpenAiToolCallDelta, OpenAiUsage};
use axum::body::Bytes;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Default)]
struct ToolCallStreamState {
    id: String,
    name: String,
    started: bool,
}

/// Incrementally converts an OpenAI Chat Completion SSE stream into the
/// Anthropic Messages SSE shape.
///
/// Event sequence emitted to the client:
///
/// ```text
/// message_start
///   (content_block_start → content_block_delta* → content_block_stop)*
/// message_delta
/// message_stop
/// ```
///
/// One text block and zero-or-more tool blocks may interleave. Block indices
/// are assigned by this translator and are **not** taken from upstream.
///
/// Invariants:
/// - `message_start` is emitted exactly once, on the first chunk carrying a
///   response id or model (guarded by `emitted_message_start`).
/// - `text_block_*` track the single text block; it is opened lazily on the
///   first text delta and closed before `message_delta`.
/// - `tool_blocks` is indexed by **our** assigned index (see
///   [`AnthropicStreamTranslator::resolve_tool_index`]); upstream tool indices
///   are best-effort and may be absent, so we soft-match by id/name or fall
///   back to a synthetic index.
/// - `finished` guards against duplicate `message_stop` if `finish()` runs
///   after a terminal chunk.
///
/// Silent `unwrap_or_default()` / `unwrap_or(...)` inside [`push`] handle
/// legitimately-absent OpenAI delta fields (e.g. missing `content` on a
/// usage-only chunk) and are part of the normal path — do not promote to
/// WARN without evidence of real data loss.
pub(crate) struct AnthropicStreamTranslator {
    requested_model: String,
    response_id: Option<String>,
    model: Option<String>,
    emitted_message_start: bool,
    finished: bool,
    finish_reason: Option<String>,
    usage: OpenAiUsage,
    text_block_index: Option<usize>,
    text_block_open: bool,
    tool_blocks: BTreeMap<usize, ToolCallStreamState>,
    next_tool_index: usize,
    last_tool_index: Option<usize>,
}

impl AnthropicStreamTranslator {
    pub(crate) fn new(requested_model: String) -> Self {
        Self {
            requested_model,
            response_id: None,
            model: None,
            emitted_message_start: false,
            finished: false,
            finish_reason: None,
            usage: OpenAiUsage::default(),
            text_block_index: None,
            text_block_open: false,
            tool_blocks: BTreeMap::new(),
            next_tool_index: 0,
            last_tool_index: None,
        }
    }

    /// Ingest one OpenAI chunk and emit zero or more Anthropic SSE events.
    /// First chunk with identity fields triggers `message_start`; subsequent
    /// chunks may open/append text or tool blocks. Terminal chunks (finish
    /// reason set) are coalesced with the final usage into `message_delta` +
    /// `message_stop` by [`finish`].
    pub(crate) fn push(&mut self, chunk: OpenAiChatCompletionChunk) -> Vec<Bytes> {
        let mut events = Vec::new();

        if let Some(id) = chunk.id {
            self.response_id = Some(id);
        }
        if let Some(model) = chunk.model {
            self.model = Some(model);
        }
        if let Some(usage) = chunk.usage {
            self.usage = usage;
        }

        if !self.emitted_message_start {
            events.push(sse_event(
                "message_start",
                &json!({
                    "type": "message_start",
                    "message": {
                        "id": self.response_id.clone().unwrap_or_else(fallback_message_id),
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model.clone().unwrap_or_else(|| self.requested_model.clone()),
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": {
                            "input_tokens": self.usage.prompt_tokens,
                            "output_tokens": 0,
                        }
                    }
                }),
            ));
            self.emitted_message_start = true;
        }

        for choice in chunk.choices {
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }

            if let Some(content) = choice.delta.content {
                if let Some(text) = extract_stream_text(&content) {
                    if !text.is_empty() {
                        let index = self.ensure_text_block(&mut events);
                        events.push(sse_event(
                            "content_block_delta",
                            &json!({
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "text_delta",
                                    "text": text,
                                }
                            }),
                        ));
                    }
                }
            }

            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    let tool_index = self.resolve_tool_index(&tool_call);
                    let content_index =
                        compute_tool_content_index(self.text_block_index, tool_index);
                    let mut start_event = None;
                    let mut delta_event = None;

                    {
                        let state = self.tool_blocks.entry(tool_index).or_default();

                        if let Some(id) = tool_call.id {
                            state.id = id;
                        }

                        if let Some(function) = tool_call.function {
                            if let Some(name) = function.name {
                                state.name = name;
                            }

                            if !state.started {
                                if state.id.is_empty() {
                                    state.id = format!("toolu_{}", tool_index);
                                }
                                if state.name.is_empty() {
                                    state.name = format!("tool_{}", tool_index);
                                }

                                start_event = Some(json!({
                                    "type": "content_block_start",
                                    "index": content_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": state.id,
                                        "name": state.name,
                                        "input": {}
                                    }
                                }));
                                state.started = true;
                            }

                            if let Some(arguments) = function.arguments {
                                if !arguments.is_empty() {
                                    delta_event = Some(json!({
                                        "type": "content_block_delta",
                                        "index": content_index,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": arguments,
                                        }
                                    }));
                                }
                            }
                        }
                    }

                    if let Some(payload) = start_event {
                        events.push(sse_event("content_block_start", &payload));
                    }

                    if let Some(payload) = delta_event {
                        events.push(sse_event("content_block_delta", &payload));
                    }
                }
            }
        }

        events
    }

    pub(crate) fn finish(&mut self) -> Vec<Bytes> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;

        let mut events = Vec::new();

        if !self.emitted_message_start {
            events.push(sse_event(
                "message_start",
                &json!({
                    "type": "message_start",
                    "message": {
                        "id": self.response_id.clone().unwrap_or_else(fallback_message_id),
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model.clone().unwrap_or_else(|| self.requested_model.clone()),
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": {
                            "input_tokens": self.usage.prompt_tokens,
                            "output_tokens": 0,
                        }
                    }
                }),
            ));
            self.emitted_message_start = true;
        }

        if self.text_block_open {
            events.push(sse_event(
                "content_block_stop",
                &json!({
                    "type": "content_block_stop",
                    "index": self.text_block_index.unwrap_or(0),
                }),
            ));
            self.text_block_open = false;
        }

        let mut tool_indices = self.tool_blocks.keys().cloned().collect::<Vec<_>>();
        tool_indices.sort_unstable();
        for index in tool_indices {
            if self
                .tool_blocks
                .get(&index)
                .map(|state| state.started)
                .unwrap_or(false)
            {
                events.push(sse_event(
                    "content_block_stop",
                    &json!({
                        "type": "content_block_stop",
                        "index": self.tool_content_index(index),
                    }),
                ));
            }
        }

        events.push(sse_event(
            "message_delta",
            &json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": super::super::translate_response::map_finish_reason(self.finish_reason.as_deref()),
                    "stop_sequence": Value::Null,
                },
                "usage": {
                    "output_tokens": self.usage.completion_tokens,
                }
            }),
        ));
        events.push(sse_event(
            "message_stop",
            &json!({ "type": "message_stop" }),
        ));

        events
    }

    /// Lazily open the text block on first text delta and return its index.
    /// Index is chosen to not collide with existing tool blocks.
    fn ensure_text_block(&mut self, events: &mut Vec<Bytes>) -> usize {
        let index = *self.text_block_index.get_or_insert_with(|| {
            if self.tool_blocks.is_empty() {
                0
            } else {
                self.tool_blocks.len()
            }
        });

        if !self.text_block_open {
            events.push(sse_event(
                "content_block_start",
                &json!({
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "text",
                        "text": "",
                    }
                }),
            ));
            self.text_block_open = true;
        }

        index
    }

    fn tool_content_index(&self, tool_index: usize) -> usize {
        match self.text_block_index {
            Some(0) => tool_index + 1,
            Some(text_index) if text_index <= tool_index => tool_index,
            _ => tool_index,
        }
    }

    /// Map an upstream tool delta to our local tool block index.
    ///
    /// Resolution order:
    /// 1. Upstream-provided `index` (authoritative when present).
    /// 2. Match by tool call id or function name against already-open blocks.
    /// 3. Reuse the single open tool block when there is exactly one.
    /// 4. Allocate a synthetic index via [`allocate_synthetic_tool_index`].
    ///
    /// This is the only soft-match point in the translator — upstreams that
    /// omit `index` on tool deltas still produce coherent output.
    fn resolve_tool_index(&mut self, tool_call: &OpenAiToolCallDelta) -> usize {
        if let Some(index) = tool_call.index {
            self.last_tool_index = Some(index);
            self.note_explicit_tool_index(index);
            return index;
        }

        if let Some(id) = tool_call.id.as_deref().filter(|id| !id.is_empty()) {
            if let Some(index) = self
                .tool_blocks
                .iter()
                .find_map(|(index, state)| (state.id == id).then_some(*index))
            {
                self.last_tool_index = Some(index);
                return index;
            }
        }

        if let Some(function) = tool_call.function.as_ref() {
            if let Some(name) = function.name.as_deref().filter(|name| !name.is_empty()) {
                let matches = self
                    .tool_blocks
                    .iter()
                    .filter_map(|(index, state)| (state.name == name).then_some(*index))
                    .collect::<Vec<_>>();
                if matches.len() == 1 {
                    self.last_tool_index = Some(matches[0]);
                    return matches[0];
                }
            }

            if function
                .arguments
                .as_deref()
                .is_some_and(|arguments| !arguments.is_empty())
            {
                if let Some(index) = self
                    .last_tool_index
                    .filter(|index| self.tool_blocks.contains_key(index))
                {
                    return index;
                }
            }
        }

        // Only one tool block open → route deltas there. `if let Some`
        // expresses the invariant locally without an unreachable expect.
        if self.tool_blocks.len() == 1 {
            if let Some((&index, _)) = self.tool_blocks.iter().next() {
                self.last_tool_index = Some(index);
                return index;
            }
        }

        let index = self.allocate_synthetic_tool_index();
        self.last_tool_index = Some(index);
        index
    }

    fn note_explicit_tool_index(&mut self, index: usize) {
        self.next_tool_index = self.next_tool_index.max(index.saturating_add(1));
    }

    /// Return the lowest unused index, skipping any already occupied by
    /// existing tool blocks. Used when upstream gives us no identifying info.
    fn allocate_synthetic_tool_index(&mut self) -> usize {
        while self.tool_blocks.contains_key(&self.next_tool_index) {
            self.next_tool_index += 1;
        }
        let index = self.next_tool_index;
        self.next_tool_index += 1;
        index
    }
}

fn compute_tool_content_index(text_block_index: Option<usize>, tool_index: usize) -> usize {
    match text_block_index {
        Some(0) => tool_index + 1,
        Some(text_index) if text_index <= tool_index => tool_index,
        _ => tool_index,
    }
}

fn fallback_message_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("msg_prism_{millis}")
}
