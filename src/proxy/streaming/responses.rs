//! OpenAI Responses SSE translator.

use super::{extract_stream_text, sse_named_event};
use crate::types::{OpenAiChatCompletionChunk, OpenAiToolCallDelta, OpenAiUsage};
use axum::body::Bytes;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Default)]
struct ResponsesToolStreamState {
    item_id: String,
    call_id: String,
    name: String,
    arguments: String,
    output_index: usize,
    added: bool,
}

struct ResponsesClosedOutputItem {
    output_index: usize,
    item: Value,
}

/// Translates an upstream `chat/completions` SSE stream into OpenAI
/// `/v1/responses` SSE events. Mirrors what the non-streaming translator
/// produces, but maintains incremental state for
/// `response.output_text.delta`, `response.function_call_arguments.delta`,
/// and `response.reasoning_summary_text.delta`.
///
/// Event sequence emitted to the client:
///
/// ```text
/// response.created → response.in_progress
///   (response.output_item.added
///      → response.(reasoning_summary_text|output_text|function_call_arguments).delta*
///      → response.output_item.done)*
/// response.completed  (or response.failed via `fail`)
/// ```
///
/// Each upstream chunk may touch three distinct output item kinds —
/// reasoning, message (text), and tool call. They are tracked independently
/// and closed in the order they were opened, with `sequence` incremented on
/// every emitted event.
///
/// Invariants:
/// - `response.created` + `response.in_progress` fire exactly once, guarded
///   by `emitted_created`.
/// - `reasoning_*` / `message_*` ids and output indices are assigned lazily
///   on the first relevant delta. Opening is idempotent.
/// - `tools` is indexed by our local index (see [`resolve_tool_index`]);
///   upstream may omit indices, in which case id/name soft-match is tried.
/// - `finished` prevents `finish()`/`fail()` from double-emitting terminal
///   events.
/// - `next_output_index` is bumped whenever a new output item is opened, so
///   all emitted `output_index` values are monotonically increasing.
pub(crate) struct ResponsesStreamTranslator {
    requested_model: String,
    created_at: u64,
    response_id: Option<String>,
    upstream_model: Option<String>,
    usage: OpenAiUsage,
    finish_reason: Option<String>,

    sequence: u64,
    finished: bool,
    emitted_created: bool,

    next_output_index: usize,

    reasoning_item_id: Option<String>,
    reasoning_output_index: Option<usize>,
    reasoning_part_added: bool,
    reasoning_text: String,

    message_item_id: Option<String>,
    message_output_index: Option<usize>,
    message_part_added: bool,
    message_text: String,

    tools: BTreeMap<usize, ResponsesToolStreamState>,
    next_tool_index: usize,
    last_tool_index: Option<usize>,
    output_items: Vec<ResponsesClosedOutputItem>,
}

impl ResponsesStreamTranslator {
    pub(crate) fn new(requested_model: String) -> Self {
        Self {
            requested_model,
            created_at: super::super::current_timestamp_seconds(),
            response_id: None,
            upstream_model: None,
            usage: OpenAiUsage::default(),
            finish_reason: None,
            sequence: 0,
            finished: false,
            emitted_created: false,
            next_output_index: 0,
            reasoning_item_id: None,
            reasoning_output_index: None,
            reasoning_part_added: false,
            reasoning_text: String::new(),
            message_item_id: None,
            message_output_index: None,
            message_part_added: false,
            message_text: String::new(),
            tools: BTreeMap::new(),
            next_tool_index: 0,
            last_tool_index: None,
            output_items: Vec::new(),
        }
    }

    fn next_seq(&mut self) -> u64 {
        let n = self.sequence;
        self.sequence += 1;
        n
    }

    fn model_label(&self) -> String {
        self.upstream_model
            .clone()
            .unwrap_or_else(|| self.requested_model.clone())
    }

    fn response_id_or_fallback(&self) -> String {
        self.response_id
            .clone()
            .unwrap_or_else(fallback_response_id)
    }

    fn terminal_event_name(&self) -> &'static str {
        match self.finish_reason.as_deref() {
            Some("length") => "response.incomplete",
            Some("content_filter") => "response.failed",
            _ => "response.completed",
        }
    }

    fn terminal_status(&self) -> &'static str {
        match self.finish_reason.as_deref() {
            Some("length") => "incomplete",
            Some("content_filter") => "failed",
            _ => "completed",
        }
    }

    /// Ingest one OpenAI chunk and emit zero or more Responses SSE events.
    /// Opens per-kind output items lazily and bumps the shared `sequence`
    /// counter on every event. Finish reason, usage, and stream termination
    /// are collected and flushed by [`finish`] / [`fail`].
    pub(crate) fn push(&mut self, chunk: OpenAiChatCompletionChunk) -> Vec<Bytes> {
        let mut events = Vec::new();

        if let Some(id) = chunk.id {
            self.response_id = Some(id);
        }
        if let Some(model) = chunk.model {
            self.upstream_model = Some(model);
        }
        if let Some(usage) = chunk.usage {
            self.usage = usage;
        }

        if !self.emitted_created {
            self.emit_created(&mut events);
        }

        for choice in chunk.choices {
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }

            if let Some(reasoning) = choice.delta.reasoning_content {
                if let Some(text) = extract_stream_text(&reasoning) {
                    if !text.is_empty() {
                        self.emit_reasoning_delta(&text, &mut events);
                    }
                }
            }

            if let Some(content) = choice.delta.content {
                if let Some(text) = extract_stream_text(&content) {
                    if !text.is_empty() {
                        self.emit_text_delta(&text, &mut events);
                    }
                }
            }

            if let Some(tool_calls) = choice.delta.tool_calls {
                for tool_call in tool_calls {
                    self.absorb_tool_call_delta(tool_call, &mut events);
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

        if !self.emitted_created {
            self.emit_created(&mut events);
        }

        if self.reasoning_part_added {
            self.close_reasoning(&mut events);
        }

        if self.message_item_id.is_some() {
            self.close_message(&mut events);
        }

        self.close_all_tools(&mut events);

        let response = self.final_response_object();
        let seq = self.next_seq();
        events.push(sse_named_event(
            self.terminal_event_name(),
            &json!({
                "type": self.terminal_event_name(),
                "sequence_number": seq,
                "response": response,
            }),
        ));

        events
    }

    /// Abort the stream with `response.failed`. No-op if already finished.
    /// Closes any open items in their current state before emitting the
    /// terminal event.
    pub(crate) fn fail(&mut self, code: &str, message: &str) -> Vec<Bytes> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;

        let mut events = Vec::new();

        if !self.emitted_created {
            self.emit_created(&mut events);
        }

        if self.reasoning_item_id.is_some() {
            self.close_reasoning(&mut events);
        }
        if self.message_item_id.is_some() {
            self.close_message(&mut events);
        }
        self.close_all_tools(&mut events);

        let response = self.failed_response_object(code, message);
        let seq = self.next_seq();
        events.push(sse_named_event(
            "response.failed",
            &json!({
                "type": "response.failed",
                "sequence_number": seq,
                "response": response,
            }),
        ));

        events
    }

    fn emit_created(&mut self, events: &mut Vec<Bytes>) {
        self.emitted_created = true;
        let envelope = self.in_progress_response_object();
        let seq_created = self.next_seq();
        events.push(sse_named_event(
            "response.created",
            &json!({
                "type": "response.created",
                "sequence_number": seq_created,
                "response": envelope,
            }),
        ));
        let seq_in_progress = self.next_seq();
        events.push(sse_named_event(
            "response.in_progress",
            &json!({
                "type": "response.in_progress",
                "sequence_number": seq_in_progress,
                "response": self.in_progress_response_object(),
            }),
        ));
    }

    /// Append `text` to the reasoning output item, opening it on first use.
    fn emit_reasoning_delta(&mut self, text: &str, events: &mut Vec<Bytes>) {
        if self.message_item_id.is_some() {
            self.close_message(events);
        }
        self.close_all_tools(events);

        if self.reasoning_item_id.is_none() {
            let response_id = self.response_id_or_fallback();
            let output_index = self.next_output_index;
            self.next_output_index += 1;
            let item_id = format!("rs_{response_id}_{output_index}");
            self.reasoning_item_id = Some(item_id.clone());
            self.reasoning_output_index = Some(output_index);
            let seq = self.next_seq();
            events.push(sse_named_event(
                "response.output_item.added",
                &json!({
                    "type": "response.output_item.added",
                    "sequence_number": seq,
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "reasoning",
                        "status": "in_progress",
                        "summary": []
                    }
                }),
            ));
        }

        if !self.reasoning_part_added {
            let item_id = self.reasoning_item_id.clone().unwrap();
            let output_index = self.reasoning_output_index.unwrap();
            let seq = self.next_seq();
            events.push(sse_named_event(
                "response.reasoning_summary_part.added",
                &json!({
                    "type": "response.reasoning_summary_part.added",
                    "sequence_number": seq,
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": { "type": "summary_text", "text": "" }
                }),
            ));
            self.reasoning_part_added = true;
        }

        self.reasoning_text.push_str(text);
        let item_id = self.reasoning_item_id.clone().unwrap();
        let output_index = self.reasoning_output_index.unwrap();
        let seq = self.next_seq();
        events.push(sse_named_event(
            "response.reasoning_summary_text.delta",
            &json!({
                "type": "response.reasoning_summary_text.delta",
                "sequence_number": seq,
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": 0,
                "delta": text,
            }),
        ));
    }

    /// Append `text` to the message output item, opening it on first use.
    fn emit_text_delta(&mut self, text: &str, events: &mut Vec<Bytes>) {
        // Seal non-message items before opening the message item so
        // `output_index` reflects the order each item started streaming.
        if self.reasoning_item_id.is_some() {
            self.close_reasoning(events);
        }
        self.close_all_tools(events);

        if self.message_item_id.is_none() {
            let response_id = self.response_id_or_fallback();
            let output_index = self.next_output_index;
            self.next_output_index += 1;
            let item_id = format!("msg_{response_id}_{output_index}");
            self.message_item_id = Some(item_id.clone());
            self.message_output_index = Some(output_index);
            let seq = self.next_seq();
            events.push(sse_named_event(
                "response.output_item.added",
                &json!({
                    "type": "response.output_item.added",
                    "sequence_number": seq,
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": []
                    }
                }),
            ));
        }

        if !self.message_part_added {
            let item_id = self.message_item_id.clone().unwrap();
            let output_index = self.message_output_index.unwrap();
            let seq = self.next_seq();
            events.push(sse_named_event(
                "response.content_part.added",
                &json!({
                    "type": "response.content_part.added",
                    "sequence_number": seq,
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": { "type": "output_text", "text": "", "annotations": [] }
                }),
            ));
            self.message_part_added = true;
        }

        self.message_text.push_str(text);
        let item_id = self.message_item_id.clone().unwrap();
        let output_index = self.message_output_index.unwrap();
        let seq = self.next_seq();
        events.push(sse_named_event(
            "response.output_text.delta",
            &json!({
                "type": "response.output_text.delta",
                "sequence_number": seq,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "delta": text,
            }),
        ));
    }

    /// Fold a tool-call delta into the matching tool state, opening the
    /// output item on first sight and routing fragments (id/name/arguments)
    /// into their accumulators. Must run after any pending text/reasoning
    /// items are closed so output indices stay monotonic.
    fn absorb_tool_call_delta(&mut self, tool_call: OpenAiToolCallDelta, events: &mut Vec<Bytes>) {
        // Text and reasoning items should be sealed before function_call items
        // so that `output_index` reflects the order the first piece of each
        // item arrived. We seal opportunistically once a tool delta is seen.
        if self.reasoning_item_id.is_some() {
            self.close_reasoning(events);
        }
        if self.message_item_id.is_some() {
            self.close_message(events);
        }

        let tool_index = self.resolve_tool_index(&tool_call);
        let needs_open = !self.tools.contains_key(&tool_index);
        if needs_open {
            let output_index = self.next_output_index;
            self.next_output_index += 1;
            self.tools.insert(
                tool_index,
                ResponsesToolStreamState {
                    output_index,
                    ..Default::default()
                },
            );
        }

        let state = self.tools.get_mut(&tool_index).unwrap();
        if let Some(id) = tool_call.id {
            if !id.is_empty() {
                state.call_id = id;
            }
        }
        if let Some(function) = tool_call.function {
            if let Some(name) = function.name {
                if !name.is_empty() {
                    state.name = name;
                }
            }
            if let Some(args) = function.arguments {
                if !state.added
                    && (!state.name.is_empty() || !state.call_id.is_empty() || !args.is_empty())
                {
                    Self::open_tool_item(state, events, &mut self.sequence);
                }
                if !args.is_empty() {
                    state.arguments.push_str(&args);
                    let seq = {
                        let n = self.sequence;
                        self.sequence += 1;
                        n
                    };
                    events.push(sse_named_event(
                        "response.function_call_arguments.delta",
                        &json!({
                            "type": "response.function_call_arguments.delta",
                            "sequence_number": seq,
                            "item_id": state.item_id,
                            "output_index": state.output_index,
                            "delta": args,
                        }),
                    ));
                }
            }
        }

        // Ensure the item is opened even when only id/name arrived without
        // argument text yet.
        if let Some(state) = self.tools.get_mut(&tool_index) {
            if !state.added && (!state.name.is_empty() || !state.call_id.is_empty()) {
                Self::open_tool_item(state, events, &mut self.sequence);
            }
        }
    }

    fn close_all_tools(&mut self, events: &mut Vec<Bytes>) {
        let tool_indices: Vec<usize> = self.tools.keys().copied().collect();
        for idx in tool_indices {
            self.close_tool(idx, events);
        }
    }

    /// Emit `response.output_item.added` for a tool call. Idempotent via
    /// `state.added`.
    fn open_tool_item(
        state: &mut ResponsesToolStreamState,
        events: &mut Vec<Bytes>,
        sequence: &mut u64,
    ) {
        if state.added {
            return;
        }
        if state.call_id.is_empty() {
            state.call_id = format!("call_prism_{}", state.output_index);
        }
        if state.item_id.is_empty() {
            state.item_id = format!("fc_{}", state.call_id);
        }
        if state.name.is_empty() {
            state.name = format!("tool_{}", state.output_index);
        }

        let seq = {
            let n = *sequence;
            *sequence += 1;
            n
        };
        events.push(sse_named_event(
            "response.output_item.added",
            &json!({
                "type": "response.output_item.added",
                "sequence_number": seq,
                "output_index": state.output_index,
                "item": {
                    "id": state.item_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": state.call_id,
                    "name": state.name,
                    "arguments": ""
                }
            }),
        ));
        state.added = true;
    }

    /// Emit `part.done` + `output_item.done` for the reasoning item and
    /// reset the reasoning accumulators so a future delta would reopen.
    fn close_reasoning(&mut self, events: &mut Vec<Bytes>) {
        let Some(item_id) = self.reasoning_item_id.clone() else {
            return;
        };
        let output_index = self.reasoning_output_index.unwrap_or(0);
        let text = std::mem::take(&mut self.reasoning_text);

        if self.reasoning_part_added {
            let seq1 = self.next_seq();
            events.push(sse_named_event(
                "response.reasoning_summary_text.done",
                &json!({
                    "type": "response.reasoning_summary_text.done",
                    "sequence_number": seq1,
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "text": text,
                }),
            ));
            let seq2 = self.next_seq();
            events.push(sse_named_event(
                "response.reasoning_summary_part.done",
                &json!({
                    "type": "response.reasoning_summary_part.done",
                    "sequence_number": seq2,
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": {
                        "type": "summary_text",
                        "text": text,
                    }
                }),
            ));
            self.reasoning_part_added = false;
        }

        let item = json!({
            "id": item_id.clone(),
            "type": "reasoning",
            "status": "completed",
            "summary": [
                { "type": "summary_text", "text": text.clone() }
            ]
        });
        let seq3 = self.next_seq();
        events.push(sse_named_event(
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "sequence_number": seq3,
                "output_index": output_index,
                "item": item.clone(),
            }),
        ));
        self.output_items
            .push(ResponsesClosedOutputItem { output_index, item });

        self.reasoning_item_id = None;
        self.reasoning_output_index = None;
    }

    /// Emit `part.done` + `output_item.done` for the message (text) item
    /// and reset its accumulators.
    fn close_message(&mut self, events: &mut Vec<Bytes>) {
        let Some(item_id) = self.message_item_id.clone() else {
            return;
        };
        let output_index = self.message_output_index.unwrap_or(0);
        let text = std::mem::take(&mut self.message_text);

        let seq1 = self.next_seq();
        events.push(sse_named_event(
            "response.output_text.done",
            &json!({
                "type": "response.output_text.done",
                "sequence_number": seq1,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            }),
        ));
        let seq2 = self.next_seq();
        events.push(sse_named_event(
            "response.content_part.done",
            &json!({
                "type": "response.content_part.done",
                "sequence_number": seq2,
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": text,
                    "annotations": []
                }
            }),
        ));
        let item = json!({
            "id": item_id.clone(),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": text.clone(),
                    "annotations": []
                }
            ]
        });
        let seq3 = self.next_seq();
        events.push(sse_named_event(
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "sequence_number": seq3,
                "output_index": output_index,
                "item": item.clone(),
            }),
        ));
        self.output_items
            .push(ResponsesClosedOutputItem { output_index, item });

        self.message_part_added = false;
        self.message_item_id = None;
        self.message_output_index = None;
    }

    /// Emit `output_item.done` for the tool call and remove its state.
    fn close_tool(&mut self, tool_index: usize, events: &mut Vec<Bytes>) {
        let Some(mut state) = self.tools.remove(&tool_index) else {
            return;
        };
        if self.last_tool_index == Some(tool_index) {
            self.last_tool_index = None;
        }
        if !state.added {
            Self::open_tool_item(&mut state, events, &mut self.sequence);
        }

        let seq1 = self.next_seq();
        events.push(sse_named_event(
            "response.function_call_arguments.done",
            &json!({
                "type": "response.function_call_arguments.done",
                "sequence_number": seq1,
                "item_id": state.item_id,
                "output_index": state.output_index,
                "arguments": state.arguments,
            }),
        ));
        let item = json!({
            "id": state.item_id.clone(),
            "type": "function_call",
            "status": "completed",
            "call_id": state.call_id.clone(),
            "name": state.name.clone(),
            "arguments": state.arguments.clone(),
        });
        let seq2 = self.next_seq();
        events.push(sse_named_event(
            "response.output_item.done",
            &json!({
                "type": "response.output_item.done",
                "sequence_number": seq2,
                "output_index": state.output_index,
                "item": item.clone(),
            }),
        ));
        self.output_items.push(ResponsesClosedOutputItem {
            output_index: state.output_index,
            item,
        });
    }

    /// Map an upstream tool delta to our local tool index. Mirrors
    /// [`AnthropicStreamTranslator::resolve_tool_index`]: explicit upstream
    /// index wins, then id/name soft-match, then single-open reuse, then
    /// synthetic allocation.
    fn resolve_tool_index(&mut self, tool_call: &OpenAiToolCallDelta) -> usize {
        if let Some(index) = tool_call.index {
            self.last_tool_index = Some(index);
            self.note_explicit_tool_index(index);
            return index;
        }

        if let Some(id) = tool_call.id.as_deref().filter(|id| !id.is_empty()) {
            if let Some(index) = self
                .tools
                .iter()
                .find_map(|(index, state)| (state.call_id == id).then_some(*index))
            {
                self.last_tool_index = Some(index);
                return index;
            }
        }

        if let Some(function) = tool_call.function.as_ref() {
            if let Some(name) = function.name.as_deref().filter(|name| !name.is_empty()) {
                let matches = self
                    .tools
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
                    .filter(|idx| self.tools.contains_key(idx))
                {
                    return index;
                }
            }
        }

        // Only one tool open → route deltas there. `if let Some`
        // expresses the invariant locally without an unreachable expect.
        if self.tools.len() == 1 {
            if let Some((&index, _)) = self.tools.iter().next() {
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

    /// Lowest unused index, skipping any already occupied tool slot.
    fn allocate_synthetic_tool_index(&mut self) -> usize {
        while self.tools.contains_key(&self.next_tool_index) {
            self.next_tool_index += 1;
        }
        let index = self.next_tool_index;
        self.next_tool_index += 1;
        index
    }

    fn finalized_output(&self) -> Vec<Value> {
        let mut output_items: Vec<&ResponsesClosedOutputItem> = self.output_items.iter().collect();
        output_items.sort_by_key(|item| item.output_index);
        output_items.iter().map(|item| item.item.clone()).collect()
    }

    fn in_progress_response_object(&self) -> Value {
        json!({
            "id": self.response_id_or_fallback(),
            "object": "response",
            "created_at": self.created_at,
            "status": "in_progress",
            "model": self.model_label(),
            "output": [],
        })
    }

    fn final_response_object(&self) -> Value {
        let output = self.finalized_output();
        let tool_count = output
            .iter()
            .filter(|item| item.get("type") == Some(&json!("function_call")))
            .count();

        let mut response = json!({
            "id": self.response_id_or_fallback(),
            "object": "response",
            "created_at": self.created_at,
            "completed_at": super::super::current_timestamp_seconds(),
            "status": self.terminal_status(),
            "model": self.model_label(),
            "output": output,
            "parallel_tool_calls": tool_count > 1,
            "usage": {
                "input_tokens": self.usage.prompt_tokens,
                "output_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.prompt_tokens + self.usage.completion_tokens,
            }
        });

        // `response` just built via `json!({ ... })` — always Value::Object.
        let response_object = response
            .as_object_mut()
            .expect("json! literal always yields Value::Object");
        match self.finish_reason.as_deref() {
            Some("length") => {
                response_object.insert(
                    "incomplete_details".into(),
                    json!({ "reason": "max_output_tokens" }),
                );
            }
            Some("content_filter") => {
                response_object.insert(
                    "error".into(),
                    json!({
                        "type": "content_filter",
                        "code": "content_filter",
                        "message": "upstream content filter blocked the response",
                    }),
                );
            }
            _ => {}
        }

        response
    }

    fn failed_response_object(&self, code: &str, message: &str) -> Value {
        let output = self.finalized_output();
        let tool_count = output
            .iter()
            .filter(|item| item.get("type") == Some(&json!("function_call")))
            .count();

        json!({
            "id": self.response_id_or_fallback(),
            "object": "response",
            "created_at": self.created_at,
            "completed_at": super::super::current_timestamp_seconds(),
            "status": "failed",
            "model": self.model_label(),
            "output": output,
            "parallel_tool_calls": tool_count > 1,
            "usage": {
                "input_tokens": self.usage.prompt_tokens,
                "output_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.prompt_tokens + self.usage.completion_tokens,
            },
            "error": {
                "type": code,
                "code": code,
                "message": message,
            }
        })
    }
}

fn fallback_response_id() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("resp_prism_{millis}")
}
