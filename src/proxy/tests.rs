use super::{
    anthropic_passthrough_url, anthropic_request_to_openai, openai_response_to_anthropic,
    AnthropicStreamTranslator, ResponsesStreamTranslator,
};
use crate::{
    provider::ProviderKind,
    types::{
        AnthropicBlock, AnthropicContent, AnthropicMessage, AnthropicMessagesRequest, Backend,
        OpenAiChatCompletionChunk, OpenAiChatCompletionResponse,
    },
};
use axum::body::Bytes;
use serde_json::{json, Map, Value};
use url::Url;

fn decode_sse(events: &[Bytes]) -> Vec<(String, Value)> {
    events
        .iter()
        .map(|bytes| {
            let raw = std::str::from_utf8(bytes).expect("event bytes must be utf-8");
            let mut event_name = String::new();
            let mut data_line = String::new();
            for line in raw.lines() {
                if let Some(rest) = line.strip_prefix("event: ") {
                    event_name = rest.to_string();
                } else if let Some(rest) = line.strip_prefix("data: ") {
                    data_line = rest.to_string();
                }
            }
            let payload: Value = serde_json::from_str(&data_line)
                .unwrap_or_else(|_| panic!("bad SSE data payload: {data_line}"));
            (event_name, payload)
        })
        .collect()
}

fn event_types(decoded: &[(String, Value)]) -> Vec<String> {
    decoded.iter().map(|(name, _)| name.clone()).collect()
}

fn run_translator(chunks: Vec<Value>) -> Vec<(String, Value)> {
    let mut translator = ResponsesStreamTranslator::new("gpt-5".into());
    let mut raw: Vec<Bytes> = Vec::new();
    for chunk_json in chunks {
        let chunk: OpenAiChatCompletionChunk =
            serde_json::from_value(chunk_json).expect("chunk should parse");
        raw.extend(translator.push(chunk));
    }
    raw.extend(translator.finish());
    decode_sse(&raw)
}

fn run_anthropic_stream_translator(chunks: Vec<Value>) -> Vec<(String, Value)> {
    let mut translator = AnthropicStreamTranslator::new("claude-3-7-sonnet".into());
    let mut raw: Vec<Bytes> = Vec::new();
    for chunk_json in chunks {
        let chunk: OpenAiChatCompletionChunk =
            serde_json::from_value(chunk_json).expect("chunk should parse");
        raw.extend(translator.push(chunk));
    }
    raw.extend(translator.finish());
    decode_sse(&raw)
}

#[test]
fn anthropic_passthrough_url_keeps_messages_under_existing_v1_base() {
    let url = anthropic_passthrough_url(&Url::parse("https://api.anthropic.com/v1/").unwrap());
    assert_eq!(url.as_str(), "https://api.anthropic.com/v1/messages");
}

#[test]
fn anthropic_passthrough_url_adds_v1_for_sdk_style_bases() {
    let url =
        anthropic_passthrough_url(&Url::parse("https://api.fireworks.ai/inference/").unwrap());
    assert_eq!(
        url.as_str(),
        "https://api.fireworks.ai/inference/v1/messages"
    );
}

#[test]
fn request_translation_relaxes_image_and_thinking_blocks() {
    let mut thinking_fields = Map::new();
    thinking_fields.insert(
        "thinking".into(),
        Value::String("private scratchpad".into()),
    );

    let mut image_fields = Map::new();
    image_fields.insert(
        "source".into(),
        json!({
            "type": "url",
            "url": "https://example.com/cat.png"
        }),
    );

    let backend = Backend {
        prefix: "glm".into(),
        provider: ProviderKind::Zai,
        base: Url::parse("https://api.z.ai/api/paas/v4/").unwrap(),
        api_key: "test".into(),
        credential_label: "inline".into(),
        default_model: None,
        anthropic_format: false,
    };

    let request = AnthropicMessagesRequest {
        model: "glm-4.5".into(),
        messages: vec![
            AnthropicMessage {
                role: "assistant".into(),
                content: AnthropicContent::Blocks(vec![
                    AnthropicBlock {
                        kind: "thinking".into(),
                        fields: thinking_fields,
                    },
                    AnthropicBlock::text("Visible answer".into()),
                ]),
            },
            AnthropicMessage {
                role: "user".into(),
                content: AnthropicContent::Blocks(vec![
                    AnthropicBlock {
                        kind: "image".into(),
                        fields: image_fields,
                    },
                    AnthropicBlock::text("Describe the image.".into()),
                ]),
            },
        ],
        system: None,
        max_tokens: Some(64),
        stream: Some(true),
        temperature: None,
        top_p: None,
        stop_sequences: None,
        tools: None,
        tool_choice: None,
    };

    let prepared = anthropic_request_to_openai(request, &backend).unwrap();
    let messages = prepared
        .body
        .get("messages")
        .and_then(Value::as_array)
        .unwrap();

    assert_eq!(messages.len(), 2);
    assert_eq!(
        messages[0].get("reasoning_content"),
        Some(&json!("private scratchpad"))
    );
    assert!(messages[1]
        .get("content")
        .and_then(Value::as_array)
        .is_some());
}

#[test]
fn document_block_text_source_is_expanded_inline() {
    let mut doc_fields = Map::new();
    doc_fields.insert("title".into(), Value::String("notes.txt".into()));
    doc_fields.insert(
        "source".into(),
        json!({
            "type": "text",
            "media_type": "text/plain",
            "data": "Prism is a router."
        }),
    );

    let request = AnthropicMessagesRequest {
        model: "glm-4.5".into(),
        messages: vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(vec![
                AnthropicBlock {
                    kind: "document".into(),
                    fields: doc_fields,
                },
                AnthropicBlock::text("Summarize this.".into()),
            ]),
        }],
        system: None,
        max_tokens: Some(64),
        stream: None,
        temperature: None,
        top_p: None,
        stop_sequences: None,
        tools: None,
        tool_choice: None,
    };

    let backend = Backend {
        prefix: "glm".into(),
        provider: ProviderKind::Zai,
        base: Url::parse("https://api.z.ai/api/paas/v4/").unwrap(),
        api_key: "test".into(),
        credential_label: "inline".into(),
        default_model: None,
        anthropic_format: false,
    };

    let prepared = anthropic_request_to_openai(request, &backend).unwrap();
    let content = prepared
        .body
        .pointer("/messages/0/content")
        .expect("user message has content");

    let rendered = match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => panic!("unexpected content shape: {content:?}"),
    };

    assert!(rendered.contains("notes.txt"));
    assert!(rendered.contains("Prism is a router."));
    assert!(rendered.contains("Summarize this."));
    assert!(!rendered.contains("omitted by Prism"));
}

#[test]
fn document_block_content_source_preserves_nested_image() {
    let mut doc_fields = Map::new();
    doc_fields.insert(
        "source".into(),
        json!({
            "type": "content",
            "content": [
                {"type": "text", "text": "caption before"},
                {"type": "image", "source": {
                    "type": "url",
                    "url": "https://example.com/diagram.png"
                }},
                {"type": "text", "text": "caption after"}
            ]
        }),
    );

    let request = AnthropicMessagesRequest {
        model: "glm-4.5".into(),
        messages: vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(vec![AnthropicBlock {
                kind: "document".into(),
                fields: doc_fields,
            }]),
        }],
        system: None,
        max_tokens: Some(32),
        stream: None,
        temperature: None,
        top_p: None,
        stop_sequences: None,
        tools: None,
        tool_choice: None,
    };

    let backend = Backend {
        prefix: "glm".into(),
        provider: ProviderKind::Zai,
        base: Url::parse("https://api.z.ai/api/paas/v4/").unwrap(),
        api_key: "test".into(),
        credential_label: "inline".into(),
        default_model: None,
        anthropic_format: false,
    };

    let prepared = anthropic_request_to_openai(request, &backend).unwrap();
    let parts = prepared
        .body
        .pointer("/messages/0/content")
        .and_then(Value::as_array)
        .expect("user message content should be an array of parts");

    let image_urls: Vec<&str> = parts
        .iter()
        .filter(|p| p.get("type").and_then(Value::as_str) == Some("image_url"))
        .filter_map(|p| {
            p.get("image_url")
                .and_then(|v| v.get("url"))
                .and_then(Value::as_str)
        })
        .collect();
    assert_eq!(image_urls, vec!["https://example.com/diagram.png"]);

    let texts: Vec<&str> = parts
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect();
    assert!(texts.iter().any(|t| t.contains("caption before")));
    assert!(texts.iter().any(|t| t.contains("caption after")));
}

#[test]
fn document_block_binary_source_still_falls_back_to_note() {
    let mut doc_fields = Map::new();
    doc_fields.insert(
        "source".into(),
        json!({
            "type": "base64",
            "media_type": "application/pdf",
            "data": "JVBERi0xLjQK"
        }),
    );

    let request = AnthropicMessagesRequest {
        model: "glm-4.5".into(),
        messages: vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(vec![AnthropicBlock {
                kind: "document".into(),
                fields: doc_fields,
            }]),
        }],
        system: None,
        max_tokens: Some(16),
        stream: None,
        temperature: None,
        top_p: None,
        stop_sequences: None,
        tools: None,
        tool_choice: None,
    };

    let backend = Backend {
        prefix: "glm".into(),
        provider: ProviderKind::Zai,
        base: Url::parse("https://api.z.ai/api/paas/v4/").unwrap(),
        api_key: "test".into(),
        credential_label: "inline".into(),
        default_model: None,
        anthropic_format: false,
    };

    let prepared = anthropic_request_to_openai(request, &backend).unwrap();
    let content = prepared
        .body
        .pointer("/messages/0/content")
        .expect("user message content is present");

    let rendered = match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    };
    assert!(rendered.contains("omitted by Prism"));
}

#[test]
fn response_translation_accepts_object_tool_arguments() {
    let response: OpenAiChatCompletionResponse = serde_json::from_value(json!({
        "id": "chatcmpl_demo",
        "model": "glm-4.5",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": {
                                    "city": "Seoul"
                                }
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4
        }
    }))
    .unwrap();

    let anthropic = openai_response_to_anthropic(response, "glm-4.5").unwrap();
    let content = anthropic.get("content").and_then(Value::as_array).unwrap();

    assert_eq!(content[0].get("type"), Some(&json!("tool_use")));
    assert_eq!(
        content[0]
            .get("input")
            .and_then(Value::as_object)
            .and_then(|input| input.get("city"))
            .and_then(Value::as_str),
        Some("Seoul")
    );
}

#[test]
fn responses_stream_text_only_emits_full_lifecycle() {
    let decoded = run_translator(vec![
        json!({
            "id": "resp_123",
            "model": "gpt-5",
            "choices": [
                { "index": 0, "delta": { "content": "Hello" } }
            ]
        }),
        json!({
            "id": "resp_123",
            "choices": [
                { "index": 0, "delta": { "content": " world" } }
            ]
        }),
        json!({
            "id": "resp_123",
            "choices": [
                { "index": 0, "delta": {}, "finish_reason": "stop" }
            ],
            "usage": { "prompt_tokens": 12, "completion_tokens": 4 }
        }),
    ]);

    let types = event_types(&decoded);
    assert_eq!(
        types,
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );

    // sequence_number must increment monotonically from 0.
    for (i, (_, payload)) in decoded.iter().enumerate() {
        assert_eq!(
            payload.get("sequence_number").and_then(Value::as_u64),
            Some(i as u64),
            "sequence mismatch at event {i}"
        );
    }

    let done = decoded
        .iter()
        .find(|(name, _)| name == "response.output_text.done")
        .unwrap();
    assert_eq!(
        done.1.get("text").and_then(Value::as_str),
        Some("Hello world")
    );

    let completed = decoded
        .iter()
        .find(|(name, _)| name == "response.completed")
        .unwrap();
    let response = completed.1.get("response").unwrap();
    assert_eq!(
        response.get("status").and_then(Value::as_str),
        Some("completed")
    );
    assert_eq!(
        response
            .pointer("/usage/input_tokens")
            .and_then(Value::as_u64),
        Some(12)
    );
    assert_eq!(
        response
            .pointer("/usage/output_tokens")
            .and_then(Value::as_u64),
        Some(4)
    );
    let output = response.get("output").and_then(Value::as_array).unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(
        output[0].get("type").and_then(Value::as_str),
        Some("message")
    );
}

#[test]
fn responses_stream_tool_call_emits_arguments_events() {
    let decoded = run_translator(vec![
        json!({
            "id": "resp_t1",
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "type": "function",
                                "function": { "name": "lookup_weather", "arguments": "" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "resp_t1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": { "arguments": "{\"city\":" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "resp_t1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": { "arguments": "\"Seoul\"}" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "resp_t1",
            "choices": [
                { "index": 0, "delta": {}, "finish_reason": "tool_calls" }
            ],
            "usage": { "prompt_tokens": 20, "completion_tokens": 7 }
        }),
    ]);

    let types = event_types(&decoded);
    // First 2 are created/in_progress, then the tool item lifecycle.
    assert_eq!(&types[0..2], &["response.created", "response.in_progress"]);
    assert!(types.contains(&"response.output_item.added".into()));
    assert!(types
        .iter()
        .any(|t| t == "response.function_call_arguments.delta"));
    assert!(types
        .iter()
        .any(|t| t == "response.function_call_arguments.done"));
    assert!(types.contains(&"response.output_item.done".into()));
    assert_eq!(types.last().unwrap(), "response.completed");

    let arg_done = decoded
        .iter()
        .find(|(name, _)| name == "response.function_call_arguments.done")
        .unwrap();
    assert_eq!(
        arg_done.1.get("arguments").and_then(Value::as_str),
        Some("{\"city\":\"Seoul\"}")
    );

    let added_tool_item = decoded
        .iter()
        .find(|(name, v)| {
            name == "response.output_item.added"
                && v.pointer("/item/type").and_then(Value::as_str) == Some("function_call")
        })
        .expect("output_item.added for function_call");
    assert_eq!(
        added_tool_item
            .1
            .pointer("/item/call_id")
            .and_then(Value::as_str),
        Some("call_abc")
    );

    let completed = decoded
        .iter()
        .find(|(name, _)| name == "response.completed")
        .unwrap();
    let output = completed
        .1
        .pointer("/response/output")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(
        output[0].get("type").and_then(Value::as_str),
        Some("function_call")
    );
    assert_eq!(
        output[0].get("arguments").and_then(Value::as_str),
        Some("{\"city\":\"Seoul\"}")
    );
}

#[test]
fn responses_stream_reasoning_precedes_message_events() {
    let decoded = run_translator(vec![
        json!({
            "id": "resp_r1",
            "model": "gpt-5",
            "choices": [
                { "index": 0, "delta": { "reasoning_content": "Thinking..." } }
            ]
        }),
        json!({
            "id": "resp_r1",
            "choices": [
                { "index": 0, "delta": { "reasoning_content": " done." } }
            ]
        }),
        json!({
            "id": "resp_r1",
            "choices": [
                { "index": 0, "delta": { "content": "Answer." } }
            ]
        }),
        json!({
            "id": "resp_r1",
            "choices": [
                { "index": 0, "delta": {}, "finish_reason": "stop" }
            ],
            "usage": { "prompt_tokens": 5, "completion_tokens": 3 }
        }),
    ]);

    let types = event_types(&decoded);
    let first_reasoning_delta = types
        .iter()
        .position(|t| t == "response.reasoning_summary_text.delta")
        .expect("reasoning delta should appear");
    let first_text_delta = types
        .iter()
        .position(|t| t == "response.output_text.delta")
        .expect("text delta should appear");
    assert!(
        first_reasoning_delta < first_text_delta,
        "reasoning should stream before message text"
    );

    let reasoning_done = decoded
        .iter()
        .find(|(name, _)| name == "response.reasoning_summary_text.done")
        .unwrap();
    assert_eq!(
        reasoning_done.1.get("text").and_then(Value::as_str),
        Some("Thinking... done.")
    );

    // Reasoning item must be sealed (output_item.done) before message item
    // opens.
    let reasoning_item_done = types
        .iter()
        .enumerate()
        .find(|(_, t)| t.as_str() == "response.output_item.done")
        .map(|(i, _)| i)
        .unwrap();
    let message_item_added = types
        .iter()
        .enumerate()
        .find(|(_, t)| t.as_str() == "response.output_item.added")
        .map(|(i, _)| i)
        .unwrap();
    // The first output_item.added is reasoning (before the first done).
    // The second output_item.added (for message) must come AFTER the
    // reasoning done.
    let second_added = types
        .iter()
        .enumerate()
        .skip(message_item_added + 1)
        .find(|(_, t)| t.as_str() == "response.output_item.added")
        .map(|(i, _)| i)
        .expect("message output_item.added");
    assert!(second_added > reasoning_item_done);

    let completed = decoded.last().unwrap();
    assert_eq!(completed.0, "response.completed");
    let output = completed
        .1
        .pointer("/response/output")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(output.len(), 2);
    assert_eq!(
        output[0].get("type").and_then(Value::as_str),
        Some("reasoning")
    );
    assert_eq!(
        output[1].get("type").and_then(Value::as_str),
        Some("message")
    );
}

/// Drives `chunks` through the translator and returns the wire bytes
/// (exactly what would be written to the SSE response) plus decoded events.
fn dump_translation(label: &str, chunks: Vec<Value>) -> (String, Vec<(String, Value)>) {
    let mut translator = ResponsesStreamTranslator::new("gpt-5".into());
    let mut raw: Vec<Bytes> = Vec::new();
    for chunk_json in &chunks {
        let chunk: OpenAiChatCompletionChunk =
            serde_json::from_value(chunk_json.clone()).expect("fixture chunk must parse");
        raw.extend(translator.push(chunk));
    }
    raw.extend(translator.finish());

    let wire: String = raw
        .iter()
        .map(|b| std::str::from_utf8(b).expect("utf8").to_string())
        .collect();

    println!("\n===== [{label}] INPUT chat/completions chunks =====");
    for (i, chunk) in chunks.iter().enumerate() {
        println!("# chunk {i}:");
        println!("{}", serde_json::to_string_pretty(chunk).unwrap());
    }
    println!("\n===== [{label}] OUTPUT Responses SSE stream =====");
    print!("{wire}");
    println!("===== [{label}] END =====\n");

    (wire, decode_sse(&raw))
}

/// A reasoning model that thinks, then answers in plain text.
/// Exercises: created → in_progress → reasoning item lifecycle →
/// message item lifecycle → completed, with sealed reasoning BEFORE the
/// message opens.
#[test]
fn responses_stream_fixture_reasoning_then_text() {
    let chunks = vec![
        json!({
            "id": "resp_demo_1",
            "object": "chat.completion.chunk",
            "model": "gpt-5",
            "choices": [
                { "index": 0, "delta": { "role": "assistant" } }
            ]
        }),
        json!({
            "id": "resp_demo_1",
            "choices": [
                { "index": 0, "delta": { "reasoning_content": "The user asks about weather. " } }
            ]
        }),
        json!({
            "id": "resp_demo_1",
            "choices": [
                { "index": 0, "delta": { "reasoning_content": "I'll answer from general knowledge." } }
            ]
        }),
        json!({
            "id": "resp_demo_1",
            "choices": [
                { "index": 0, "delta": { "content": "Seoul is typically " } }
            ]
        }),
        json!({
            "id": "resp_demo_1",
            "choices": [
                { "index": 0, "delta": { "content": "mild in spring." } }
            ]
        }),
        json!({
            "id": "resp_demo_1",
            "choices": [
                { "index": 0, "delta": {}, "finish_reason": "stop" }
            ],
            "usage": { "prompt_tokens": 24, "completion_tokens": 11 }
        }),
    ];

    let (wire, decoded) = dump_translation("reasoning_then_text", chunks);
    let types = event_types(&decoded);

    // Opening triplet.
    assert_eq!(&types[0..2], &["response.created", "response.in_progress"]);
    // Final event is completion.
    assert_eq!(types.last().unwrap(), "response.completed");

    // Reasoning item must fully close (its output_item.done) BEFORE the
    // message's output_item.added appears.
    let first_item_done = types
        .iter()
        .position(|t| t == "response.output_item.done")
        .expect("reasoning item.done");
    let added_positions: Vec<usize> = types
        .iter()
        .enumerate()
        .filter(|(_, t)| t.as_str() == "response.output_item.added")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        added_positions.len(),
        2,
        "expected reasoning + message adds"
    );
    let message_added = added_positions[1];
    assert!(
        message_added > first_item_done,
        "message add at {message_added} must follow reasoning done at {first_item_done}"
    );

    // Sequence numbers strictly increase from 0.
    for (i, (_, v)) in decoded.iter().enumerate() {
        assert_eq!(
            v.get("sequence_number").and_then(Value::as_u64),
            Some(i as u64),
            "sequence mismatch at event {i}"
        );
    }

    // Wire output is a valid SSE stream with blank-line separators.
    assert!(wire.contains("event: response.created\n"));
    assert!(wire.contains("event: response.output_text.delta\n"));
    assert!(wire.ends_with("\n\n"));

    let completed = decoded.last().unwrap();
    let response = completed.1.get("response").unwrap();
    assert_eq!(
        response.pointer("/status").and_then(Value::as_str),
        Some("completed")
    );
    let output = response.get("output").and_then(Value::as_array).unwrap();
    assert_eq!(output.len(), 2);
    assert_eq!(
        output[0].pointer("/summary/0/text").and_then(Value::as_str),
        Some("The user asks about weather. I'll answer from general knowledge.")
    );
    assert_eq!(
        output[1].pointer("/content/0/text").and_then(Value::as_str),
        Some("Seoul is typically mild in spring.")
    );
    assert_eq!(
        response
            .pointer("/usage/total_tokens")
            .and_then(Value::as_u64),
        Some(35)
    );
}

/// Reasoning model that thinks, then emits a tool call with streamed
/// arguments. Exercises reasoning sealing before function_call, plus
/// incremental `response.function_call_arguments.delta` concatenation.
#[test]
fn responses_stream_fixture_reasoning_then_tool_call() {
    let chunks = vec![
        json!({
            "id": "resp_demo_2",
            "object": "chat.completion.chunk",
            "model": "gpt-5",
            "choices": [
                { "index": 0, "delta": { "role": "assistant" } }
            ]
        }),
        json!({
            "id": "resp_demo_2",
            "choices": [
                { "index": 0, "delta": { "reasoning_content": "User wants live weather. " } }
            ]
        }),
        json!({
            "id": "resp_demo_2",
            "choices": [
                { "index": 0, "delta": { "reasoning_content": "Call the tool." } }
            ]
        }),
        json!({
            "id": "resp_demo_2",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_wx_1",
                                "type": "function",
                                "function": { "name": "lookup_weather", "arguments": "" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "resp_demo_2",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": { "arguments": "{\"city\":" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "resp_demo_2",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": { "arguments": "\"Seoul\"" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "resp_demo_2",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": { "arguments": ",\"units\":\"c\"}" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "resp_demo_2",
            "choices": [
                { "index": 0, "delta": {}, "finish_reason": "tool_calls" }
            ],
            "usage": { "prompt_tokens": 31, "completion_tokens": 14 }
        }),
    ];

    let (wire, decoded) = dump_translation("reasoning_then_tool_call", chunks);
    let types = event_types(&decoded);

    // Must have three argument delta events (one per upstream args chunk).
    let arg_delta_count = types
        .iter()
        .filter(|t| t.as_str() == "response.function_call_arguments.delta")
        .count();
    assert_eq!(arg_delta_count, 3);

    // Argument done carries the concatenated full string.
    let arg_done = decoded
        .iter()
        .find(|(n, _)| n == "response.function_call_arguments.done")
        .unwrap();
    assert_eq!(
        arg_done.1.get("arguments").and_then(Value::as_str),
        Some("{\"city\":\"Seoul\",\"units\":\"c\"}")
    );

    // Reasoning item must be sealed (done) before function_call item opens.
    let first_item_done = types
        .iter()
        .position(|t| t == "response.output_item.done")
        .unwrap();
    let function_call_added_idx = decoded
        .iter()
        .enumerate()
        .find(|(_, (n, v))| {
            n == "response.output_item.added"
                && v.pointer("/item/type").and_then(Value::as_str) == Some("function_call")
        })
        .map(|(i, _)| i)
        .unwrap();
    assert!(function_call_added_idx > first_item_done);

    // Final envelope has reasoning item + function_call item with the
    // concatenated arguments string.
    let completed = decoded.last().unwrap();
    let output = completed
        .1
        .pointer("/response/output")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(output.len(), 2);
    assert_eq!(
        output[0].get("type").and_then(Value::as_str),
        Some("reasoning")
    );
    assert_eq!(
        output[1].get("type").and_then(Value::as_str),
        Some("function_call")
    );
    assert_eq!(
        output[1].get("call_id").and_then(Value::as_str),
        Some("call_wx_1")
    );
    assert_eq!(
        output[1].get("arguments").and_then(Value::as_str),
        Some("{\"city\":\"Seoul\",\"units\":\"c\"}")
    );

    // Wire output sanity.
    assert!(wire.contains("event: response.function_call_arguments.delta\n"));
    assert!(wire.contains("event: response.function_call_arguments.done\n"));
}

#[test]
fn responses_stream_tool_then_text_seals_tool_before_message() {
    let decoded = run_translator(vec![
        json!({
            "id": "resp_mix_1",
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_lookup",
                                "type": "function",
                                "function": { "name": "lookup_weather", "arguments": "{\"city\":\"Seoul\"}" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "resp_mix_1",
            "choices": [
                { "index": 0, "delta": { "content": "It is sunny." } }
            ]
        }),
        json!({
            "id": "resp_mix_1",
            "choices": [
                { "index": 0, "delta": {}, "finish_reason": "stop" }
            ],
            "usage": { "prompt_tokens": 9, "completion_tokens": 6 }
        }),
    ]);

    let function_done = decoded
        .iter()
        .enumerate()
        .find(|(_, (name, value))| {
            name == "response.output_item.done"
                && value.pointer("/item/type").and_then(Value::as_str) == Some("function_call")
        })
        .map(|(index, _)| index)
        .expect("function call done");
    let message_added = decoded
        .iter()
        .enumerate()
        .find(|(_, (name, value))| {
            name == "response.output_item.added"
                && value.pointer("/item/type").and_then(Value::as_str) == Some("message")
        })
        .map(|(index, _)| index)
        .expect("message item added");
    assert!(message_added > function_done);

    let completed = decoded.last().unwrap();
    assert_eq!(completed.0, "response.completed");
    let output = completed
        .1
        .pointer("/response/output")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(
        output
            .iter()
            .map(|item| item.get("type").and_then(Value::as_str).unwrap())
            .collect::<Vec<_>>(),
        vec!["function_call", "message"]
    );
}

#[test]
fn responses_stream_text_then_reasoning_seals_message_before_reasoning() {
    let decoded = run_translator(vec![
        json!({
            "id": "resp_mix_2",
            "model": "gpt-5",
            "choices": [
                { "index": 0, "delta": { "content": "First answer. " } }
            ]
        }),
        json!({
            "id": "resp_mix_2",
            "choices": [
                { "index": 0, "delta": { "thinking": "Then reflect." } }
            ]
        }),
        json!({
            "id": "resp_mix_2",
            "choices": [
                { "index": 0, "delta": {}, "finish_reason": "stop" }
            ],
            "usage": { "prompt_tokens": 8, "completion_tokens": 5 }
        }),
    ]);

    let message_done = decoded
        .iter()
        .enumerate()
        .find(|(_, (name, value))| {
            name == "response.output_item.done"
                && value.pointer("/item/type").and_then(Value::as_str) == Some("message")
        })
        .map(|(index, _)| index)
        .expect("message item done");
    let reasoning_added = decoded
        .iter()
        .enumerate()
        .find(|(_, (name, value))| {
            name == "response.output_item.added"
                && value.pointer("/item/type").and_then(Value::as_str) == Some("reasoning")
        })
        .map(|(index, _)| index)
        .expect("reasoning item added");
    assert!(reasoning_added > message_done);

    let completed = decoded.last().unwrap();
    let output = completed
        .1
        .pointer("/response/output")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(
        output
            .iter()
            .map(|item| item.get("type").and_then(Value::as_str).unwrap())
            .collect::<Vec<_>>(),
        vec!["message", "reasoning"]
    );
    assert_eq!(
        output[1].pointer("/summary/0/text").and_then(Value::as_str),
        Some("Then reflect.")
    );
}

#[test]
fn responses_stream_reasoning_aliases_are_supported() {
    let decoded = run_translator(vec![
        json!({
            "id": "resp_alias_1",
            "model": "gpt-5",
            "choices": [
                { "index": 0, "delta": { "reasoning_details": [{ "text": "Plan. " }, { "text": "Verify." }] } }
            ]
        }),
        json!({
            "id": "resp_alias_1",
            "choices": [
                { "index": 0, "delta": { "content": "Done." } }
            ]
        }),
        json!({
            "id": "resp_alias_1",
            "choices": [
                { "index": 0, "delta": {}, "finish_reason": "stop" }
            ]
        }),
    ]);

    let reasoning_done = decoded
        .iter()
        .find(|(name, _)| name == "response.reasoning_summary_text.done")
        .unwrap();
    assert_eq!(
        reasoning_done.1.get("text").and_then(Value::as_str),
        Some("Plan. Verify.")
    );
}

#[test]
fn responses_stream_length_finish_emits_incomplete() {
    let decoded = run_translator(vec![
        json!({
            "id": "resp_term_1",
            "model": "gpt-5",
            "choices": [
                { "index": 0, "delta": { "content": "Truncated answer" } }
            ]
        }),
        json!({
            "id": "resp_term_1",
            "choices": [
                { "index": 0, "delta": {}, "finish_reason": "length" }
            ],
            "usage": { "prompt_tokens": 11, "completion_tokens": 4 }
        }),
    ]);

    let terminal = decoded.last().unwrap();
    assert_eq!(terminal.0, "response.incomplete");
    assert_eq!(
        terminal
            .1
            .pointer("/response/status")
            .and_then(Value::as_str),
        Some("incomplete")
    );
    assert_eq!(
        terminal
            .1
            .pointer("/response/incomplete_details/reason")
            .and_then(Value::as_str),
        Some("max_output_tokens")
    );
}

#[test]
fn responses_stream_content_filter_finish_emits_failed() {
    let decoded = run_translator(vec![
        json!({
            "id": "resp_term_2",
            "model": "gpt-5",
            "choices": [
                { "index": 0, "delta": { "content": "Partial" } }
            ]
        }),
        json!({
            "id": "resp_term_2",
            "choices": [
                { "index": 0, "delta": {}, "finish_reason": "content_filter" }
            ]
        }),
    ]);

    let terminal = decoded.last().unwrap();
    assert_eq!(terminal.0, "response.failed");
    assert_eq!(
        terminal
            .1
            .pointer("/response/status")
            .and_then(Value::as_str),
        Some("failed")
    );
    assert_eq!(
        terminal
            .1
            .pointer("/response/error/code")
            .and_then(Value::as_str),
        Some("content_filter")
    );
}

#[test]
fn responses_stream_failure_preserves_partial_output() {
    let mut translator = ResponsesStreamTranslator::new("gpt-5".into());
    let mut raw: Vec<Bytes> = Vec::new();

    let chunk: OpenAiChatCompletionChunk = serde_json::from_value(json!({
        "id": "resp_fail_1",
        "model": "gpt-5",
        "choices": [
            { "index": 0, "delta": { "content": "Partial answer" } }
        ]
    }))
    .unwrap();

    raw.extend(translator.push(chunk));
    raw.extend(translator.fail("upstream_stream_error", "socket closed"));

    let decoded = decode_sse(&raw);
    let terminal = decoded.last().unwrap();
    assert_eq!(terminal.0, "response.failed");
    assert_eq!(
        terminal
            .1
            .pointer("/response/error/code")
            .and_then(Value::as_str),
        Some("upstream_stream_error")
    );
    assert_eq!(
        terminal
            .1
            .pointer("/response/output/0/content/0/text")
            .and_then(Value::as_str),
        Some("Partial answer")
    );
}

#[test]
fn responses_stream_missing_tool_index_prefers_last_open_tool() {
    let decoded = run_translator(vec![
        json!({
            "id": "resp_tool_idx_1",
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": { "name": "first_tool", "arguments": "" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "resp_tool_idx_1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_b",
                                "type": "function",
                                "function": { "name": "second_tool", "arguments": "" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "resp_tool_idx_1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "function": { "arguments": "{\"city\":\"Seoul\"}" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "resp_tool_idx_1",
            "choices": [
                { "index": 0, "delta": {}, "finish_reason": "tool_calls" }
            ]
        }),
    ]);

    let completed = decoded.last().unwrap();
    let output = completed
        .1
        .pointer("/response/output")
        .and_then(Value::as_array)
        .unwrap();

    let args_for = |call_id: &str| {
        output
            .iter()
            .find(|item| item.get("call_id").and_then(Value::as_str) == Some(call_id))
            .and_then(|item| item.get("arguments").and_then(Value::as_str))
    };

    assert_eq!(args_for("call_a"), Some(""));
    assert_eq!(args_for("call_b"), Some("{\"city\":\"Seoul\"}"));
}

#[test]
fn anthropic_stream_missing_tool_index_prefers_last_open_tool() {
    let decoded = run_anthropic_stream_translator(vec![
        json!({
            "id": "chatcmpl_tool_idx_1",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": { "name": "first_tool", "arguments": "" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "chatcmpl_tool_idx_1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_b",
                                "type": "function",
                                "function": { "name": "second_tool", "arguments": "" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "chatcmpl_tool_idx_1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "function": { "arguments": "{\"city\":\"Seoul\"}" }
                            }
                        ]
                    }
                }
            ]
        }),
        json!({
            "id": "chatcmpl_tool_idx_1",
            "choices": [
                { "index": 0, "delta": {}, "finish_reason": "tool_calls" }
            ]
        }),
    ]);

    let city_delta = decoded
        .iter()
        .find(|(name, value)| {
            name == "content_block_delta"
                && value.pointer("/delta/partial_json").and_then(Value::as_str)
                    == Some("{\"city\":\"Seoul\"}")
        })
        .expect("tool arguments delta");
    assert_eq!(city_delta.1.get("index").and_then(Value::as_u64), Some(1));
}

#[test]
fn responses_stream_finish_without_body_still_completes() {
    let mut translator = ResponsesStreamTranslator::new("gpt-5".into());
    let raw = translator.finish();
    let decoded = decode_sse(&raw);
    let types = event_types(&decoded);
    assert_eq!(
        types,
        vec![
            "response.created",
            "response.in_progress",
            "response.completed",
        ]
    );
    let completed = decoded.last().unwrap();
    let output = completed
        .1
        .pointer("/response/output")
        .and_then(Value::as_array)
        .unwrap();
    assert!(output.is_empty());
}
