"""Dual-protocol (OpenAI + Anthropic) LLM client tests — no live gateway."""

from __future__ import annotations

import json

from airecon.proxy.llm import (
    LLMClient,
    _to_anthropic_messages,
    _to_anthropic_tools,
    _to_openai_messages,
)


class _FakeResp:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _client(anthropic: bool) -> LLMClient:
    c = object.__new__(LLMClient)
    c._is_anthropic = anthropic
    c._api_key = "sk-test"
    c._provider = "anthropic" if anthropic else "openai"
    return c


async def _collect(agen):
    return [ev async for ev in agen]


class TestAnthropicMessages:
    def test_system_extracted_and_tool_pairing(self):
        system, msgs = _to_anthropic_messages(
            [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Tokyo"},
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "22C"},
            ]
        )
        assert system == "be helpful"
        assert msgs[0] == {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        assert msgs[1]["content"][0] == {
            "type": "tool_use",
            "id": "call_a",
            "name": "get_weather",
            "input": {"city": "Tokyo"},
        }
        assert msgs[2]["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "call_a",
            "content": "22C",
        }

    def test_fifo_binds_missing_tool_call_id(self):
        _, msgs = _to_anthropic_messages(
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_z", "function": {"name": "t", "arguments": {}}}
                    ],
                },
                {"role": "tool", "content": "ok"},
            ]
        )
        assert msgs[-1]["content"][0]["tool_use_id"] == "call_z"

    def test_orphan_tool_result_dropped(self):
        _, msgs = _to_anthropic_messages(
            [
                {"role": "user", "content": "go"},
                {"role": "tool", "tool_call_id": "ghost", "content": "x"},
            ]
        )
        assert all(
            b.get("type") != "tool_result" for m in msgs for b in m["content"]
        )

    def test_consecutive_same_role_coalesced(self):
        _, msgs = _to_anthropic_messages(
            [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ]
        )
        assert len(msgs) == 1
        assert [b["text"] for b in msgs[0]["content"]] == ["one", "two"]


class TestAnthropicTools:
    def test_function_schema_to_input_schema(self):
        out = _to_anthropic_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "w",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )
        assert out[0]["name"] == "get_weather"
        assert out[0]["input_schema"] == {"type": "object", "properties": {}}


class TestAnthropicStream:
    async def test_parses_real_event_shape(self):
        lines = [
            'event: message_start',
            'data: {"type":"message_start","message":{"usage":{"input_tokens":0}}}',
            'event: content_block_start',
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}',
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call_x","name":"get_weather","input":{}}}',
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\":\\"Tokyo\\"}"}}',
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"input_tokens":9,"output_tokens":4}}',
            "data: [DONE]",
        ]
        evs = await _collect(_client(True)._iter_anthropic_stream(_FakeResp(lines), None))
        kinds = [e["kind"] for e in evs]
        assert "thinking" in kinds and "tool" in kinds and "finish" in kinds
        thinking = next(e for e in evs if e["kind"] == "thinking")
        assert thinking["text"] == "hmm"
        tool_evs = [e for e in evs if e["kind"] == "tool"]
        assert tool_evs[0]["name"] == "get_weather"
        assert "".join(e["args"] for e in tool_evs) == '{"city":"Tokyo"}'
        finish = next(e for e in evs if e["kind"] == "finish")
        assert finish["reason"] == "tool_calls"
        usage = next(e for e in evs if e["kind"] == "usage")
        assert usage["usage"] == {"prompt_tokens": 9, "completion_tokens": 4}


class TestOpenAIStream:
    async def test_parses_reasoning_and_tool_deltas(self):
        lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"think"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"t","arguments":"{}"}}]}}]}',
            "data: [DONE]",
        ]
        evs = await _collect(_client(False)._iter_openai_stream(_FakeResp(lines), None))
        kinds = [e["kind"] for e in evs]
        assert kinds == ["thinking", "finish", "text", "tool"]


class TestAuthHeaders:
    def test_anthropic_uses_x_api_key_and_version(self):
        h = _client(True)._auth_headers()
        assert h["x-api-key"] == "sk-test"
        assert h["anthropic-version"] == "2023-06-01"
        assert h["Authorization"] == "Bearer sk-test"

    def test_openai_uses_bearer_only(self):
        h = _client(False)._auth_headers()
        assert h["Authorization"] == "Bearer sk-test"
        assert "x-api-key" not in h


class TestOpenAIMessagesRegression:
    def test_still_pairs_tool_results(self):
        msgs = _to_openai_messages(
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "t", "arguments": {"a": 1}}}
                    ],
                },
                {"role": "tool", "content": "ok"},
            ]
        )
        assert msgs[-1]["role"] == "tool"
        assert msgs[-1]["tool_call_id"] == "c1"
        assert json.loads(msgs[-2]["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
