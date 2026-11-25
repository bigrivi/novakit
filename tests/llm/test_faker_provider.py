"""Offline coverage for novakit.llm.faker.

Behavioral contract only: determinism, reply families, content-driven tools,
streaming and abort semantics. No network, no nova imports.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

import pytest

from novakit.llm.faker import FakerLLMProvider
from novakit.llm.provider import (
    Done,
    Error,
    Message,
    ReasoningDelta,
    TextDelta,
    ToolCall,
)


def _tools(*names: str) -> list[dict[str, Any]]:
    """Build minimal tool schemas with empty object parameters."""
    return [
        {"function": {"name": name, "parameters": {"type": "object"}}} for name in names
    ]


def _ask_tools() -> list[dict[str, Any]]:
    """Build an ask_user schema that accepts generated questions."""
    return [
        {
            "function": {
                "name": "ask_user",
                "parameters": {
                    "type": "object",
                    "properties": {"questions": {"type": "array"}},
                    "required": ["questions"],
                },
            }
        }
    ]


async def _stream(
    provider: FakerLLMProvider,
    messages: list[Any],
    tools: list[dict[str, Any]] | None = None,
    abort_event: asyncio.Event | None = None,
) -> list[Any]:
    """Collect every event from one deterministic stream."""
    return [
        event
        async for event in provider.chat_stream(
            messages, model="fake", tools=tools, abort_event=abort_event
        )
    ]


@pytest.mark.asyncio
async def test_chat_is_deterministic_for_same_request() -> None:
    provider = FakerLLMProvider(seed=7, reasoning_probability=0)
    messages = [{"role": "user", "content": "Explain this project."}]

    first = await provider.chat(messages, model="fake")
    second = await provider.chat(messages, model="fake")

    assert isinstance(first, Done)
    assert isinstance(second, Done)
    assert first.content == second.content
    assert first.content


@pytest.mark.asyncio
async def test_chat_without_seed_is_still_deterministic() -> None:
    provider = FakerLLMProvider(reasoning_probability=0)
    messages = [{"role": "user", "content": "Explain this project."}]

    first = await provider.chat(messages, model="fake")
    second = await provider.chat(messages, model="fake")

    assert isinstance(first, Done)
    assert isinstance(second, Done)
    assert first.content == second.content


@pytest.mark.asyncio
async def test_greeting_uses_greeting_replies() -> None:
    provider = FakerLLMProvider(seed=7, reasoning_probability=0)

    for greeting in ("hi", "hello", "hey", "你好", "嗨", "早上好", "晚上好"):
        response = await provider.chat([{"role": "user", "content": greeting}])

        assert isinstance(response, Done)
        assert response.content in FakerLLMProvider._GREETING_REPLIES


@pytest.mark.asyncio
async def test_plain_request_uses_generic_replies() -> None:
    provider = FakerLLMProvider(seed=7, reasoning_probability=0)

    response = await provider.chat([{"role": "user", "content": "Explain this."}])

    assert isinstance(response, Done)
    assert response.content in FakerLLMProvider._GENERIC_REPLIES


@pytest.mark.asyncio
async def test_message_objects_and_extra_kwargs_are_accepted() -> None:
    provider = FakerLLMProvider(seed=7, reasoning_probability=0)

    response = await provider.chat(
        [Message(role="user", content="hello")],
        model="fake",
        stream=True,
        session_id="s-1",
        reasoning_effort="low",
    )

    assert isinstance(response, Done)
    assert response.content in FakerLLMProvider._GREETING_REPLIES


@pytest.mark.asyncio
async def test_empty_and_userless_histories_fall_back_to_generic() -> None:
    provider = FakerLLMProvider(seed=7, reasoning_probability=0)

    empty = await provider.chat([])
    userless = await provider.chat([{"role": "tool", "content": "x"}])
    non_string = await provider.chat([{"role": "user", "content": 123}])  # type: ignore[dict-item]

    assert isinstance(empty, Done) and empty.content in provider._GENERIC_REPLIES
    assert isinstance(userless, Done) and "x" in userless.content
    assert isinstance(non_string, Done) and non_string.content


@pytest.mark.asyncio
async def test_chat_reports_simulated_error() -> None:
    provider = FakerLLMProvider(seed=1, error_probability=1)

    response = await provider.chat([{"role": "user", "content": "Hello"}])
    events = await _stream(provider, [{"role": "user", "content": "Hello"}])

    assert isinstance(response, Error)
    assert isinstance(events[0], Error)


@pytest.mark.asyncio
async def test_tool_summary_embeds_compacted_result() -> None:
    provider = FakerLLMProvider(
        seed=4, reasoning_probability=0, tool_call_probability=0
    )
    messages = [
        {"role": "user", "content": "inspect the project"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "README content"},
    ]

    events = await _stream(provider, messages, tools=_tools("read"))

    assert not any(isinstance(event, ToolCall) for event in events)
    assert isinstance(events[-1], Done)
    assert "README content" in events[-1].content


@pytest.mark.asyncio
async def test_failure_tool_result_uses_failure_template() -> None:
    provider = FakerLLMProvider(
        seed=4, reasoning_probability=0, tool_call_probability=0
    )
    messages = [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "Error: failed to read"},
    ]

    response = await provider.chat(messages, tools=_tools("read"))

    assert isinstance(response, Done)
    assert response.content in FakerLLMProvider._FAILURE_REPLIES


@pytest.mark.asyncio
async def test_blank_tool_result_uses_fallback_text() -> None:
    provider = FakerLLMProvider(
        seed=4, reasoning_probability=0, tool_call_probability=0
    )
    messages = [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "   "},
    ]

    response = await provider.chat(messages, tools=_tools("read"))

    assert isinstance(response, Done)
    assert "工具没有返回可展示的内容" in response.content


@pytest.mark.asyncio
async def test_long_tool_result_is_truncated_to_500_chars() -> None:
    provider = FakerLLMProvider(
        seed=4, reasoning_probability=0, tool_call_probability=0
    )
    messages = [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 600},
    ]

    response = await provider.chat(messages, tools=_tools("read"))

    assert isinstance(response, Done)
    assert "x" * 500 in response.content
    assert "x" * 501 not in response.content


@pytest.mark.asyncio
async def test_random_tool_call_uses_top_level_schema_shape() -> None:
    provider = FakerLLMProvider(
        seed=1,
        reasoning_probability=0,
        tool_call_probability=1,
        continue_tool_probability=1,
        max_tool_calls_per_turn=1,
    )
    tools = [
        {
            "name": "read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }
    ]

    response = await provider.chat(
        [{"role": "tool", "tool_call_id": "c1", "content": "ok"}], tools=tools
    )

    assert isinstance(response, Done)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "read"
    assert json.loads(response.tool_calls[0].arguments) == {"path": "."}


@pytest.mark.asyncio
async def test_zero_tool_probability_after_results_returns_text() -> None:
    provider = FakerLLMProvider(
        seed=1,
        reasoning_probability=0,
        tool_call_probability=0,
        continue_tool_probability=0,
    )
    messages = [{"role": "tool", "tool_call_id": "c1", "content": "ok"}]

    response = await provider.chat(messages, tools=_tools("read"))

    assert isinstance(response, Done)
    assert response.tool_calls == []
    assert response.content


@pytest.mark.asyncio
async def test_max_tool_rounds_stops_further_calls() -> None:
    provider = FakerLLMProvider(
        seed=1,
        reasoning_probability=0,
        tool_call_probability=1,
        continue_tool_probability=1,
        max_tool_rounds=1,
    )
    messages = [
        {"role": "user", "content": "inspect"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]

    response = await provider.chat(messages, tools=_tools("read"))

    assert isinstance(response, Done)
    assert response.tool_calls == []


@pytest.mark.asyncio
async def test_nameless_tool_schema_is_skipped() -> None:
    provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=1
    )

    response = await provider.chat(
        [{"role": "tool", "tool_call_id": "c1", "content": "ok"}],
        tools=[{"function": {}}],
    )

    assert isinstance(response, Done)
    assert response.tool_calls == []
    assert response.content


@pytest.mark.asyncio
async def test_no_tools_returns_text() -> None:
    provider = FakerLLMProvider(seed=1, reasoning_probability=0)

    response = await provider.chat([{"role": "user", "content": "hi"}], tools=None)

    assert isinstance(response, Done)
    assert response.content


@pytest.mark.asyncio
async def test_stream_emits_text_and_matching_done() -> None:
    provider = FakerLLMProvider(seed=1, reasoning_probability=0)

    events = await _stream(provider, [{"role": "user", "content": "Hello"}])

    assert isinstance(events[-1], Done)
    assert any(isinstance(event, TextDelta) for event in events)
    assert events[-1].content == "".join(
        event.content for event in events if isinstance(event, TextDelta)
    )


@pytest.mark.asyncio
async def test_stream_reasoning_comes_before_text() -> None:
    provider = FakerLLMProvider(seed=1, reasoning_probability=1)

    events = await _stream(provider, [{"role": "user", "content": "Hello"}])
    deltas = [event for event in events if isinstance(event, ReasoningDelta)]

    assert deltas
    assert isinstance(events[0], ReasoningDelta)
    assert isinstance(events[-1], Done)
    assert any(isinstance(event, TextDelta) for event in events)


@pytest.mark.asyncio
async def test_stream_abort_before_text_ends_aborted() -> None:
    provider = FakerLLMProvider(seed=1, reasoning_probability=0)
    abort_event = asyncio.Event()
    abort_event.set()

    events = await _stream(
        provider, [{"role": "user", "content": "Hello"}], abort_event=abort_event
    )

    assert isinstance(events[-1], Done)
    assert events[-1].aborted is True
    assert not any(isinstance(event, TextDelta) for event in events)


@pytest.mark.asyncio
async def test_stream_abort_during_reasoning_ends_aborted() -> None:
    provider = FakerLLMProvider(seed=1, reasoning_probability=1)
    abort_event = asyncio.Event()
    abort_event.set()

    events = await _stream(
        provider, [{"role": "user", "content": "Hello"}], abort_event=abort_event
    )

    assert isinstance(events[-1], Done)
    assert events[-1].aborted is True
    assert not any(isinstance(event, ReasoningDelta) for event in events)


@pytest.mark.asyncio
async def test_stream_abort_before_tool_calls_ends_aborted() -> None:
    provider = FakerLLMProvider(
        seed=4, reasoning_probability=0, tool_call_probability=1
    )
    abort_event = asyncio.Event()
    abort_event.set()

    events = await _stream(
        provider,
        [{"role": "user", "content": "inspect the project"}],
        tools=_tools("read"),
        abort_event=abort_event,
    )

    assert isinstance(events[-1], Done)
    assert events[-1].aborted is True
    assert not any(isinstance(event, ToolCall) for event in events)


@pytest.mark.asyncio
async def test_stream_tool_calls_carry_into_done() -> None:
    provider = FakerLLMProvider(
        seed=4, reasoning_probability=0, tool_call_probability=1
    )
    tools = [
        {"function": {"name": "read", "parameters": {"type": "object"}}},
        {"function": {"name": "glob", "parameters": {"type": "object"}}},
    ]

    events = await _stream(
        provider, [{"role": "user", "content": "帮我读取文件并查找文件"}], tools=tools
    )
    calls = [event for event in events if isinstance(event, ToolCall)]

    assert len(calls) == 2
    assert len({call.id for call in calls}) == 2
    assert isinstance(events[-1], Done)
    assert len(events[-1].tool_calls) == 2


@pytest.mark.asyncio
async def test_content_driven_read_extracts_path() -> None:
    provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=0
    )
    tools = [
        {
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}},
                    "required": ["filePath"],
                },
            }
        }
    ]

    events = await _stream(
        provider,
        [{"role": "user", "content": "帮我看下 README.md 的文件内容"}],
        tools=tools,
    )
    calls = [event for event in events if isinstance(event, ToolCall)]

    assert len(calls) == 1
    assert json.loads(calls[0].arguments)["filePath"] == "README.md"


@pytest.mark.asyncio
async def test_content_driven_caps_multi_tool_burst_at_five() -> None:
    provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=0
    )
    tools = _tools("read", "glob", "grep", "shell", "code_run", "web_search")

    events = await _stream(
        provider,
        [
            {
                "role": "user",
                "content": "读取文件 查找文件 搜索代码 运行命令 运行代码 搜索网络",
            }
        ],
        tools=tools,
    )
    names = [event.name for event in events if isinstance(event, ToolCall)]

    assert names == ["read", "glob", "grep", "shell", "code_run"]
    assert isinstance(events[-1], Done)
    assert len(events[-1].tool_calls) == 5


@pytest.mark.asyncio
async def test_content_driven_skips_unavailable_tools() -> None:
    provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=0
    )

    events = await _stream(
        provider, [{"role": "user", "content": "读取文件内容"}], tools=_tools("glob")
    )

    assert not any(isinstance(event, ToolCall) for event in events)
    assert isinstance(events[-1], Done)


def test_content_driven_edge_inputs_return_no_match() -> None:
    names = FakerLLMProvider._content_driven_tool_names
    assert names("", {"read"}) == []
    assert names("读取文件", set()) == []


def test_available_tool_map_dedupes_and_skips_nameless() -> None:
    mapping = FakerLLMProvider._available_tool_map(
        [
            {"function": {"name": "read"}},
            {"function": {"name": "read"}},
            {"function": {}},
            {"name": "glob"},
        ]
    )

    assert set(mapping) == {"read", "glob"}


@pytest.mark.asyncio
async def test_web_fetch_extracts_url() -> None:
    provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=0
    )
    tools = [
        {
            "function": {
                "name": "web_fetch",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            }
        }
    ]

    events = await _stream(
        provider,
        [{"role": "user", "content": "帮我抓取网页 https://example.com/docs"}],
        tools=tools,
    )
    calls = [event for event in events if isinstance(event, ToolCall)]

    assert len(calls) == 1
    assert json.loads(calls[0].arguments)["url"] == "https://example.com/docs"


@pytest.mark.asyncio
async def test_glob_fills_path_query_and_pattern_fields() -> None:
    provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=0
    )
    tools = [
        {
            "function": {
                "name": "glob",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "query": {"type": "string"},
                        "pattern": {"type": "string"},
                    },
                },
            }
        }
    ]

    events = await _stream(
        provider,
        [{"role": "user", "content": "查找文件 ./src/app.py " + "x" * 100}],
        tools=tools,
    )
    args = json.loads(next(e for e in events if isinstance(e, ToolCall)).arguments)

    assert args["path"] == "./src/app.py"
    assert args["query"] == args["pattern"]
    assert len(args["query"]) == 80


@pytest.mark.asyncio
async def test_read_image_tool_fires_on_image_request() -> None:
    provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=0
    )
    tools = [
        {
            "function": {
                "name": "read_image",
                "parameters": {
                    "type": "object",
                    "properties": {"filePath": {"type": "string"}},
                },
            }
        }
    ]

    events = await _stream(
        provider, [{"role": "user", "content": "帮我看图分析图片内容"}], tools=tools
    )
    calls = [event for event in events if isinstance(event, ToolCall)]

    assert len(calls) == 1
    assert calls[0].name == "read_image"


@pytest.mark.asyncio
async def test_edit_carries_mock_diff_for_python_file() -> None:
    provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=0
    )
    tools = [
        {
            "function": {
                "name": "edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string"},
                        "oldString": {"type": "string"},
                        "newString": {"type": "string"},
                    },
                    "required": ["filePath", "oldString", "newString"],
                },
            }
        }
    ]

    events = await _stream(
        provider, [{"role": "user", "content": "帮我修改文件 app.py"}], tools=tools
    )
    args = json.loads(next(e for e in events if isinstance(e, ToolCall)).arguments)

    assert args["filePath"] == "app.py"
    assert args["oldString"] != args["newString"]
    assert "def greet" in args["oldString"]


@pytest.mark.asyncio
async def test_edit_carries_mock_diff_for_typescript_file() -> None:
    provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=0
    )
    tools = [
        {
            "function": {
                "name": "edit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string"},
                        "oldString": {"type": "string"},
                        "newString": {"type": "string"},
                    },
                },
            }
        }
    ]

    events = await _stream(
        provider, [{"role": "user", "content": "帮我编辑文件 widget.tsx"}], tools=tools
    )
    args = json.loads(next(e for e in events if isinstance(e, ToolCall)).arguments)

    assert args["filePath"] == "widget.tsx"
    assert "function greet" in args["oldString"]
    assert "`hi ${name}`" in args["newString"]


def test_mock_edit_strings_cover_both_families() -> None:
    py_old, py_new = FakerLLMProvider._mock_edit_strings("a.py")
    ts_old, ts_new = FakerLLMProvider._mock_edit_strings("a.ts")

    assert "def greet" in py_old and 'f"' not in py_old and py_old != py_new
    assert "function greet" in ts_old and ts_old != ts_new


@pytest.mark.asyncio
async def test_todo_write_generates_valid_list() -> None:
    provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=0
    )
    tools = [
        {
            "function": {
                "name": "todo_write",
                "parameters": {
                    "type": "object",
                    "properties": {"todos": {"type": "array"}},
                    "required": ["todos"],
                },
            }
        }
    ]

    events = await _stream(
        provider, [{"role": "user", "content": "帮我列一个任务清单"}], tools=tools
    )
    todos = json.loads(next(e for e in events if isinstance(e, ToolCall)).arguments)[
        "todos"
    ]
    statuses = {todo["status"] for todo in todos}

    assert len(todos) >= 3
    assert sum(1 for t in todos if t["status"] == "in_progress") == 1
    assert {"completed", "pending"} <= statuses


@pytest.mark.asyncio
async def test_ask_user_survey_covers_all_question_types() -> None:
    provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=0
    )

    events = await _stream(
        provider,
        [{"role": "user", "content": "ask me a survey to configure the project"}],
        tools=_ask_tools(),
    )
    questions = json.loads(
        next(e for e in events if isinstance(e, ToolCall)).arguments
    )["questions"]

    assert [q["input_type"] for q in questions] == [
        "text",
        "select",
        "select",
        "textarea",
    ]
    assert questions[2]["multiple"] is True
    assert questions[1]["options"] and questions[1]["options"][0]["description"]


@pytest.mark.asyncio
async def test_ask_user_branches_cover_every_variant() -> None:
    provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=0
    )
    messages = [
        "问我做个调查问卷",
        "ask me about the weather in Paris",
        "问我北京天气怎么样",
        "问我主题选深色还是浅色",
        "ask me about theme color",
        "问我 language 设置",
        "问我是否部署到生产",
        "问我确认删除吗",
        "问我文件路径填什么",
        "问我邮箱地址",
        "问我用 vue 还是 react",
        "问我详细需求",
        "问我一个问题",
    ]

    for text in messages:
        events = await _stream(
            provider, [{"role": "user", "content": text}], _ask_tools()
        )
        calls = [e for e in events if isinstance(e, ToolCall)]

        assert len(calls) == 1, text
        questions = json.loads(calls[0].arguments)["questions"]
        assert questions and all("id" in q and "input_type" in q for q in questions), (
            text
        )


@pytest.mark.asyncio
async def test_ask_user_weather_english_variant() -> None:
    provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=0
    )

    events = await _stream(
        provider,
        [{"role": "user", "content": "ask me about the weather in Paris"}],
        tools=_ask_tools(),
    )
    questions = json.loads(
        next(e for e in events if isinstance(e, ToolCall)).arguments
    )["questions"]

    assert questions
    assert questions[0]["header"] in {"当前城市", "Current City", "出行城市"}


def test_generate_arguments_covers_every_value_kind() -> None:
    generate = FakerLLMProvider._generate_arguments
    schema = {
        "type": "object",
        "properties": {
            "s": {"type": "string"},
            "i": {"type": "integer"},
            "n": {"type": "number"},
            "b": {"type": "boolean"},
            "a": {"type": "array"},
            "o": {"type": "object"},
            "e": {"type": "string", "enum": ["x", "y"]},
            "d": {"type": "string", "default": "v"},
            "u": {"type": "null"},
            "bad": "not-a-dict",
        },
        "required": ["s", "u", "missing"],
    }

    args = generate("read", schema)

    assert args["s"] == "fake-read-s"
    assert args["i"] == 1 and args["n"] == 1
    assert args["b"] is False
    assert args["a"] == [] and args["o"] == {}
    assert args["e"] == "x" and args["d"] == "v"
    assert args["u"] == "" and args["missing"] == ""
    assert "bad" not in args
    assert generate("x", None) == {}  # type: ignore[arg-type]
    assert generate("x", {"type": "object"}) == {}


def test_string_argument_covers_examples_and_fallback() -> None:
    cases = {
        "filePath": "README.md",
        "path": ".",
        "pattern": "*.py",
        "include": "*.py",
        "query": "example query",
        "url": "https://example.com",
        "format": "markdown",
        "code": "print('fake')",
        "script_path": "",
        "command": "printf 'fake'",
        "action": "get_state",
        "skill_name": "example-skill",
        "skill_ref": "example-skill",
    }
    for name, expected in cases.items():
        assert FakerLLMProvider._string_argument("tool", name) == expected
    assert FakerLLMProvider._string_argument("read", "other") == "fake-read-other"


def test_contextual_arguments_rejects_non_dict_schema() -> None:
    base = FakerLLMProvider._contextual_arguments(
        "read",
        "msg",
        ["not", "a", "dict"],
        random.Random(0),  # type: ignore[arg-type]
    )

    assert base == {}


def test_contextual_arguments_rejects_non_dict_properties() -> None:
    from collections import UserDict

    schema: dict[str, Any] = {"properties": UserDict({"a": "x"})}
    base = FakerLLMProvider._contextual_arguments(
        "read", "msg", schema, random.Random(0)
    )

    assert base == {}


def test_failure_markers_and_greeting_normalization() -> None:
    looks_failure = FakerLLMProvider._looks_like_failure
    assert looks_failure("Error: boom")
    assert looks_failure("FAILED")
    assert looks_failure("failure here")
    assert looks_failure("发生错误了")
    assert looks_failure("任务失败")
    assert not looks_failure("all good")
    assert FakerLLMProvider._looks_like_greeting("  HELLO  ")
    assert not FakerLLMProvider._looks_like_greeting("hello there")


def test_stream_chunks_split_cjk_and_ascii() -> None:
    chunks = FakerLLMProvider._stream_chunks("ab 你好 cd")

    assert "".join(chunks) == "ab 你好 cd"
    assert "你" in chunks and "好" in chunks
    assert FakerLLMProvider._stream_chunks("") == []


def test_extract_url_path_and_condense() -> None:
    assert FakerLLMProvider._extract_url("see https://a.com/x now") == "https://a.com/x"
    assert FakerLLMProvider._extract_url("no link") == ""
    assert FakerLLMProvider._extract_path("open ./src/app.py please") == "./src/app.py"
    assert FakerLLMProvider._extract_path("see https://a.com/x now") == ""
    assert FakerLLMProvider._extract_path("no path here") == ""
    assert FakerLLMProvider._condense("  a  b  ") == "a b"
    assert len(FakerLLMProvider._condense("x" * 100)) == 80


def test_tool_rounds_and_results_helpers() -> None:
    messages: list[Any] = [
        {"role": "user", "content": "hi"},
        Message(role="assistant", content="", tool_calls=[{"id": "c1"}]),
        {"role": "assistant", "content": "no calls"},
        {"role": "tool", "tool_call_id": "c1", "content": "out"},
        Message(role="tool", content="more", tool_call_id="c2"),
    ]

    assert FakerLLMProvider._tool_rounds(messages) == 1
    assert FakerLLMProvider._tool_results(messages) == ["out", "more"]
    assert FakerLLMProvider._latest_tool_result(messages) == "more"
    assert (
        FakerLLMProvider._latest_tool_result([{"role": "user", "content": "x"}]) is None
    )
    assert FakerLLMProvider._latest_user_message(messages) == "hi"


def test_constructor_clamps_bounds() -> None:
    provider = FakerLLMProvider(max_tool_calls_per_turn=0, stream_delay=-1)

    assert provider._max_tool_calls_per_turn == 1
    assert provider._stream_delay == 0.0


@pytest.mark.asyncio
async def test_count_tokens_contract() -> None:
    provider = FakerLLMProvider()

    assert await provider.count_tokens("12345678") == 2
    assert await provider.count_tokens("ab") == 1
    assert await provider.count_tokens("") == 0


@pytest.mark.asyncio
async def test_stream_delay_sleeps_between_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[float] = []

    async def fake_sleep(duration: float) -> None:
        recorded.append(duration)

    monkeypatch.setattr("novakit.llm.faker.asyncio.sleep", fake_sleep)
    provider = FakerLLMProvider(seed=1, reasoning_probability=1, stream_delay=0.05)

    events = await _stream(provider, [{"role": "user", "content": "Hello"}])

    assert any(isinstance(event, ReasoningDelta) for event in events)
    assert recorded
    assert all(duration == 0.05 for duration in recorded)


@pytest.mark.asyncio
async def test_stream_delay_applies_to_tool_and_text_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[float] = []

    async def fake_sleep(duration: float) -> None:
        recorded.append(duration)

    monkeypatch.setattr("novakit.llm.faker.asyncio.sleep", fake_sleep)

    tool_provider = FakerLLMProvider(
        seed=1, reasoning_probability=0, tool_call_probability=1, stream_delay=0.01
    )
    await _stream(
        tool_provider,
        [{"role": "user", "content": "inspect the project"}],
        tools=_tools("read"),
    )
    tool_sleeps = list(recorded)
    recorded.clear()

    text_provider = FakerLLMProvider(seed=1, reasoning_probability=0, stream_delay=0.01)
    await _stream(text_provider, [{"role": "user", "content": "Hello"}])
    text_sleeps = list(recorded)

    assert tool_sleeps and all(s == 0.01 for s in tool_sleeps)
    assert text_sleeps and all(s == 0.01 for s in text_sleeps)


@pytest.mark.asyncio
async def test_no_stream_delay_skips_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[float] = []

    async def fake_sleep(duration: float) -> None:
        recorded.append(duration)

    monkeypatch.setattr("novakit.llm.faker.asyncio.sleep", fake_sleep)
    provider = FakerLLMProvider(seed=1, reasoning_probability=1, stream_delay=0.0)

    await _stream(provider, [{"role": "user", "content": "Hello"}])

    assert recorded == []
