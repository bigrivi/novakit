"""Offline tests for novakit.llm.tokenizer.

Covers type/script-aware estimation, the tiktoken fast path and its
fallbacks, CJK counting including the huge-input sampling path, and model-id
normalisation plus context-window resolution.
"""

from __future__ import annotations

import sys

import pytest
import tiktoken

from novakit.llm import tokenizer
from novakit.llm.provider import Message
from novakit.llm.tokenizer import (
    CHARS_PER_TOKEN_TEXT,
    CHARS_PER_TOKEN_TOOL,
    IMAGE_CHAR_ESTIMATE,
    count_cjk_chars,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens_by_type,
    normalise_model_id,
    resolve_context_limit,
    resolve_context_window,
)

CJK_TEXT = "中文测试"


def _user(content: object) -> Message:
    return Message(role="user", content=content)  # type: ignore[arg-type]


class TestEstimateTokensByType:
    def test_empty_text_returns_zero(self) -> None:
        assert estimate_tokens_by_type("", is_tool_result=False) == 0

    def test_none_text_returns_zero(self) -> None:
        assert estimate_tokens_by_type(None, is_tool_result=False) == 0  # type: ignore[arg-type]

    def test_prose_uses_four_chars_per_token(self) -> None:
        text = "Hello world, this is a test."

        result = estimate_tokens_by_type(text)

        assert result == len(text) // CHARS_PER_TOKEN_TEXT

    def test_tool_results_use_two_chars_per_token(self) -> None:
        assert (
            estimate_tokens_by_type("A" * 100, is_tool_result=True)
            == 100 // CHARS_PER_TOKEN_TOOL
        )

    def test_single_char_clamps_to_one(self) -> None:
        assert estimate_tokens_by_type("a") == 1

    def test_cjk_counts_one_token_per_char(self) -> None:
        assert estimate_tokens_by_type(CJK_TEXT) == len(CJK_TEXT)

    def test_cjk_in_tool_results_still_counts_per_char(self) -> None:
        assert estimate_tokens_by_type("中文" * 5, is_tool_result=True) == 10

    def test_mixed_cjk_and_latin(self) -> None:
        text = "hi" + CJK_TEXT

        result = estimate_tokens_by_type(text)

        assert result == len(CJK_TEXT) + 2 // CHARS_PER_TOKEN_TEXT


class TestCountCjkChars:
    def test_counts_cjk_in_short_text(self) -> None:
        assert count_cjk_chars("ab中文cd") == 2

    def test_no_cjk_returns_zero(self) -> None:
        assert count_cjk_chars("hello world") == 0

    @pytest.mark.parametrize(
        "char",
        ["あ", "ア", "中", "한", "𠀀"],
    )
    def test_each_cjk_block_counts(self, char: str) -> None:
        assert count_cjk_chars(char) == 1

    def test_huge_input_is_sampled(self) -> None:
        text = "ab中" * 700_000
        assert len(text) > tokenizer._EXACT_COUNT_MAX_CHARS

        result = count_cjk_chars(text)

        assert abs(result - 700_000) < 1_000

    def test_empty_samples_return_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tokenizer, "_EXACT_COUNT_MAX_CHARS", 4)
        monkeypatch.setattr(tokenizer, "_CJK_SAMPLE_WINDOW", 0)

        assert count_cjk_chars("abcdef") == 0

    def test_is_cjk_predicate(self) -> None:
        assert tokenizer._is_cjk("中") is True
        assert tokenizer._is_cjk("a") is False


class TestEstimateMessageTokens:
    def test_user_string_uses_prose_ratio(self) -> None:
        text = "the assistant completed the recommendations for the plaza"

        result = estimate_message_tokens(_user(text), model="unknown")

        assert result == len(text) // CHARS_PER_TOKEN_TEXT

    def test_tool_string_uses_dense_ratio(self) -> None:
        result = estimate_message_tokens(
            Message(role="tool", content="A" * 100), model="unknown"
        )

        assert result == 100 // CHARS_PER_TOKEN_TOOL

    def test_dict_message_with_string_content(self) -> None:
        result = estimate_message_tokens(
            {"role": "user", "content": "Hello world"}, model="unknown"
        )

        assert result == len("Hello world") // CHARS_PER_TOKEN_TEXT

    def test_dict_tool_message_uses_dense_ratio(self) -> None:
        result = estimate_message_tokens(
            {"role": "tool", "content": "AB" * 50}, model="unknown"
        )

        assert result == 100 // CHARS_PER_TOKEN_TOOL

    def test_dict_without_role_and_content_defaults(self) -> None:
        assert estimate_message_tokens({}, model="unknown") == 0

    def test_none_content_counts_zero(self) -> None:
        assert estimate_message_tokens(_user(None), model="unknown") == 0

    def test_text_blocks_sum(self) -> None:
        msg = _user(
            [{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]
        )

        assert estimate_message_tokens(msg, model="unknown") == 2

    def test_tool_role_text_blocks_use_dense_ratio(self) -> None:
        msg = Message(role="tool", content=[{"type": "text", "text": "A" * 100}])

        assert (
            estimate_message_tokens(msg, model="unknown") == 100 // CHARS_PER_TOKEN_TOOL
        )

    def test_image_block_has_fixed_estimate(self) -> None:
        msg = _user([{"type": "image"}])

        result = estimate_message_tokens(msg, model="unknown")

        assert result == IMAGE_CHAR_ESTIMATE // CHARS_PER_TOKEN_TEXT

    def test_thinking_block_counts_as_prose(self) -> None:
        msg = _user(
            [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "Hello"},
            ]
        )

        assert estimate_message_tokens(msg, model="unknown") == 4

    def test_thinking_block_accepts_non_string(self) -> None:
        msg = _user([{"type": "thinking", "thinking": {"a": 1}}])

        assert estimate_message_tokens(msg, model="unknown") > 0

    def test_unhandled_block_types_count_zero(self) -> None:
        msg = _user([{"type": "toolCall", "name": "read"}])

        assert estimate_message_tokens(msg, model="unknown") == 0

    def test_non_dict_blocks_are_skipped(self) -> None:
        msg = _user([{"type": "text", "text": "Hello"}, "stray"])

        assert estimate_message_tokens(msg, model="unknown") == 1

    def test_tool_calls_add_arguments_estimate(self) -> None:
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[{"name": "read", "arguments": {"file": "test.py"}}],
        )

        assert estimate_message_tokens(msg, model="unknown") > 0

    def test_non_dict_tool_calls_are_skipped(self) -> None:
        msg = Message(
            role="assistant",
            content="ABCD",
            tool_calls=["junk", {"name": "read", "arguments": {"x": 1}}],
        )

        dict_msg = {"role": "assistant", "content": "ABCD", "tool_calls": ["junk"]}

        assert estimate_message_tokens(msg, model="unknown") > 1
        assert estimate_message_tokens(dict_msg, model="unknown") == 1

    def test_dict_tool_calls_none_defaults_to_empty(self) -> None:
        assert (
            estimate_message_tokens(
                {"role": "assistant", "content": "ABCD", "tool_calls": None},
                model="unknown",
            )
            == 1
        )

    def test_non_text_blocks_skip_tiktoken(self) -> None:
        msg = _user([{"type": "toolCall", "name": "read"}])
        msg.tool_calls = [{"name": "read", "arguments": {"file": "test.py"}}]

        result = estimate_message_tokens(msg, model="gpt-4o")

        assert result > 0

    def test_openai_model_uses_tiktoken(self) -> None:
        text = "Hello world"
        expected = len(tiktoken.encoding_for_model("gpt-4o").encode(text))

        result = estimate_message_tokens(_user(text), model="gpt-4o")

        assert result == expected == 2

    def test_openai_prefix_model_uses_tiktoken(self) -> None:
        result = estimate_message_tokens(_user("Hello world"), model="openai-owl")

        assert result > 0

    def test_tiktoken_unavailable_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "tiktoken", None)

        result = estimate_message_tokens(_user("Hello world"), model="gpt-4o")

        assert result == len("Hello world") // CHARS_PER_TOKEN_TEXT

    def test_tiktoken_key_error_uses_base_encoding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(model: str) -> tiktoken.Encoding:
            raise KeyError(model)

        monkeypatch.setattr(tiktoken, "encoding_for_model", _raise)

        result = tokenizer._estimate_with_tiktoken(_user("Hello world"), "gpt-4o")

        assert result == len(tiktoken.get_encoding("cl100k_base").encode("Hello world"))

    def test_tiktoken_generic_error_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            tiktoken, "get_encoding", lambda name: (_ for _ in ()).throw(RuntimeError())
        )

        assert tokenizer._estimate_with_tiktoken(_user("Hello"), "gpt-zzz-9") is None

    def test_empty_text_returns_none(self) -> None:
        assert tokenizer._estimate_with_tiktoken(_user(""), "gpt-4o") is None

    def test_list_with_text_joins_parts(self) -> None:
        msg = _user([{"type": "text", "text": "Hello"}])

        result = tokenizer._estimate_with_tiktoken(msg, model="gpt-4o")

        assert result == len(tiktoken.encoding_for_model("gpt-4o").encode("Hello"))

    def test_list_without_text_returns_none(self) -> None:
        msg = _user([{"type": "image"}, "stray"])

        assert tokenizer._estimate_with_tiktoken(msg, model="gpt-4o") is None


class TestEstimateMessagesTokens:
    def test_empty_list(self) -> None:
        assert estimate_messages_tokens([]) == 0

    def test_sums_messages(self) -> None:
        messages = [_user("Hello"), _user("Hi there!"), _user("How are you?")]

        result = estimate_messages_tokens(messages, model="unknown")

        assert result == sum(
            estimate_message_tokens(m, model="unknown") for m in messages
        )


class TestNormaliseModelId:
    def test_gateway_prefix_is_stripped(self) -> None:
        assert normalise_model_id("meta/muse-spark-1.2") == "muse-spark-1.2"

    def test_ollama_tag_is_stripped(self) -> None:
        assert normalise_model_id("gemma4:26b") == "gemma4"

    def test_bracket_variant_is_stripped(self) -> None:
        assert normalise_model_id("qwen[8k]") == "qwen"

    def test_dated_suffix_is_stripped(self) -> None:
        assert normalise_model_id("gpt-4o-2026-04-15") == "gpt-4o"
        assert normalise_model_id("claude-sonnet-4-20250514") == "claude-sonnet-4"

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gpt-4o-free", "gpt-4o"),
            ("gpt-5-contributor", "gpt-5"),
            ("gpt-4o-preview", "gpt-4o"),
            ("gpt-4o-latest", "gpt-4o"),
            ("gpt-4o-cloud", "gpt-4o"),
            ("gpt-5-thinking", "gpt-5"),
        ],
    )
    def test_tier_suffixes_are_stripped(self, model: str, expected: str) -> None:
        assert normalise_model_id(model) == expected

    def test_whitespace_and_case(self) -> None:
        assert normalise_model_id("  OpenAI/GPT-4o-2026-04-15  ") == "gpt-4o"

    def test_empty(self) -> None:
        assert normalise_model_id("  ") == ""


class TestResolveContextWindow:
    def test_exact_model_id(self) -> None:
        assert resolve_context_window("gpt-4", "openai") == (8192, "exact")

    def test_exact_after_normalisation(self) -> None:
        assert resolve_context_window("org/gpt-4", "openai") == (8192, "exact")

    def test_variant_marker_overrides_family(self) -> None:
        assert resolve_context_window("big-pickle[1m]", "gw") == (1000000, "variant")
        assert resolve_context_window("big-pickle-1m", "gw") == (1000000, "variant")

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gpt-4o-2024", 128000),
            ("gpt-5-xhigh", 400000),
            ("o1-mini", 128000),
            ("o1-preview", 200000),
            ("codex-mini", 400000),
            ("claude-sonnet-4-5", 200000),
            ("claude-fable-1", 1000000),
            ("gemini-2.5-pro", 1048576),
            ("gemma-2-9b", 8192),
            ("gemma3-4b", 131072),
            ("meta/muse-spark-1.3", 1048576),
            ("llama-3.1-8b", 131072),
            ("deepseek-v3", 131072),
            ("qwen3-8b", 131072),
            ("qwen2-7b", 32768),
            ("kimi-k2-0905", 262144),
            ("moonshot-v1", 131072),
            ("glm-4.6", 204800),
            ("minimax-m1", 1000000),
            ("grok-4", 262144),
            ("mistral-large-2", 131072),
            ("mixtral-8x7b", 32768),
            ("codestral-latest", 262144),
            ("phi-4", 16384),
            ("command-r-plus", 131072),
            ("nova-pro", 300000),
            ("mimo-v2-flash", 131072),
        ],
    )
    def test_family_windows(self, model: str, expected: int) -> None:
        window, source = resolve_context_window(model, "gw")

        assert window == expected
        assert source.startswith("family:")

    def test_unknown_model_falls_back_to_default(self) -> None:
        assert resolve_context_window("unknown-model", "openai") == (
            tokenizer.DEFAULT_CONTEXT_WINDOW,
            "default",
        )

    def test_o1_mini_beats_plain_o1(self) -> None:
        assert resolve_context_window("o1-mini", "openai")[0] == 128000


class TestResolveContextLimit:
    def test_known_model_returns_window(self) -> None:
        assert resolve_context_limit("gpt-4o", provider="openai") == 128000
        assert resolve_context_limit("gpt-4", provider="openai") == 8192
        assert resolve_context_limit("gemma4:26b", provider="ollama") == 32000

    def test_unknown_model_warns_and_returns_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="novakit.llm.tokenizer"):
            result = resolve_context_limit("unknown-model", provider="openai")

        assert result == 128000
        assert any("unknown-model" in r.message for r in caplog.records)

    def test_known_model_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="novakit.llm.tokenizer"):
            resolve_context_limit("gpt-4o", provider="openai")

        assert caplog.records == []
