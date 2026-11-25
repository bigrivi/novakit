"""Offline tests for novakit.llm.reasoning and novakit.llm.user_agent.

Reasoning levels come only from config (or the catalogue spec row); the tests
pin the declaration rules, the per-provider wire spellings, and the
User-Agent identity token.
"""

from __future__ import annotations

import platform
import re

import pytest

from novakit import __version__
from novakit.llm import reasoning
from novakit.llm.reasoning import (
    EFFORT_PROVIDER_TYPES,
    LEVELS_CONFIG_KEY,
    apply_effort,
    fit_effort,
    resolve_effort_levels,
)
from novakit.llm.user_agent import default_user_agent

DEFAULT_SHAPE = re.compile(r"^novakit/\S+ \([a-z0-9_]+; [a-z0-9_]+\)$")


class TestResolveEffortLevels:
    def test_declared_levels_are_returned_in_order(self) -> None:
        levels = resolve_effort_levels(
            "gpt-5.5",
            "openai-compatible",
            {"reasoning_effort_levels": ["low", "medium", "high"]},
        )

        assert levels == ["low", "medium", "high"]

    def test_response_provider_accepts_declarations(self) -> None:
        levels = resolve_effort_levels(
            "gpt-5",
            "openai-response",
            {"reasoning_effort_levels": ["low", "high"]},
        )

        assert levels == ["low", "high"]

    def test_effort_provider_types(self) -> None:
        assert {"openai-compatible", "openai-response"} == EFFORT_PROVIDER_TYPES
        assert LEVELS_CONFIG_KEY == "reasoning_effort_levels"

    @pytest.mark.parametrize(
        "model",
        ["gpt-5.5", "gpt-5.1", "deepseek-v4", "qwen3.7-plus", "grok-4", "kimi-k2-0905"],
    )
    def test_a_recognisable_name_still_offers_nothing(self, model: str) -> None:
        assert resolve_effort_levels(model, "openai-compatible", {"name": model}) == []

    def test_alias_needs_a_declaration(self) -> None:
        assert resolve_effort_levels("mimo-v2.5", "openai-compatible", {}) == []
        assert resolve_effort_levels(
            "mimo-v2.5",
            "openai-compatible",
            {"reasoning_effort_levels": ["low", "high"]},
        ) == ["low", "high"]

    def test_missing_model_config_offers_nothing(self) -> None:
        assert resolve_effort_levels("gpt-5.5", "openai-compatible", None) == []

    @pytest.mark.parametrize("provider_type", ["anthropic", "ollama", "faker", ""])
    def test_provider_without_the_concept_offers_nothing(
        self, provider_type: str
    ) -> None:
        levels = resolve_effort_levels(
            "claude-opus-5",
            provider_type,
            {"reasoning_effort_levels": ["low", "high"]},
        )

        assert levels == []

    def test_bare_string_is_accepted_as_one_level(self) -> None:
        assert resolve_effort_levels(
            "gpt-5-pro", "openai-response", {"reasoning_effort_levels": "high"}
        ) == ["high"]

    def test_tuple_declaration_is_accepted(self) -> None:
        assert resolve_effort_levels(
            "gpt-5.5", "openai-compatible", {"reasoning_effort_levels": ("low", "high")}
        ) == ["low", "high"]

    def test_blank_and_duplicate_entries_are_dropped(self) -> None:
        assert resolve_effort_levels(
            "gpt-5.5",
            "openai-compatible",
            {"reasoning_effort_levels": ["low", "  ", "high", "low", " high "]},
        ) == ["low", "high"]

    @pytest.mark.parametrize("declared", [[], (), None, 42, {"low": 1}, ["  "]])
    def test_a_declaration_of_nothing_means_no_control(self, declared: object) -> None:
        assert (
            resolve_effort_levels(
                "gpt-5.5", "openai-compatible", {"reasoning_effort_levels": declared}
            )
            == []
        )

    def test_catalogue_spec_supplies_levels(self) -> None:
        levels = resolve_effort_levels(
            "gpt-5.5",
            "openai-compatible",
            None,
            {"reasoning_effort_levels": ["low", "high"]},
        )

        assert levels == ["low", "high"]

    def test_catalogue_spec_dedupes_and_trims(self) -> None:
        levels = resolve_effort_levels(
            "gpt-5.5",
            "openai-compatible",
            None,
            {"reasoning_effort_levels": ["low", " ", "high", "low"]},
        )

        assert levels == ["low", "high"]

    def test_config_declaration_beats_catalogue(self) -> None:
        levels = resolve_effort_levels(
            "gpt-5.5",
            "openai-compatible",
            {"reasoning_effort_levels": ["low"]},
            {"reasoning_effort_levels": ["low", "high"]},
        )

        assert levels == ["low"]

    @pytest.mark.parametrize(
        "spec",
        [
            None,
            {},
            {"reasoning_effort_levels": "high"},
            {"reasoning_effort_levels": []},
        ],
    )
    def test_empty_catalogue_means_no_control(self, spec: dict | None) -> None:
        assert resolve_effort_levels("gpt-5.5", "openai-compatible", None, spec) == []


class TestFitEffort:
    def test_a_declared_level_survives(self) -> None:
        assert fit_effort(["low", "high"], "high") == "high"

    def test_an_undeclared_level_is_dropped(self) -> None:
        assert fit_effort(["low", "high"], "xhigh") is None

    def test_no_levels_means_nothing_survives(self) -> None:
        assert fit_effort([], "high") is None

    @pytest.mark.parametrize("effort", [None, ""])
    def test_absent_effort_stays_absent(self, effort: str | None) -> None:
        assert fit_effort(["low", "high"], effort) is None


class TestApplyEffort:
    def test_chat_completions_takes_a_flat_field(self) -> None:
        assert apply_effort({}, "high", "openai-compatible") == {
            "reasoning_effort": "high"
        }

    def test_responses_api_nests_it(self) -> None:
        assert apply_effort({}, "high", "openai-response") == {
            "reasoning": {"effort": "high"}
        }

    def test_existing_flat_value_is_left_alone(self) -> None:
        body = apply_effort({"reasoning_effort": "medium"}, "high", "openai-compatible")

        assert body["reasoning_effort"] == "medium"

    def test_existing_nested_block_is_kept_intact(self) -> None:
        body = apply_effort(
            {"reasoning": {"summary": "auto"}}, "high", "openai-response"
        )

        assert body["reasoning"] == {"summary": "auto"}

    @pytest.mark.parametrize("provider_type", ["anthropic", "ollama", "faker"])
    def test_providers_without_the_concept_are_left_alone(
        self, provider_type: str
    ) -> None:
        body = {"model": "claude-opus-5"}

        assert apply_effort(body, "high", provider_type) == body

    @pytest.mark.parametrize("effort", [None, ""])
    def test_no_effort_leaves_the_body_untouched(self, effort: str | None) -> None:
        body: dict = {"model": "gpt-5.5"}

        assert apply_effort(body, effort, "openai-compatible") is body

    def test_module_exports(self) -> None:
        assert set(reasoning.__all__) == {
            "EFFORT_PROVIDER_TYPES",
            "LEVELS_CONFIG_KEY",
            "apply_effort",
            "fit_effort",
            "resolve_effort_levels",
        }


class TestDefaultUserAgent:
    def test_the_default_names_the_product_and_its_version(self) -> None:
        agent = default_user_agent()

        assert DEFAULT_SHAPE.match(agent), agent
        assert __version__ in agent
        assert "python" not in agent.lower()

    def test_custom_product_and_version(self) -> None:
        agent = default_user_agent("myapp", "9.9.9")

        assert agent.startswith("myapp/9.9.9 (")

    def test_explicit_version_overrides_package_version(self) -> None:
        agent = default_user_agent(version="1.2.3")

        assert agent.startswith("novakit/1.2.3 (")
        assert __version__ not in agent.split(" ")[0]

    def test_blank_platform_falls_back_to_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(platform, "system", lambda: "")
        monkeypatch.setattr(platform, "machine", lambda: "")

        assert default_user_agent() == f"novakit/{__version__} (unknown; unknown)"
