"""Offline tour of the LLM utility layer: tokenizer, images, reasoning, retry.

What you'll learn:
    * Token estimation and context-window helpers from
      ``novakit.llm.tokenizer``.
    * Image data-URL parsing, sizing, downscaling, and budgeting from
      ``novakit.llm.images``.
    * Reasoning-effort levels (``resolve`` / ``fit`` / ``apply``).
    * ``RetryBudget`` decisions plus one live retried stream.
    * Request/session hooks, the ``User-Agent`` token, and ``stream_text_once``.

Run: ``uv sync --locked --extra images`` once, then ``python examples/13_llm_utilities.py``
"""

from __future__ import annotations

import asyncio
import base64
import io
import tempfile
from pathlib import Path

from _support.mock_llm_server import start_flaky_chat_server, start_hook_echo_server

from novakit.llm import Done, FakerLLMProvider, Message, OpenAIProvider
from novakit.llm.images import (
    DOWNSCALE_BYTES,
    IMAGE_REMOVED,
    bound_images,
    downscale_image,
    image_data_url,
    image_size_bytes,
    parse_image_ref,
)
from novakit.llm.oneshot import stream_text_once
from novakit.llm.policy import RetryBudget
from novakit.llm.provider import (
    RETRY_STATUS_CODES,
    RetryNotice,
    is_terminal_429,
)
from novakit.llm.reasoning import apply_effort, fit_effort, resolve_effort_levels
from novakit.llm.request_hook import (
    resolve_hook_path,
    run_request_hook,
    run_session_hook,
)
from novakit.llm.tokenizer import (
    count_cjk_chars,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens_by_type,
    normalise_model_id,
    resolve_context_limit,
    resolve_context_window,
)
from novakit.llm.user_agent import default_user_agent
from novakit.version import __version__


def show_tokenizer() -> None:
    """Token estimation, CJK handling, and context-window resolution."""
    print("--- tokenizer ---")
    text = "Hello world, this is a test."
    print("chars/4 estimate:", estimate_tokens_by_type(text))
    print(
        "tool-result estimate:", estimate_tokens_by_type("A" * 100, is_tool_result=True)
    )
    print("empty estimate:", estimate_tokens_by_type(""))
    print("cjk estimate:", estimate_tokens_by_type("中文测试"))
    print("cjk chars in 'ab中文cd':", count_cjk_chars("ab中文cd"))

    user_tokens = estimate_message_tokens(
        Message(role="user", content=text), model="unknown"
    )
    print("user message tokens:", user_tokens)
    print(
        "tool message tokens:",
        estimate_message_tokens(
            Message(role="tool", content="A" * 100), model="unknown"
        ),
    )
    print(
        "two-message total:",
        estimate_messages_tokens(
            [Message(role="user", content="hi"), Message(role="user", content=text)],
            model="unknown",
        ),
    )
    print("normalised model id:", normalise_model_id("meta/muse-spark-1.2"))
    window, source = resolve_context_window("gpt-4o", "openai")
    print(f"gpt-4o window: {window} (source={source})")
    print("gpt-4o limit:", resolve_context_limit("gpt-4o", provider="openai"))
    print("gpt-4 window:", resolve_context_window("gpt-4", "openai"))


def _make_photo_ref(edge: int = 1800) -> str:
    from PIL import Image

    noise = Image.effect_noise((edge, edge), 128).convert("RGB")
    buffer = io.BytesIO()
    noise.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def show_images() -> None:
    """Data-URL parsing, sizing, Pillow downscaling, and the request budget."""
    print("--- images ---")
    print("parse data URL:", parse_image_ref("data:image/jpeg;base64,QUJD"))
    print("parse bare payload:", parse_image_ref("QUJD"))
    print("build data URL:", image_data_url("QUJD"))
    print("payload bytes:", image_size_bytes("data:image/png;base64,QUJD"))

    photo = _make_photo_ref()
    print(
        "photo bytes:", image_size_bytes(photo), f"(downscale over {DOWNSCALE_BYTES})"
    )
    smaller = downscale_image(photo)
    print(
        "downscaled:",
        smaller[:30] + "...",
        "bytes:",
        image_size_bytes(smaller),
    )
    print("tiny passthrough:", downscale_image("data:image/png;base64,QUJD")[-4:])

    ten_bytes = "data:image/png;base64," + base64.b64encode(b"0123456789").decode()
    messages = [
        Message(role="user", content=f"turn {i}", images=[ten_bytes]) for i in range(4)
    ]
    bounded = bound_images(messages, trigger=25, target=10)
    print("bound images dropped:", [m.images is None for m in bounded])
    print("removal note mentions tool:", "tool" in IMAGE_REMOVED)
    print("original untouched:", messages[0].content)


def show_reasoning() -> None:
    """Effort levels: declaration, fitting, and per-provider wire spelling."""
    print("--- reasoning ---")
    config = {"reasoning_effort_levels": ["low", "medium", "high"]}
    print(
        "declared levels:",
        resolve_effort_levels("gpt-5.5", "openai-compatible", config),
    )
    print("no config:", resolve_effort_levels("gpt-5.5", "openai-compatible", {}))
    print("anthropic:", resolve_effort_levels("claude-opus-5", "anthropic", config))
    print("fit high:", fit_effort(["low", "high"], "high"))
    print("fit ultra:", fit_effort(["low", "high"], "ultra"))
    print("openai-compatible:", apply_effort({}, "high", "openai-compatible"))
    print("openai-response:", apply_effort({}, "high", "openai-response"))
    print("anthropic passthrough:", apply_effort({"a": 1}, "high", "anthropic"))


def show_retry_policy() -> None:
    """Retry budgets and the terminal-429 check, without touching the network."""
    print("--- retry policy ---")
    print(
        "500 retried:",
        500 in RETRY_STATUS_CODES,
        "| 429 in statuses:",
        429 in RETRY_STATUS_CODES,
    )
    print("429 slow-down terminal:", is_terminal_429(429, "slow down, rate limited"))
    print(
        "429 quota terminal:", is_terminal_429(429, "you have insufficient_quota left")
    )
    print("500 quota terminal:", is_terminal_429(500, "insufficient_quota"))

    budget = RetryBudget(
        use_retry=True,
        retry_429=True,
        honor_retry_after=False,
        statuses=RETRY_STATUS_CODES,
    )
    first = budget.connection_failure()
    print("connection failure backoff:", first.wait_seconds, "attempt:", first.attempt)
    fresh = RetryBudget(
        use_retry=True,
        retry_429=True,
        honor_retry_after=False,
        statuses=RETRY_STATUS_CODES,
    )
    decision = fresh.response_failure(503, retry_after=30.0, terminal_429=False)
    print("503 backoff:", decision.wait_seconds if decision else None)
    print(
        "terminal 429 decision:",
        budget.response_failure(429, retry_after=None, terminal_429=True),
    )


async def show_live_retry() -> None:
    """One real retried stream: the mock fails once with 503, then answers."""
    print("--- live retry ---")
    runner, base_url, calls = await start_flaky_chat_server()
    try:
        provider = OpenAIProvider(api_key="test-key", base_url=base_url)
        print(f"provider base_url={base_url} (first call returns 503)")
        events = [
            e
            async for e in provider.chat_stream(
                [{"role": "user", "content": "hi"}], model="m"
            )
        ]
        print("frames:", [type(e).__name__ for e in events])
        if isinstance(events[0], RetryNotice):
            print(f"retry notice: wait={events[0].wait_seconds}s")
        if isinstance(events[-1], Done):
            print(f"recovered text: {events[-1].content!r} after {calls['n']} calls")
    finally:
        await runner.cleanup()


async def show_request_hook(tmp: Path) -> None:
    """A hook script supplying per-session headers, observed through a local HTTP server."""
    print("--- request hook ---")
    hook = tmp / "headers_hook.py"
    hook.write_text(
        "import json, sys\n"
        "ctx = json.load(sys.stdin)\n"
        'print(json.dumps({"headers": {"x-demo-session": ctx.get("session_id", "")}}))\n',
        encoding="utf-8",
    )
    print("hook path resolved:", resolve_hook_path(str(hook)))
    print("empty hook path:", resolve_hook_path(""))
    print("direct hook run:", run_request_hook(str(hook), "sess-1"))
    print("session hook run:", run_session_hook(str(hook), "sess-1"))

    captured: dict[str, str] = {}
    runner, base_url = await start_hook_echo_server(captured)
    try:
        provider = OpenAIProvider(
            api_key="k",
            base_url=base_url,
            request_session_hook=str(hook),
        )
        result = await provider.chat(
            [{"role": "user", "content": "hi"}],
            model="m",
            session_id="sess-live",
        )
        if isinstance(result, Done):
            print(f"chat result: {result.content!r}")
        print("server saw header:", captured.get("x-demo-session"))
        print("server saw user-agent:", captured.get("user_agent"))
    finally:
        await runner.cleanup()


def show_user_agent() -> None:
    """The stable User-Agent token every provider sends."""
    print("--- user agent ---")
    agent = default_user_agent()
    print("default:", agent)
    print("sdk version:", __version__)
    print("custom:", default_user_agent("myapp", "9.9.9"))


async def show_oneshot() -> None:
    """One-shot streaming text calls, success plus the failure fallback."""
    print("--- oneshot ---")
    llm = FakerLLMProvider(seed=7)
    deltas: list[str] = []
    text = await stream_text_once(
        llm,
        messages=[Message(role="user", content="hello")],
        model="faker",
        on_delta=deltas.append,
        label="greeting",
    )
    print("text:", (text or "")[:60].replace("\n", " "))
    print("joined deltas:", "".join(deltas))

    failing = FakerLLMProvider(seed=7, error_probability=1.0)
    fallback = await stream_text_once(
        failing, messages=[Message(role="user", content="hi")], model="faker"
    )
    print("failing provider returns:", fallback)


async def main() -> None:
    """Run every utility demo as printed sections."""
    print("python:", "3.12+")
    show_tokenizer()
    show_images()
    show_reasoning()
    show_retry_policy()
    await show_live_retry()
    with tempfile.TemporaryDirectory() as tmpdir:
        await show_request_hook(Path(tmpdir))
    show_user_agent()
    await show_oneshot()
    print("LLM utility tour complete")


if __name__ == "__main__":
    asyncio.run(main())
