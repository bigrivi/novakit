"""Local mock LLM wire servers for examples 12 and 13.

Teaching stub only: speaks just enough of each provider protocol
(OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Ollama)
plus the small flaky/echo servers used by the retry and request-hook
sections. Runs on 127.0.0.1 with an ephemeral port; no external network.
"""

from __future__ import annotations

import json
from typing import Any

from aiohttp import web

HOST = "127.0.0.1"
BAD_MODEL = "bad-model"


def _bad() -> web.Response:
    return web.json_response(
        {"error": {"message": "invalid model", "type": "invalid_request_error"}},
        status=401,
    )


async def _chat_completions(request: web.Request) -> web.Response:
    body: dict[str, Any] = await request.json()
    request.app["seen"]["chat"] = {
        "body": body,
        "auth": request.headers.get("Authorization", ""),
    }
    if body.get("model") == BAD_MODEL:
        return _bad()
    if body.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        model = body.get("model", "")
        if model == "tool-model":
            events: list[dict[str, Any]] = [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '"Oslo"}'}}
                                ]
                            }
                        }
                    ]
                },
            ]
        elif model == "think-model":
            events = [
                {"choices": [{"delta": {"reasoning_content": "thinking it over..."}}]},
                {"choices": [{"delta": {"content": "Hello world"}}]},
            ]
        else:
            events = [
                {"choices": [{"delta": {"content": "Hello "}}]},
                {"choices": [{"delta": {"content": "world"}}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 4,
                        "prompt_tokens_details": {"cached_tokens": 3},
                    },
                },
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
        for event in events:
            await resp.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp
    return web.json_response(
        {
            "choices": [{"message": {"content": "Hello world", "tool_calls": []}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        }
    )


async def _responses(request: web.Request) -> web.Response:
    body = await request.json()
    request.app["seen"]["responses"] = {
        "body": body,
        "user_agent": request.headers.get("User-Agent", ""),
    }
    if body.get("model") == BAD_MODEL:
        return _bad()
    if body.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for event in (
            {"type": "response.output_text.delta", "delta": "Hello "},
            {"type": "response.output_text.delta", "delta": "world"},
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 9,
                        "output_tokens": 4,
                        "input_tokens_details": {"cached_tokens": 1},
                    }
                },
            },
        ):
            await resp.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        await resp.write_eof()
        return resp
    return web.json_response(
        {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hello world"}],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 2},
            },
        }
    )


async def _anthropic_messages(request: web.Request) -> web.Response:
    body = await request.json()
    request.app["seen"]["anthropic"] = {"body": body}
    if body.get("model") == BAD_MODEL:
        return _bad()
    if body.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for kind, event in (
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "usage": {
                            "input_tokens": 8,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 1,
                            "cache_creation_input_tokens": 1,
                        }
                    },
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello world"},
                },
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 6},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ):
            await resp.write(f"event: {kind}\n".encode())
            await resp.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        await resp.write_eof()
        return resp
    return web.json_response(
        {
            "content": [{"type": "text", "text": "Hello world"}],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 5,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 1,
            },
        }
    )


async def _ollama_chat(request: web.Request) -> web.Response:
    body = await request.json()
    request.app["seen"]["ollama"] = {"body": body}
    if body.get("model") == BAD_MODEL:
        return _bad()
    if body.get("stream"):
        resp = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
        await resp.prepare(request)
        for event in (
            {
                "message": {"content": "", "reasoning_content": "let me think"},
                "done": False,
            },
            {"message": {"content": "Hello "}, "done": False},
            {"message": {"content": "world"}, "done": True},
        ):
            await resp.write(json.dumps(event).encode() + b"\n")
        await resp.write_eof()
        return resp
    return web.json_response({"message": {"content": "Hello world"}, "done": True})


async def start_mock_llm_server() -> tuple[web.AppRunner, str, dict[str, Any]]:
    """Start the four-protocol mock server; return (runner, base_url, seen)."""
    seen: dict[str, Any] = {}
    app = web.Application()
    app["seen"] = seen
    app.router.add_post("/chat/completions", _chat_completions)
    app.router.add_post("/responses", _responses)
    app.router.add_post("/v1/messages", _anthropic_messages)
    app.router.add_post("/api/chat", _ollama_chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, 0)
    await site.start()
    port = runner.addresses[0][1]
    return runner, f"http://{HOST}:{port}", seen


async def start_flaky_chat_server() -> tuple[web.AppRunner, str, dict[str, int]]:
    """Fail the first chat-completions call with 503, then stream an answer."""
    calls = {"n": 0}

    async def handler(request: web.Request) -> web.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return web.Response(status=503, text="busy, try again")
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(
            b'data: {"choices": [{"delta": {"content": "recovered"}}]}\n\n'
        )
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, 0)
    await site.start()
    return runner, f"http://{HOST}:{runner.addresses[0][1]}", calls


async def start_hook_echo_server(
    captured: dict[str, str],
) -> tuple[web.AppRunner, str]:
    """Echo chat completions while capturing the hook header and User-Agent."""

    async def handler(request: web.Request) -> web.Response:
        captured["x-demo-session"] = request.headers.get("x-demo-session", "")
        captured["user_agent"] = request.headers.get("User-Agent", "")
        return web.json_response({"choices": [{"message": {"content": "ok"}}]})

    app = web.Application()
    app.router.add_post("/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, 0)
    await site.start()
    return runner, f"http://{HOST}:{runner.addresses[0][1]}"
