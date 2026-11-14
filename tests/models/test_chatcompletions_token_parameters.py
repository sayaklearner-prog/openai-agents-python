"""Tests for Chat Completions token parameter compatibility (max_tokens vs max_completion_tokens)."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from openai import AsyncOpenAI

from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner


def _create_mock_transport(captured_requests: list[dict[str, Any]], stream: bool) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/chat/completions"
        captured_requests.append(json.loads(request.content))
        if stream:
            chunk = {
                "id": "chatcmpl_test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
            }
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hello"},
                    }
                ],
            },
        )

    return httpx2.MockTransport(handler)


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("model_name", "expected_key"),
    [
        ("o1", "max_completion_tokens"),
        ("o1-mini", "max_completion_tokens"),
        ("o1-preview", "max_completion_tokens"),
        ("o3", "max_completion_tokens"),
        ("o3-mini", "max_completion_tokens"),
        ("gpt-5", "max_completion_tokens"),
        ("gpt-5-mini", "max_completion_tokens"),
        ("ft:o1-mini:org:proj:suffix", "max_completion_tokens"),
        ("gpt-4o", "max_tokens"),
        ("gpt-4o-mini", "max_tokens"),
        ("gpt-4", "max_tokens"),
    ],
)
async def test_official_openai_token_parameter_mapping(
    model_name: str,
    expected_key: str,
    stream: bool,
) -> None:
    captured: list[dict[str, Any]] = []
    transport = _create_mock_transport(captured, stream=stream)

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        http_client=httpx2.AsyncClient(transport=transport),
    ) as client:
        model = OpenAIChatCompletionsModel(model=model_name, openai_client=client)
        agent = Agent(name="test", model=model, model_settings=ModelSettings(max_tokens=256))

        if stream:
            result = Runner.run_streamed(agent, "Hi")
            async for _ in result.stream_events():
                pass
        else:
            await Runner.run(agent, "Hi")

    assert len(captured) == 1
    body = captured[0]
    assert body.get(expected_key) == 256
    unexpected_key = "max_tokens" if expected_key == "max_completion_tokens" else "max_completion_tokens"
    assert unexpected_key not in body


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "model_name",
    ["o1", "o3-mini", "gpt-5", "llama-3", "custom-model"],
)
async def test_third_party_endpoint_preserves_max_tokens(
    model_name: str,
    stream: bool,
) -> None:
    captured: list[dict[str, Any]] = []
    transport = _create_mock_transport(captured, stream=stream)

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="http://localhost:8000/v1",
        http_client=httpx2.AsyncClient(transport=transport),
    ) as client:
        model = OpenAIChatCompletionsModel(model=model_name, openai_client=client)
        agent = Agent(name="test", model=model, model_settings=ModelSettings(max_tokens=512))

        if stream:
            result = Runner.run_streamed(agent, "Hi")
            async for _ in result.stream_events():
                pass
        else:
            await Runner.run(agent, "Hi")

    assert len(captured) == 1
    body = captured[0]
    assert body.get("max_tokens") == 512
    assert "max_completion_tokens" not in body


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_extra_args_precedence_over_max_tokens(stream: bool) -> None:
    captured: list[dict[str, Any]] = []
    transport = _create_mock_transport(captured, stream=stream)

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        http_client=httpx2.AsyncClient(transport=transport),
    ) as client:
        model = OpenAIChatCompletionsModel(model="o3-mini", openai_client=client)
        agent = Agent(
            name="test",
            model=model,
            model_settings=ModelSettings(
                max_tokens=256,
                extra_args={"max_completion_tokens": 1024},
            ),
        )

        if stream:
            result = Runner.run_streamed(agent, "Hi")
            async for _ in result.stream_events():
                pass
        else:
            await Runner.run(agent, "Hi")

    assert len(captured) == 1
    body = captured[0]
    assert body.get("max_completion_tokens") == 1024
    assert "max_tokens" not in body


@pytest.mark.allow_call_model_methods
@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", ["o3-mini", "gpt-4o"])
async def test_duplicate_parameter_raises_type_error(model_name: str) -> None:
    captured: list[dict[str, Any]] = []
    transport = _create_mock_transport(captured, stream=False)

    async with AsyncOpenAI(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        http_client=httpx2.AsyncClient(transport=transport),
    ) as client:
        model = OpenAIChatCompletionsModel(model=model_name, openai_client=client)
        agent = Agent(
            name="test",
            model=model,
            model_settings=ModelSettings(
                max_tokens=256,
                extra_args={"max_tokens": 512},
            ),
        )

        with pytest.raises(
            TypeError,
            match="chat.completions.create\\(\\) got multiple values for keyword argument 'max_tokens'",
        ):
            await Runner.run(agent, "Hi")
