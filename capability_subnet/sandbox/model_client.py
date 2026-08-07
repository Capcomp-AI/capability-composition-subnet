"""The candidate model endpoint.

Every candidate is served behind the same OpenAI-compatible chat-completions API,
whether that is vLLM, SGLang or the in-process reference client used for local
development. The agent loop therefore has exactly one way to talk to a model, and
swapping the serving runtime cannot change what a candidate is asked or how its
reply is parsed.

Sampling is greedy and seeded per instance. Two runs of the same package on the
same instance produce the same conversation, which is what makes a re-run a
verification rather than a new experiment.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from capability_subnet.common import constants as C

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelMessage:
    """One message in the conversation."""

    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_api(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(slots=True)
class ModelReply:
    """One assistant turn."""

    content: str | None
    tool_calls: list[dict[str, Any]]
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class ModelClient(Protocol):
    """Anything the agent loop can talk to."""

    def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
        *,
        seed: int,
        max_tokens: int,
    ) -> ModelReply: ...


class OpenAICompatibleClient:
    """Talks to a served candidate over the chat-completions API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "not-required",
        timeout: float = 120.0,
        enable_thinking: bool = C.SANDBOX_ENABLE_THINKING,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.enable_thinking = enable_thinking

    def complete(
        self,
        messages: list[ModelMessage],
        tools: list[dict[str, Any]],
        *,
        seed: int,
        max_tokens: int,
    ) -> ModelReply:
        import httpx

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [message.to_api() for message in messages],
            "temperature": C.SANDBOX_TEMPERATURE,
            "top_p": C.SANDBOX_TOP_P,
            "seed": seed,
            "max_tokens": max_tokens,
            # Qwen3's chat template defaults to thinking mode, which opens every
            # reply with a <think> block. At this token budget a single such
            # block consumes the whole allowance before the model has called one
            # tool, so every instance would fail on a template default rather
            # than on the merged package. Turned off explicitly, and part of the
            # published contract because it changes what candidates are asked.
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

        # Only sent when there is something to choose from. An OpenAI-compatible
        # server rejects `tool_choice: "auto"` alongside an empty tool list with
        # a 400, and the general-capability probe deliberately offers no tools —
        # so sending it unconditionally made every probe request fail, and the
        # retention gate would have read that as a candidate scoring zero.
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        try:
            response = httpx.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            # A serving failure is infrastructure, not a model failure. Returning
            # it as an error lets the orchestrator mark the whole trace as a
            # harness failure and exclude it, instead of scoring a zero the
            # candidate did not earn.
            #
            # A 4xx carries the server's reason in the body, and without it the
            # log says only "400 Bad Request" — which is indistinguishable
            # between a prompt longer than the context window, an unsupported
            # sampling parameter and a malformed tool schema. Each is a different
            # fix and the operator cannot tell which they have.
            detail = ""
            reason = getattr(exc, "response", None)
            if reason is not None:
                try:
                    body = reason.json()
                    detail = str(body.get("message") or body.get("error") or body)[:400]
                except Exception:  # noqa: BLE001 - a non-JSON body is still worth quoting
                    detail = (reason.text or "")[:400]
            log.warning("model endpoint call failed: %s%s", exc, f" — {detail}" if detail else "")
            return ModelReply(
                None,
                [],
                error=f"model endpoint unavailable: {exc}{f' — {detail}' if detail else ''}",
            )

        try:
            choice = payload["choices"][0]
            message = choice["message"]
            usage = payload.get("usage", {}) or {}
            return ModelReply(
                content=message.get("content"),
                tool_calls=list(message.get("tool_calls") or []),
                input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage.get("completion_tokens", 0) or 0),
                finish_reason=choice.get("finish_reason", "stop"),
            )
        except (KeyError, IndexError, TypeError) as exc:
            log.warning("model endpoint returned an unexpected shape: %s", exc)
            return ModelReply(None, [], error=f"malformed completion response: {exc}")


def parse_tool_call(raw: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    """Extract ``(name, arguments, error)`` from one tool call.

    Arguments arrive as a JSON string. A model that emits invalid JSON gets a
    parse error back as a normal tool result and can retry — recovering from its
    own malformed call is part of what the tool-calling capability means.
    """
    function = raw.get("function") or {}
    name = str(function.get("name") or "")

    raw_arguments = function.get("arguments")
    if raw_arguments in (None, ""):
        return name, {}, None
    if isinstance(raw_arguments, dict):
        return name, raw_arguments, None

    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return name, {}, f"Argumente sind kein gültiges JSON: {exc}"

    if not isinstance(parsed, dict):
        return name, {}, "Argumente müssen ein JSON-Objekt sein"
    return name, parsed, None
