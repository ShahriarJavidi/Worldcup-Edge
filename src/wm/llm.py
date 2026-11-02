"""Thin wrapper around the Anthropic SDK used by every LLM step in the project."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import anthropic

from wm.config import Settings

# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """The model did not return a usable answer."""


@dataclass(frozen=True)
class Completion:
    """One response's text plus what it cost to get it."""

    text: str
    model: str  # the model that answered; differs from the request after a server-side fallback
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    latency_s: float


def parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"The model returned invalid JSON: {text[:200]!r}") from exc


class Claude:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        self.settings = settings
        # A bare Anthropic() also resolves `ant auth login` profiles when no key is set.
        self.client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
    ) -> str:
        """Run one request and return the response text (valid JSON when ``schema`` is given)."""
        return self.run(system=system, prompt=prompt, schema=schema, max_tokens=max_tokens).text

    def run(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 16000,
    ) -> Completion:
        """Like :meth:`complete`, but also returns token usage and wall-clock latency."""
        output_config: dict[str, Any] = {"effort": self.settings.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}

        kwargs: dict[str, Any] = {}
        if self.settings.model in _FALLBACK_MODELS:
            kwargs = {"betas": [_FALLBACK_BETA], "extra_body": {"fallbacks": "default"}}

        started = time.perf_counter()
        try:
            response = self.client.beta.messages.create(
                model=self.settings.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config=output_config,
                **kwargs,
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic authentication failed. Check ANTHROPIC_API_KEY in .env.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API.") from exc
        latency = time.perf_counter() - started

        if response.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        if response.stop_reason == "max_tokens":
            raise LLMError("The response was cut off (max_tokens reached).")

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMError("The model returned an empty response.")
        usage = response.usage
        return Completion(
            text=text,
            model=str(getattr(response, "model", None) or self.settings.model),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            latency_s=latency,
        )

    def complete_json(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        return parse_json(self.complete(system=system, prompt=prompt, schema=schema))
