"""Thin wrapper around the Anthropic SDK used by every LLM step in the project."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from wm.config import Settings

# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """The model did not return a usable answer."""


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
        output_config: dict[str, Any] = {"effort": self.settings.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}

        kwargs: dict[str, Any] = {}
        if self.settings.model in _FALLBACK_MODELS:
            kwargs = {"betas": [_FALLBACK_BETA], "extra_body": {"fallbacks": "default"}}

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

        if response.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        if response.stop_reason == "max_tokens":
            raise LLMError("The response was cut off (max_tokens reached).")

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise LLMError("The model returned an empty response.")
        return text

    def complete_json(self, *, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        text = self.complete(system=system, prompt=prompt, schema=schema)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"The model returned invalid JSON: {text[:200]!r}") from exc
