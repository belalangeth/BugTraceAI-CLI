"""LLM client shell mixin — Codex (ChatGPT login) wire format.

The `codex` provider authenticates with the OAuth tokens stored by the
official Codex CLI (`codex auth login`, ChatGPT account) and talks to the
ChatGPT backend's Responses API:

    POST https://chatgpt.com/backend-api/codex/responses

Wire format differs from OpenAI chat/completions:
- `input` array instead of `messages` (system prompt becomes top-level
  `instructions`)
- `max_output_tokens` instead of `max_tokens`
- `ChatGPT-Account-ID` header alongside `Authorization: Bearer`
- text is returned under `output[*].content[].text` (type `output_text`)
  instead of `choices[0].message.content`
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from bugtrace.utils.logger import get_logger

logger = get_logger("core.llm_client")


class LLMCodexMixin:
    """Responses API wire builders for the `codex` (ChatGPT login) provider.

    Routing is driven by the provider preset's `api_format: "responses"`.
    """

    async def _ensure_codex_token(self) -> Optional[str]:
        """Lazy-load and auto-refresh the Codex (ChatGPT) access token.

        The underlying get_valid_codex_token() already refreshes via
        auth.openai.com when the JWT is near expiry; we additionally cache
        the result for a few minutes to avoid re-reading the token file on
        every call.
        """
        now = time.time()
        if self._codex_token_cache and now < self._codex_token_expires:
            return self._codex_token_cache
        try:
            from bugtrace.core.codex_auth import get_valid_codex_token
            pair = await get_valid_codex_token()
            if pair:
                self._codex_token_cache, self._codex_account_id = pair
                self._codex_token_expires = now + 300  # Re-check every 5 min
                return self._codex_token_cache
            return None
        except Exception as e:
            logger.error(f"Codex token load failed: {e}")
            return None

    def _build_codex_headers(self, module_name: str) -> Dict[str, str]:
        """Build headers for the ChatGPT backend Codex Responses API."""
        headers = {
            "Authorization": f"Bearer {self._codex_token_cache or ''}",
            "Content-Type": "application/json",
            "User-Agent": "bugtraceai-cli",
        }
        if getattr(self, "_codex_account_id", None):
            headers["ChatGPT-Account-ID"] = self._codex_account_id
        return headers

    async def _maybe_refresh_codex_models(self) -> None:
        """Refresh the active model slots from the ChatGPT backend /models endpoint.

        Only acts when the codex provider is active and auto-discovery is
        enabled; never raises and never blocks a scan — on any failure the
        static preset's models are kept unchanged.
        """
        from bugtrace.core.config import settings

        if not getattr(settings, "CODEX_MODEL_AUTODISCOVERY", True):
            return
        if self.provider_id != "codex" or self.api_format != "responses":
            return
        try:
            from bugtrace.core.codex_auth import fetch_codex_models
            from bugtrace.core.codex_models import (
                apply_assignments,
                plan_codex_assignments,
            )

            slugs = await fetch_codex_models()
            if not slugs:
                logger.info("[Codex] Model discovery returned nothing — keeping preset models")
                return
            assignments = plan_codex_assignments(
                slugs,
                preferred_heavy=getattr(settings, "DEFAULT_MODEL", "") or "",
                preferred_light=getattr(settings, "REPORTING_MODEL", "") or "",
            )
            if not assignments:
                return
            if apply_assignments(assignments, settings):
                # Sync the client's model-shifting pool with the new PRIMARY_MODELS.
                pool = assignments.get("PRIMARY_MODELS", "")
                if pool:
                    self.models = [m.strip() for m in pool.split(",") if m.strip()]
                logger.info(
                    f"[Codex] Auto-discovered {len(slugs)} model(s) → primary pool: {self.models}"
                )
        except Exception as e:
            logger.warning(f"[Codex] Model discovery failed (keeping preset models): {e}")

    def _build_codex_payload(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Build a Responses API payload for the ChatGPT backend.

        Key differences from the OpenAI chat/completions format:
        - system message becomes the top-level `instructions` string
        - remaining messages go into the `input` array
        - `stream` MUST be true: the chatgpt.com/backend-api/codex/responses
          endpoint rejects non-streaming requests with HTTP 400
        - no token-cap parameter is accepted at all: the endpoint rejects
          max_output_tokens / max_tokens / max_completion_tokens with HTTP 400
          ("Unsupported parameter") — the backend applies its own default
        - `temperature` is omitted because the backend applies its own default
          for reasoning models and rejects explicit temperatures on some of them
        """
        instructions = None
        input_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                instructions = msg.get("content")
            else:
                input_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        payload: Dict[str, Any] = {
            "model": model,
            "input": input_messages,
            "store": False,
            "stream": True,
        }
        if instructions:
            payload["instructions"] = instructions
        return payload

    async def _consume_codex_sse(self, resp, module_name: str = "") -> Tuple[str, Dict[str, int]]:
        """Read a Responses-API SSE stream into (text, usage).

        The ChatGPT backend streams `response.output_text.delta` events and
        finishes with a `response.completed` event carrying the usage.
        ``resp`` must already have a 200 status.
        """
        text_parts: List[str] = []
        usage: Dict[str, int] = {}
        try:
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "response.output_text.delta":
                    text_parts.append(event.get("delta") or "")
                elif etype == "response.completed":
                    response = event.get("response") or {}
                    u = response.get("usage") or {}
                    usage["input_tokens"] = int(u.get("input_tokens", 0) or 0)
                    usage["output_tokens"] = int(u.get("output_tokens", 0) or 0)
        except Exception as e:
            logger.warning(f"[Codex] SSE read error for {module_name}: {e}")
        return "".join(text_parts), usage

    async def _handle_codex_response_common(
        self,
        text: str,
        usage: Dict[str, int],
        current_model: str,
        module_name: str,
        prompt: str,
        model_override: Optional[str],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        latency_ms: float,
    ) -> Optional[str]:
        """Shared post-stream processing for codex completions: refusal check,
        telemetry and audit, mirroring the non-streaming _handle_api_response.
        """
        if not text:
            self._record_model_call(current_model, success=False, latency_ms=latency_ms)
            logger.warning(f"Codex API: {current_model} returned no text content.")
            return None
        data = {
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            }
        }
        result = await self._handle_refusal(
            text, current_model, model_override, prompt, module_name, system_prompt, temperature, max_tokens
        )
        if result != text:
            return result
        self._record_model_call(current_model, success=True, latency_ms=latency_ms)
        await self._update_telemetry(data, current_model, module_name)
        await self._audit_log(module_name, current_model, prompt, text)
        logger.info(f"Codex API: Using {current_model} for {module_name}")
        return text