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

import time
from typing import Any, Dict, List, Optional

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
        """Build a non-streaming Responses API payload.

        Key differences from the OpenAI chat/completions format:
        - system message becomes the top-level `instructions` string
        - remaining messages go into the `input` array
        - `max_output_tokens` (not `max_tokens`); `temperature` is omitted
          because the ChatGPT backend applies its own default for reasoning
          models and rejects explicit temperatures on some of them
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
            "max_output_tokens": max_tokens,
        }
        if instructions:
            payload["instructions"] = instructions
        return payload