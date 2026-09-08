"""
Provider Management Endpoints — View and switch LLM providers.

Provides GET /providers, GET /provider, PUT /provider, PATCH /provider/models
for runtime provider management and API key configuration.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from bugtrace.core.config import settings
from bugtrace.utils.logger import get_logger
from bugtrace.utils.env_writer import update_env_var

logger = get_logger("api.routes.providers")

router = APIRouter(tags=["providers"])

PROVIDERS_DIR = settings.BASE_DIR / "bugtrace" / "data" / "providers"


# ──── Response/Request Models ────


class ProviderSummary(BaseModel):
    id: str
    name: str
    recommended: bool = False
    api_key_configured: bool
    api_key_hint: str


class ProviderDetail(BaseModel):
    provider: str
    name: str
    base_url: str
    api_key_configured: bool
    api_key_hint: str
    features: Dict[str, Any]
    models: Dict[str, str]
    pricing: Dict[str, Any]


class TestProviderRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None


class SwitchProviderRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None


class ModelOverrideRequest(BaseModel):
    """Override individual model assignments at runtime."""
    DEFAULT_MODEL: Optional[str] = None
    CODE_MODEL: Optional[str] = None
    ANALYSIS_MODEL: Optional[str] = None
    PRIMARY_MODELS: Optional[str] = None
    VISION_MODEL: Optional[str] = None
    WAF_DETECTION_MODELS: Optional[str] = None
    MUTATION_MODEL: Optional[str] = None
    SKEPTICAL_MODEL: Optional[str] = None
    REPORTING_MODEL: Optional[str] = None


class CodexModelsRequest(BaseModel):
    """Manual model picks for the codex (ChatGPT login) provider.

    Omitted fields fall back to auto-discovery defaults (the newest served
    model per tier). Both values must be slugs the backend currently serves.
    reasoning_effort / reasoning_effort_main / reasoning_effort_fast
    (low/medium/high/xhigh/max) are sent to the ChatGPT backend on every
    codex call; empty string clears that slot (backend default).
    """
    main_model: Optional[str] = None
    light_model: Optional[str] = None
    reasoning_effort: Optional[str] = None  # legacy global — applied to both slots
    reasoning_effort_main: Optional[str] = None
    reasoning_effort_fast: Optional[str] = None


# Valid reasoning effort values the ChatGPT backend accepts (probed live).
# `max` is rejected by some models (e.g. gpt-5.4-mini) but accepted by the
# heavy ones; keep it selectable and let the backend enforce per-model.
_VALID_REASONING_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


# ──── Helpers ────


def _load_preset(provider_id: str) -> Dict[str, Any]:
    """Load a provider preset JSON file."""
    path = PROVIDERS_DIR / f"{provider_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Provider '{provider_id}' not found")
    return json.loads(path.read_text())


def _mask_api_key(key: Optional[str]) -> str:
    """Mask API key for display: show first 8 + last 4 chars."""
    if not key:
        return ""
    if len(key) > 12:
        return key[:8] + "..." + key[-4:]
    return "***"


def _check_api_key(preset: Dict[str, Any]) -> tuple:
    """Check if provider's API key is configured. Returns (configured, masked_hint)."""
    # The codex provider has no API key — it authenticates via the Codex CLI
    # login (~/.codex/auth.json). Report configured when that login exists.
    if preset.get("api_format") == "responses" or preset.get("id") == "codex":
        from bugtrace.core.codex_auth import codex_auth_available
        return codex_auth_available(), ""
    key_env = preset.get("api_key_env", "")
    key_value = os.environ.get(key_env) or getattr(settings, key_env, None)
    return bool(key_value), _mask_api_key(key_value)


# ──── Endpoints ────


@router.get("/providers", response_model=List[ProviderSummary])
async def list_providers():
    """List all available LLM providers with API key status."""
    providers = []
    if not PROVIDERS_DIR.exists():
        return providers

    for path in sorted(PROVIDERS_DIR.glob("*.json")):
        try:
            preset = json.loads(path.read_text())
            configured, hint = _check_api_key(preset)
            providers.append(ProviderSummary(
                id=preset["id"],
                name=preset["name"],
                recommended=preset.get("recommended", False),
                api_key_configured=configured,
                api_key_hint=hint if configured else preset.get("api_key_hint", ""),
            ))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Skipping invalid provider preset {path.name}: {e}")

    return providers


@router.get("/provider", response_model=ProviderDetail)
async def get_current_provider():
    """Get the currently active provider configuration."""
    provider_cfg = getattr(settings, '_provider_config', {})
    if not provider_cfg:
        # Fallback: load from file
        try:
            provider_cfg = _load_preset(settings.PROVIDER)
        except HTTPException:
            raise HTTPException(status_code=404, detail=f"Active provider '{settings.PROVIDER}' preset not found")

    configured, hint = _check_api_key(provider_cfg)

    # Return current runtime model assignments (may differ from preset defaults)
    current_models = {}
    for field in ["DEFAULT_MODEL", "CODE_MODEL", "ANALYSIS_MODEL", "PRIMARY_MODELS",
                  "VISION_MODEL", "WAF_DETECTION_MODELS", "MUTATION_MODEL",
                  "SKEPTICAL_MODEL", "REPORTING_MODEL"]:
        current_models[field] = getattr(settings, field, "")

    return ProviderDetail(
        provider=settings.PROVIDER,
        name=provider_cfg.get("name", settings.PROVIDER),
        base_url=provider_cfg.get("base_url", ""),
        api_key_configured=configured,
        api_key_hint=hint if configured else provider_cfg.get("api_key_hint", ""),
        features=provider_cfg.get("features", {}),
        models=current_models,
        pricing=provider_cfg.get("pricing", {}),
    )


@router.get("/providers/{provider_id}", response_model=ProviderDetail)
async def get_provider_detail(provider_id: str):
    """Get the full preset configuration for any provider by ID."""
    preset = _load_preset(provider_id)
    configured, hint = _check_api_key(preset)
    return ProviderDetail(
        provider=preset["id"],
        name=preset["name"],
        base_url=preset.get("base_url", ""),
        api_key_configured=configured,
        api_key_hint=hint if configured else preset.get("api_key_hint", ""),
        features=preset.get("features", {}),
        models=preset.get("models", {}),
        pricing=preset.get("pricing", {}),
    )


@router.put("/provider")
async def switch_provider(req: SwitchProviderRequest):
    """Switch the active LLM provider and optionally set its API key.

    This updates the runtime configuration. The change persists until
    the server restarts (use bugtraceaicli.conf for permanent changes).
    """
    # Validate provider exists
    preset = _load_preset(req.provider)

    # If API key provided, write to .env
    if req.api_key:
        key_env = preset.get("api_key_env", "")
        if not key_env:
            raise HTTPException(status_code=400, detail="Provider has no api_key_env defined")
        if not update_env_var(key_env, req.api_key):
            raise HTTPException(status_code=500, detail="Failed to write API key to .env")
        # Also set on settings object for immediate use
        if hasattr(settings, key_env):
            object.__setattr__(settings, key_env, req.api_key)
        os.environ[key_env] = req.api_key

    # Switch provider
    object.__setattr__(settings, 'PROVIDER', req.provider)

    # Reload preset (applies model defaults)
    settings._load_provider_preset()

    # Reinitialize LLM client with new provider. Delegate to the client's own
    # consolidated reconfigure so EVERY provider-scoped attribute moves together
    # (api_format, base_url, models, headers, concurrency, failover, rate limiters)
    # — a partial hot-reload here previously left api_format stale, so an Anthropic
    # provider kept sending OpenAI-format payloads until a full restart.
    try:
        from bugtrace.core.llm_client import llm_client
        llm_client.reconfigure_from_active_preset()
        logger.info(f"LLM client reinitialized for provider: {req.provider}")
        # Codex model names drift as the ChatGPT backend evolves — refresh the
        # slots from /models right after switching so the scan starts on models
        # the backend actually serves.
        if preset.get("api_format") == "responses" or req.provider == "codex":
            try:
                await llm_client._maybe_refresh_codex_models()
            except Exception as e:
                logger.warning(f"Codex model auto-discovery after switch failed: {e}")
    except ImportError:
        logger.warning("Could not reimport llm_client for hot-reload")

    configured, hint = _check_api_key(preset)
    # Models may have been remapped by auto-discovery — report live values.
    models_map = preset.get("models", {}).keys()
    return {
        "message": f"Switched to provider: {preset['name']}",
        "provider": req.provider,
        "api_key_configured": configured,
        "api_key_hint": hint,
        "models": {k: getattr(settings, k, "") for k in models_map},
    }


@router.post("/provider/test")
async def test_provider_key(req: TestProviderRequest):
    """Test an API key against a provider by making a real LLM call."""
    import httpx

    preset = _load_preset(req.provider)
    base_url = preset.get("base_url", "")
    if not base_url:
        raise HTTPException(status_code=400, detail="Provider has no base_url configured")

    # Use provided key, or fall back to configured key
    api_format = preset.get("api_format", "openai")
    is_codex = (api_format == "responses")
    api_key = req.api_key
    if not api_key and not is_codex:
        key_env = preset.get("api_key_env", "")
        api_key = os.environ.get(key_env) or getattr(settings, key_env, None)
    if not api_key and not is_codex:
        return {"success": False, "message": "No API key provided and none configured."}

    # Pick the fastest/cheapest model from preset for testing. The codex
    # provider's model slots are auto-discovered from the ChatGPT backend at
    # runtime, so prefer the live settings value over the static preset file
    # (which may name a model the backend no longer serves).
    models = preset.get("models", {})
    test_model = models.get("ANALYSIS_MODEL") or models.get("DEFAULT_MODEL") or ""
    if is_codex:
        live_model = getattr(settings, "ANALYSIS_MODEL", "") or getattr(settings, "DEFAULT_MODEL", "")
        if live_model:
            test_model = live_model
    if not test_model:
        return {"success": False, "message": "No model configured for this provider."}

    # Build headers + body in the provider's wire format. Anthropic uses the
    # Messages API (x-api-key, model without the anthropic/ prefix); codex uses
    # the ChatGPT backend Responses API with the Codex CLI OAuth token; everyone
    # else uses the OpenAI-compatible /chat/completions shape (Authorization: Bearer).
    if api_format == "responses":
        from bugtrace.core.codex_auth import get_valid_codex_token
        codex_pair = await get_valid_codex_token()
        if not codex_pair:
            return {"success": False, "message": "No valid Codex login found. Run `codex auth login` (Sign in with ChatGPT) first."}
        token, account_id = codex_pair
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        if account_id:
            headers["ChatGPT-Account-ID"] = account_id
        body = {
            "model": test_model,
            "input": [{"role": "user", "content": "Are you alive? Answer only yes."}],
            "store": False,
            "stream": True,  # backend rejects non-streaming AND token-cap params
        }
    elif api_format == "anthropic":
        headers: Dict[str, str] = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": test_model.replace("anthropic/", "", 1),
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Are you alive? Answer only yes."}],
        }
    else:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept-Language": "en-US,en",
        }
        # Apply provider-specific headers
        for k, v in preset.get("headers", {}).items():
            headers[k] = v
        body = {
            "model": test_model,
            "messages": [{"role": "user", "content": "Are you alive? Answer only yes."}],
        }
        if req.provider == "openai" and test_model.startswith("gpt-5"):
            body["max_completion_tokens"] = 5
        else:
            body["max_tokens"] = 5

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(base_url, json=body, headers=headers)
        if resp.status_code == 200:
            return {"success": True, "message": "API Key Valid Response"}
        elif resp.status_code == 401:
            return {"success": False, "message": "Invalid API key. Please check and try again."}
        elif resp.status_code == 403:
            return {"success": False, "message": "API key rejected — insufficient permissions or account issue."}
        elif resp.status_code == 429:
            return {"success": False, "message": "Rate limited — key is valid but too many requests. Try again later."}
        else:
            detail = ""
            try:
                data = resp.json()
                detail = data.get("error", {}).get("message", "") if isinstance(data.get("error"), dict) else str(data.get("error", ""))
            except Exception:
                pass
            return {"success": False, "message": f"Provider returned HTTP {resp.status_code}. {detail}".strip()}
    except httpx.TimeoutException:
        return {"success": False, "message": "Connection timed out. Check the provider URL."}
    except Exception as e:
        return {"success": False, "message": f"Connection failed: {str(e)}"}


@router.patch("/provider/models")
async def override_models(req: ModelOverrideRequest):
    """Override individual model assignments at runtime."""
    updated = {}
    for field, value in req.model_dump(exclude_none=True).items():
        if hasattr(settings, field):
            old = getattr(settings, field)
            object.__setattr__(settings, field, value)
            updated[field] = {"from": old, "to": value}
            logger.info(f"Model override: {field} = {value}")

    if not updated:
        raise HTTPException(status_code=400, detail="No valid model fields provided")

    # Update LLM client model list if PRIMARY_MODELS changed
    if "PRIMARY_MODELS" in updated:
        try:
            from bugtrace.core.llm_client import llm_client
            llm_client.models = [m.strip() for m in settings.PRIMARY_MODELS.split(",")]
        except ImportError:
            pass

    return {"updated": updated, "message": f"Updated {len(updated)} model assignment(s)"}


# ──── Codex (ChatGPT login) model discovery ────


def _sync_codex_client_models(assignments: Dict[str, str]) -> None:
    """Point the live LLM client's model-shifting pool at the new PRIMARY_MODELS."""
    try:
        from bugtrace.core.llm_client import llm_client
        pool = [m.strip() for m in (assignments.get("PRIMARY_MODELS") or "").split(",") if m.strip()]
        if pool:
            llm_client.models = pool
    except Exception:
        logger.debug("Codex model sync skipped (llm_client unavailable)")


def _codex_assignment_message() -> str:
    """Shared hint for when the Codex CLI login is missing."""
    return "No Codex login found. Run `codex auth login` (Sign in with ChatGPT) on the CLI host, then reload."


@router.get("/provider/codex/models")
async def get_codex_models():
    """Return the models the ChatGPT backend serves for the Codex login.

    Read-only: never writes. ``assignments`` is a preview of what
    auto-discovery would apply (respecting currently pinned slugs).
    """
    from bugtrace.core.codex_auth import codex_auth_available, fetch_codex_models
    from bugtrace.core.codex_models import plan_codex_assignments

    if not codex_auth_available():
        return {"success": False, "configured": False, "message": _codex_assignment_message(), "models": [], "assignments": {}}
    slugs = await fetch_codex_models()
    if not slugs:
        return {"success": False, "configured": True, "message": "Could not reach the ChatGPT model list — keeping the preset models.", "models": [], "assignments": {}}
    assignments = plan_codex_assignments(
        slugs,
        preferred_heavy=getattr(settings, "DEFAULT_MODEL", "") or "",
        preferred_light=getattr(settings, "REPORTING_MODEL", "") or "",
    )
    return {
        "success": True,
        "configured": True,
        "models": slugs,
        "assignments": assignments,
        "reasoning_effort": (getattr(settings, "CODEX_REASONING_EFFORT", "") or ""),
        "reasoning_effort_main": (getattr(settings, "CODEX_REASONING_EFFORT_MAIN", "") or ""),
        "reasoning_effort_fast": (getattr(settings, "CODEX_REASONING_EFFORT_FAST", "") or ""),
    }


@router.post("/provider/codex/models")
async def refresh_codex_models(req: Optional[CodexModelsRequest] = None):
    """Fetch the served model list and (re)apply slot assignments.

    Empty body = refresh with auto-discovery defaults. A body with
    main_model/light_model pins those exact slugs (validated against the
    discovered list) into the heavy/light slots. Also updates the live LLM
    client so the next scan uses the new pool.
    """
    from bugtrace.core.codex_auth import codex_auth_available, fetch_codex_models
    from bugtrace.core.codex_models import apply_assignments, plan_codex_assignments

    req = req or CodexModelsRequest()
    if not codex_auth_available():
        return {"success": False, "configured": False, "message": _codex_assignment_message(), "models": [], "assignments": {}}
    slugs = await fetch_codex_models()
    if not slugs:
        return {"success": False, "configured": True, "message": "Could not reach the ChatGPT model list — no changes applied.", "models": [], "assignments": {}}

    # Explicit picks must be slugs the backend actually serves right now.
    for pick in (req.main_model, req.light_model):
        if pick and pick not in slugs:
            return {"success": False, "configured": True, "message": f"'{pick}' is not in the discovered model list.", "models": slugs, "assignments": {}}

    # Persist reasoning effort (runtime + .env) when provided.
    # legacy reasoning_effort applies to both slots (keeps the old WEB/API
    # clients working); per-slot fields override their side only.
    for field, env_key, attr in (
        (req.reasoning_effort, "CODEX_REASONING_EFFORT", "CODEX_REASONING_EFFORT"),
        (req.reasoning_effort_main, "CODEX_REASONING_EFFORT_MAIN", "CODEX_REASONING_EFFORT_MAIN"),
        (req.reasoning_effort_fast, "CODEX_REASONING_EFFORT_FAST", "CODEX_REASONING_EFFORT_FAST"),
    ):
        if field is None:
            continue
        effort = field.strip().lower()
        if effort and effort not in _VALID_REASONING_EFFORTS:
            return {
                "success": False,
                "configured": True,
                "message": f"Invalid reasoning effort '{field}'. Valid: {sorted(_VALID_REASONING_EFFORTS)}.",
                "models": slugs,
                "assignments": {},
            }
        update_env_var(env_key, effort)
        object.__setattr__(settings, attr, effort)

    assignments = plan_codex_assignments(
        slugs,
        preferred_heavy=req.main_model or "",
        preferred_light=req.light_model or "",
    )
    if not assignments:
        return {"success": False, "configured": True, "message": "No usable models returned by the backend.", "models": [], "assignments": {}}

    apply_assignments(assignments, settings)
    _sync_codex_client_models(assignments)
    return {
        "success": True,
        "configured": True,
        "models": slugs,
        "assignments": assignments,
        "reasoning_effort": (getattr(settings, "CODEX_REASONING_EFFORT", "") or ""),
        "reasoning_effort_main": (getattr(settings, "CODEX_REASONING_EFFORT_MAIN", "") or ""),
        "reasoning_effort_fast": (getattr(settings, "CODEX_REASONING_EFFORT_FAST", "") or ""),
        "message": "Codex models refreshed and applied to the current session.",
    }
