"""
Codex (ChatGPT login) token management.

Reads the OAuth credentials written by the official OpenAI Codex CLI
(`codex auth login` with a ChatGPT account) from `~/.codex/auth.json`
(or `$CODEX_HOME/auth.json`) and handles automatic refresh via
auth.openai.com when the access token is expired or near expiry.

No login flow here — the user authenticates with the Codex CLI itself:
    npm install -g @openai/codex
    codex auth login        # pick "Sign in with ChatGPT"

This is the same credential-borrowing approach used by
simonw/llm-openai-via-codex: we never see the password, we only read the
tokens the Codex CLI has already stored, refresh them when needed, and
write the refreshed tokens back so the Codex CLI stays in sync.
"""
import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

from bugtrace.utils.logger import get_logger

logger = get_logger("core.codex_auth")

# OAuth constants — public client_id used by the Codex CLI itself for
# ChatGPT-account token refresh (see openai/codex and llm-openai-via-codex).
CODEX_REFRESH_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REFRESH_SKEW_SECONDS = 30

# ChatGPT backend used for both generation (/responses) and model discovery
# (/models). Mirrors the base URL of the official Codex CLI.
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_MODELS_URL = f"{CODEX_BASE_URL}/models?client_version=1.0.0"

DEFAULT_TOKEN_FILE = Path("~/.codex/auth.json")

# Refresh errors that mean the refresh token itself is dead → user must
# re-run `codex auth login`.
FATAL_REFRESH_ERRORS = (
    "refresh_token_expired",
    "refresh_token_reused",
    "refresh_token_invalidated",
)


def _get_token_path() -> Path:
    """Resolve the Codex auth file path.

    Precedence: BUGTRACE_CODEX_TOKEN_FILE > [CODEX] TOKEN_FILE setting >
    $CODEX_HOME/auth.json > ~/.codex/auth.json.
    """
    env_path = os.environ.get("BUGTRACE_CODEX_TOKEN_FILE")
    if env_path:
        return Path(env_path).expanduser()
    try:
        from bugtrace.core.config import settings
        configured = getattr(settings, "CODEX_TOKEN_FILE", "") or ""
        if configured:
            return Path(configured).expanduser()
    except Exception:
        pass
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home) / "auth.json"
    return DEFAULT_TOKEN_FILE.expanduser()


def load_codex_auth() -> Optional[Dict]:
    """Read the Codex auth file. Returns the parsed dict or None."""
    path = _get_token_path()
    if not path.exists():
        logger.debug(f"Codex auth file not found: {path}")
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read Codex auth file {path}: {e}")
        return None
    if not isinstance(data, dict):
        logger.warning(f"Codex auth file {path} is malformed (not a JSON object)")
        return None
    tokens = data.get("tokens")
    if not isinstance(tokens, dict) or not tokens.get("access_token"):
        # Some Codex setups store only an API key (OPENAI_API_KEY) — that is a
        # different auth mode; direct the user to the `openai` provider preset.
        if data.get("OPENAI_API_KEY"):
            logger.warning(
                "~/.codex/auth.json contains an API key login, not ChatGPT OAuth "
                "tokens. Use the `openai` provider preset with OPENAI_API_KEY "
                "instead of the `codex` provider."
            )
        return None
    return data


def save_codex_auth(data: Dict) -> bool:
    """Atomically write the Codex auth file back with restrictive permissions."""
    path = _get_token_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        os.chmod(path, 0o600)
        return True
    except (OSError, TypeError) as e:
        logger.error(f"Failed to write Codex auth file {path}: {e}")
        return False


def _jwt_exp(token: str) -> Optional[int]:
    """Decode the `exp` claim from a JWT access token (seconds since epoch)."""
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except Exception:
        return None


async def refresh_codex_token(refresh_token: str) -> Optional[Dict]:
    """Exchange a refresh_token for new Codex tokens from auth.openai.com."""
    body = {
        "client_id": CODEX_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                CODEX_REFRESH_URL,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    error_body = (await resp.text())[:500]
                    try:
                        error_code = (json.loads(error_body) or {}).get("error")
                    except Exception:
                        error_code = None
                    if error_code in FATAL_REFRESH_ERRORS:
                        logger.error(
                            f"Codex token refresh failed: {error_code} — the login is dead. "
                            "Re-run `codex auth login`."
                        )
                    else:
                        logger.error(f"Codex token refresh failed ({resp.status}): {error_body}")
                    return None
                data = await resp.json()
                if not data.get("access_token"):
                    logger.error("Codex token refresh: no access_token in response")
                    return None
                return data
    except asyncio.TimeoutError as e:
        logger.error(f"Codex token refresh timed out: {e}")
        return None
    except Exception as e:
        logger.error(f"Codex token refresh exception: {e}")
        return None


async def get_valid_codex_token() -> Optional[Tuple[str, Optional[str]]]:
    """Return a valid (access_token, account_id) pair from the Codex CLI login.

    Refreshes automatically when the access token is expired or near expiry.
    Returns None when no usable ChatGPT OAuth login exists.
    """
    data = load_codex_auth()
    if not data:
        return None
    tokens = data.get("tokens") or {}
    access_token = tokens.get("access_token")
    if not access_token:
        return None

    account_id = tokens.get("account_id")
    exp = _jwt_exp(access_token)
    if exp is not None and time.time() < (exp - REFRESH_SKEW_SECONDS):
        return access_token, account_id

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        logger.error(
            "Codex access token expired and no refresh token is available. "
            "Re-run `codex auth login`."
        )
        return None

    logger.info("Codex access token expired — refreshing via auth.openai.com...")
    new_tokens = await refresh_codex_token(refresh_token)
    if not new_tokens or not new_tokens.get("access_token"):
        logger.error(
            "Codex token refresh failed. Re-run `codex auth login` to re-authenticate."
        )
        return None

    # Write the refreshed tokens back so the Codex CLI stays in sync.
    if new_tokens.get("access_token"):
        tokens["access_token"] = new_tokens["access_token"]
    if new_tokens.get("id_token"):
        tokens["id_token"] = new_tokens["id_token"]
    if new_tokens.get("refresh_token"):
        tokens["refresh_token"] = new_tokens["refresh_token"]
    data["tokens"] = tokens
    data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    save_codex_auth(data)

    return tokens["access_token"], tokens.get("account_id")


def codex_auth_available() -> bool:
    """Synchronous availability check (used by the WEB provider list).

    True when a ChatGPT OAuth login exists in the Codex auth file.
    """
    data = load_codex_auth()
    return bool(data)


def _extract_model_slugs(payload: Dict) -> Optional[List[str]]:
    """Pull the served model slugs out of a Codex /models response.

    Mirrors the filter used by the official Codex CLI ecosystem: only models
    that are actually callable through the Responses API and are listed (not
    hidden/legacy) are usable.
    """
    try:
        models = payload.get("models", []) or []
        slugs = [
            m.get("slug")
            for m in models
            if isinstance(m, dict)
            and m.get("slug")
            and m.get("supported_in_api")
            and m.get("visibility") == "list"
        ]
        return slugs or None
    except Exception as e:
        logger.warning(f"Failed to parse Codex /models response: {e}")
        return None


async def fetch_codex_models() -> Optional[List[str]]:
    """Fetch the model slugs the ChatGPT backend serves for this login.

    Returns a list ordered as the backend reports it (newest/primary first),
    or None when there is no valid login or the call fails — callers then keep
    the model list from the preset instead of erroring out.
    """
    pair = await get_valid_codex_token()
    if not pair:
        logger.warning("Codex model discovery skipped — no valid ChatGPT login")
        return None
    token, account_id = pair
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if account_id:
        headers["ChatGPT-Account-ID"] = account_id
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                CODEX_MODELS_URL,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Codex /models returned HTTP {resp.status} — keeping preset models")
                    return None
                payload = await resp.json()
                return _extract_model_slugs(payload)
    except asyncio.TimeoutError:
        logger.warning("Codex /models request timed out — keeping preset models")
        return None
    except Exception as e:
        logger.warning(f"Codex model discovery failed: {e}")
        return None