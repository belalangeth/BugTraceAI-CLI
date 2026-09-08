"""Unit tests for the Codex (ChatGPT login) provider support.

Covers:
- ~/.codex/auth.json parsing, JWT expiry and auto-refresh
- Responses API payload/header builders (LLMCodexMixin)
- Responses API response parsing in generate (LLMGenerateMixin)
- [CODEX] config loader and the codex.json provider preset
"""

import base64
import configparser
import json
import os
import stat
import time
from pathlib import Path

from bugtrace.core import codex_auth as ca
from bugtrace.core import codex_models as cm
from bugtrace.core.codex_auth import _extract_model_slugs
from bugtrace.core.config_loaders import SettingsLoadersMixin
from bugtrace.core.llm_shell.codex_wire import LLMCodexMixin
from bugtrace.core.llm_shell.generate import LLMGenerateMixin

REPO_ROOT = Path(__file__).resolve().parent.parent


# ───────────────────────────── helpers ─────────────────────────────


def make_jwt(exp: int) -> str:
    """Build a minimal JWT carrying only the `exp` claim."""
    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()
    header = b64(json.dumps({"alg": "none"}).encode())
    payload = b64(json.dumps({"exp": exp}).encode())
    return f"{header}.{payload}.sig"


class _FakeResp:
    def __init__(self, status=200, data=None):
        self.status = status
        self._data = data or {}

    async def json(self):
        return self._data

    async def text(self):
        return json.dumps(self._data)


class _FakeTracker:
    def record_usage(self, **kwargs):
        pass


class _FakeSem:
    """Minimal asyncio-semaphore stand-in with a readable _value."""

    _value = 5

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _Client(LLMGenerateMixin, LLMCodexMixin):
    """Minimal client: generate mixin + codex mixin with stubbed telemetry."""

    def __init__(self):
        self.req_count = 0
        self.token_tracker = _FakeTracker()
        self.model_metrics = {}
        self.REFUSAL_PHRASES = []
        self.models = ["gpt-5.4"]
        self._rate_limiters = {}
        self._codex_token_cache = None
        self._codex_token_expires = 0.0
        self._codex_account_id = None

    def _record_model_call(self, *args, **kwargs):
        pass

    def _get_model_semaphore(self, model):
        return _FakeSem()

    def _is_anthropic_model(self, model):
        return False

    async def _rate_limit_acquire(self, provider_id=None):
        return None

    async def _handle_refusal(self, text, *args, **kwargs):
        return text

    async def _update_telemetry(self, *args, **kwargs):
        pass

    async def _audit_log(self, *args, **kwargs):
        pass

    async def update_balance(self):
        pass


def _write_auth(tmp_path: Path, data: dict, name: str = "auth.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


# ───────────────────────────── codex_auth ─────────────────────────────


def test_jwt_exp_decodes_claim():
    exp = int(time.time()) + 3600
    assert ca._jwt_exp(make_jwt(exp)) == exp


def test_jwt_exp_returns_none_on_garbage():
    assert ca._jwt_exp("not.a.jwt") is None


def test_load_codex_auth_reads_chatgpt_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "_get_token_path", lambda: _write_auth(tmp_path, {
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": "tok-abc",
            "refresh_token": "refresh-xyz",
            "account_id": "acct-1",
        },
    }))
    data = ca.load_codex_auth()
    assert data is not None
    assert data["tokens"]["access_token"] == "tok-abc"


def test_load_codex_auth_rejects_apikey_only_login(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "_get_token_path", lambda: _write_auth(tmp_path, {
        "OPENAI_API_KEY": "sk-123",
    }))
    assert ca.load_codex_auth() is None


def test_load_codex_auth_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ca, "_get_token_path", lambda: tmp_path / "nope.json")
    assert ca.load_codex_auth() is None


def test_save_codex_auth_writes_0600(tmp_path, monkeypatch):
    target = tmp_path / "auth.json"
    monkeypatch.setattr(ca, "_get_token_path", lambda: target)
    assert ca.save_codex_auth({"tokens": {"access_token": "a"}}) is True
    assert target.exists()
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o600
    assert json.loads(target.read_text())["tokens"]["access_token"] == "a"


def test_get_valid_token_returns_unexpired(tmp_path, monkeypatch):
    exp = int(time.time()) + 3600
    monkeypatch.setattr(ca, "_get_token_path", lambda: _write_auth(tmp_path, {
        "tokens": {
            "access_token": make_jwt(exp),
            "refresh_token": "refresh-xyz",
            "account_id": "acct-1",
        },
    }))

    async def boom(*args, **kwargs):
        raise AssertionError("refresh should not be called for a fresh token")

    monkeypatch.setattr(ca, "refresh_codex_token", boom)
    pair = asyncio_run(ca.get_valid_codex_token())
    assert pair == (make_jwt(exp), "acct-1")


def test_get_valid_token_refreshes_when_expired(tmp_path, monkeypatch):
    expired = int(time.time()) - 3600
    auth_path = _write_auth(tmp_path, {
        "tokens": {
            "access_token": make_jwt(expired),
            "refresh_token": "refresh-xyz",
            "account_id": "acct-1",
        },
    })
    monkeypatch.setattr(ca, "_get_token_path", lambda: auth_path)

    async def fake_refresh(refresh_token):
        assert refresh_token == "refresh-xyz"
        return {
            "access_token": make_jwt(int(time.time()) + 3600),
            "refresh_token": "refresh-new",
            "id_token": "id-new",
        }

    monkeypatch.setattr(ca, "refresh_codex_token", fake_refresh)
    token, account_id = asyncio_run(ca.get_valid_codex_token())
    assert account_id == "acct-1"
    assert ca._jwt_exp(token) > time.time()  # refreshed token returned
    # File rewritten with the new tokens so the Codex CLI stays in sync.
    saved = json.loads(auth_path.read_text())
    assert saved["tokens"]["access_token"] == token
    assert saved["tokens"]["refresh_token"] == "refresh-new"
    assert "last_refresh" in saved


def test_get_valid_token_none_when_no_refresh_token(tmp_path, monkeypatch):
    expired = int(time.time()) - 3600
    monkeypatch.setattr(ca, "_get_token_path", lambda: _write_auth(tmp_path, {
        "tokens": {"access_token": make_jwt(expired)},  # no refresh_token
    }))
    assert asyncio_run(ca.get_valid_codex_token()) is None


def test_get_valid_token_none_when_refresh_fails(tmp_path, monkeypatch):
    expired = int(time.time()) - 3600
    monkeypatch.setattr(ca, "_get_token_path", lambda: _write_auth(tmp_path, {
        "tokens": {
            "access_token": make_jwt(expired),
            "refresh_token": "refresh-xyz",
        },
    }))

    async def failing_refresh(refresh_token):
        return None

    monkeypatch.setattr(ca, "refresh_codex_token", failing_refresh)
    assert asyncio_run(ca.get_valid_codex_token()) is None


# ───────────────────────────── codex_wire ─────────────────────────────


class _CodexMixinHost(LLMCodexMixin):
    def __init__(self, token="tok", account_id="acct-1"):
        self._codex_token_cache = token
        self._codex_token_expires = time.time() + 300
        self._codex_account_id = account_id


def test_build_codex_payload_converts_messages():
    host = _CodexMixinHost()
    payload = host._build_codex_payload(
        "gpt-5.4",
        [
            {"role": "system", "content": "You are a pentester."},
            {"role": "user", "content": "Analyze this."},
        ],
        max_tokens=1500,
        temperature=0.7,
    )
    assert payload["model"] == "gpt-5.4"
    assert payload["instructions"] == "You are a pentester."
    assert payload["input"] == [{"role": "user", "content": "Analyze this."}]
    assert payload["store"] is False
    assert payload["max_output_tokens"] == 1500
    # ChatGPT backend applies its own default — no explicit temperature.
    assert "temperature" not in payload
    assert "messages" not in payload


def test_build_codex_payload_omits_instructions_when_no_system():
    host = _CodexMixinHost()
    payload = host._build_codex_payload(
        "gpt-5.4-mini",
        [{"role": "user", "content": "Hi"}],
        max_tokens=100,
    )
    assert "instructions" not in payload
    assert payload["input"] == [{"role": "user", "content": "Hi"}]


def test_build_codex_headers():
    host = _CodexMixinHost()
    headers = host._build_codex_headers("ping")
    assert headers["Authorization"] == "Bearer tok"
    assert headers["ChatGPT-Account-ID"] == "acct-1"
    assert headers["Content-Type"] == "application/json"


def test_build_codex_headers_without_account_id():
    host = _CodexMixinHost(account_id=None)
    headers = host._build_codex_headers("ping")
    assert "ChatGPT-Account-ID" not in headers


# ───────────────────────────── request routing ─────────────────────────────


class _FakePost:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    def __init__(self, resp):
        self._resp = resp
        self.posted = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, url, headers=None, json=None, timeout=None):
        self.posted = (url, headers, json)
        return _FakePost(self._resp)


class _FakeOrchestrator:
    def __init__(self, resp):
        self.session_obj = _FakeSession(resp)

    def session(self, *args, **kwargs):
        return self.session_obj


# ───────────────────────────── response parsing ─────────────────────────────


def _responses_payload(text, usage=None):
    return {
        "id": "resp_1",
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        ],
        "usage": usage or {"input_tokens": 11, "output_tokens": 22, "total_tokens": 33},
    }


def test_handle_api_response_parses_responses_output():
    client = _Client()
    resp = _FakeResp(200, _responses_payload("hello world"))
    result = asyncio_run(client._handle_api_response(
        resp, "gpt-5.4", "test", "prompt", 10.0, None, None, 0.7, 1500,
        is_responses=True,
    ))
    assert result == "hello world"
    # Usage mapped to the telemetry fields BugTraceAI expects.
    assert resp._data["usage"]["prompt_tokens"] == 11
    assert resp._data["usage"]["completion_tokens"] == 22
    assert resp._data["usage"]["total_tokens"] == 33


def test_handle_api_response_empty_output_returns_none():
    client = _Client()
    resp = _FakeResp(200, {"output": []})
    result = asyncio_run(client._handle_api_response(
        resp, "gpt-5.4", "test", "prompt", 10.0, None, None, 0.7, 1500,
        is_responses=True,
    ))
    assert result is None


def test_handle_api_response_defaults_to_chat_format():
    client = _Client()
    resp = _FakeResp(200, {"choices": [{"message": {"content": "chat answer"}}]})
    result = asyncio_run(client._handle_api_response(
        resp, "qwen/qwen3-coder", "test", "prompt", 10.0, None, None, 0.7, 1500,
    ))
    assert result == "chat answer"


def test_attempt_model_generation_routes_codex_request(monkeypatch):
    """The codex provider sends a Responses API payload to the ChatGPT backend
    with the borrowed Codex CLI token, and parses the output back out."""
    import bugtrace.core.llm_shell.generate as generate_module

    async def fake_token():
        return ("tok123", "acct-1")

    monkeypatch.setattr("bugtrace.core.codex_auth.get_valid_codex_token", fake_token)

    client = _Client()
    client.api_format = "responses"
    client.provider_id = "codex"
    client.base_url = "https://chatgpt.com/backend-api/codex/responses"

    resp = _FakeResp(200, _responses_payload("routed!"))
    fake_orch = _FakeOrchestrator(resp)
    monkeypatch.setattr(generate_module, "orchestrator", fake_orch)

    result = asyncio_run(client._attempt_model_generation(
        "gpt-5.4",
        {"Authorization": "Bearer ignored"},
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "test", "prompt", 0.7, 100, None, None,
        {"provider_id": "codex", "base_url": "https://chatgpt.com/backend-api/codex/responses"},
    ))
    assert result == "routed!"

    url, headers, payload = fake_orch.session_obj.posted
    assert url == "https://chatgpt.com/backend-api/codex/responses"
    assert headers["Authorization"] == "Bearer tok123"
    assert headers["ChatGPT-Account-ID"] == "acct-1"
    assert payload["model"] == "gpt-5.4"
    assert payload["instructions"] == "sys"
    assert payload["input"] == [{"role": "user", "content": "hi"}]
    assert payload["max_output_tokens"] == 100


# ───────────────────────────── config + preset ─────────────────────────────


def test_codex_config_loader_section():
    class Dummy(SettingsLoadersMixin):
        CODEX_AUTH_ENABLED = False
        CODEX_TOKEN_FILE = "~/.codex/auth.json"

    dummy = Dummy()
    config = configparser.ConfigParser()
    config.read_string(
        "[CODEX]\n"
        "ENABLED = True\n"
        "TOKEN_FILE = /tmp/my-auth.json\n"
    )
    dummy._load_codex_config(config)
    assert dummy.CODEX_AUTH_ENABLED is True
    assert dummy.CODEX_TOKEN_FILE == "/tmp/my-auth.json"


def test_codex_config_loader_missing_section_is_noop():
    class Dummy(SettingsLoadersMixin):
        CODEX_AUTH_ENABLED = False
        CODEX_TOKEN_FILE = "~/.codex/auth.json"

    dummy = Dummy()
    dummy._load_codex_config(configparser.ConfigParser())
    assert dummy.CODEX_AUTH_ENABLED is False


def test_codex_preset_is_valid_responses_provider():
    preset_path = REPO_ROOT / "bugtrace" / "data" / "providers" / "codex.json"
    preset = json.loads(preset_path.read_text())
    assert preset["id"] == "codex"
    assert preset["api_format"] == "responses"
    assert preset["api_key_env"] == ""  # no API key — ChatGPT login instead
    assert preset["base_url"].endswith("/backend-api/codex/responses")
    assert preset["models"]["PRIMARY_MODELS"]


# ───────────────────────────── model auto-discovery ─────────────────────────────

SAMPLE_SLUGS = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]


def test_is_light_slug():
    assert cm.is_light_slug("gpt-5.4-mini") is True
    assert cm.is_light_slug("gpt-5.4-nano") is True
    assert cm.is_light_slug("gpt-5.4") is False
    assert cm.is_light_slug("gpt-5-codex") is False


def test_plan_codex_assignments_heavy_light_pool():
    a = cm.plan_codex_assignments(SAMPLE_SLUGS)
    assert a["DEFAULT_MODEL"] == "gpt-5.4"
    assert a["VISION_MODEL"] == "gpt-5.4"
    assert a["REPORTING_MODEL"] == "gpt-5.4-mini"
    assert a["MUTATION_MODEL"] == "gpt-5.4-mini"
    assert a["PRIMARY_MODELS"] == "gpt-5.4,gpt-5.4-mini,gpt-5.4-nano"
    # Every slot field the codebase knows must be present.
    for f in cm.HEAVY_SLOT_FIELDS + cm.LIGHT_SLOT_FIELDS + ["PRIMARY_MODELS"]:
        assert f in a


def test_plan_keeps_pinned_slug_when_still_served():
    a = cm.plan_codex_assignments(
        SAMPLE_SLUGS,
        preferred_heavy="gpt-5.4-mini",  # user pinned the mini for heavy slots
        preferred_light="gpt-5.4-mini",
    )
    assert a["DEFAULT_MODEL"] == "gpt-5.4-mini"  # pinned slug still served


def test_plan_falls_forward_when_pinned_slug_dropped():
    a = cm.plan_codex_assignments(
        ["gpt-5.5", "gpt-5.5-mini"],
        preferred_heavy="gpt-5.4",  # no longer served
        preferred_light="gpt-5.4-mini",
    )
    assert a["DEFAULT_MODEL"] == "gpt-5.5"
    assert a["REPORTING_MODEL"] == "gpt-5.5-mini"


def test_plan_empty_slugs_returns_empty():
    assert cm.plan_codex_assignments([]) == {}
    assert cm.plan_codex_assignments(None) == {}


def test_extract_model_slugs_filters_unsupported_and_hidden():
    payload = {"models": [
        {"slug": "gpt-5.4", "supported_in_api": True, "visibility": "list"},
        {"slug": "gpt-5.4-mini", "supported_in_api": True, "visibility": "list"},
        {"slug": "hidden-model", "supported_in_api": True, "visibility": "internal"},
        {"slug": "no-api-model", "supported_in_api": False, "visibility": "list"},
        {"slug": "legacy", "visibility": "list"},  # missing supported_in_api
    ]}
    assert _extract_model_slugs(payload) == ["gpt-5.4", "gpt-5.4-mini"]


def test_extract_model_slugs_bad_payload():
    assert _extract_model_slugs({}) is None


def test_apply_assignments_updates_settings_and_provider_config():
    class _DummySettings:
        def __init__(self):
            self.DEFAULT_MODEL = "gpt-5.4"
            self.PRIMARY_MODELS = "gpt-5.4"
            self._provider_config = {"models": {"DEFAULT_MODEL": "gpt-5.4"}}

    dummy = _DummySettings()
    assert cm.apply_assignments(
        {"DEFAULT_MODEL": "gpt-5.5", "PRIMARY_MODELS": "gpt-5.5,gpt-5.5-mini"}, dummy
    ) is True
    assert dummy.DEFAULT_MODEL == "gpt-5.5"
    assert dummy._provider_config["models"]["PRIMARY_MODELS"] == "gpt-5.5,gpt-5.5-mini"
    assert cm.apply_assignments({}, dummy) is False


class _CodexClientHost(LLMCodexMixin):
    def __init__(self):
        self.provider_id = "codex"
        self.api_format = "responses"
        self.models = ["gpt-5.4", "gpt-5.4-mini"]


async def _fake_fetch_ok():
    return SAMPLE_SLUGS


async def _fake_fetch_none():
    return None


async def _fake_fetch_boom():
    raise RuntimeError("backend unreachable")


def test_maybe_refresh_applies_discovered_models(monkeypatch):
    import copy
    from bugtrace.core.config import settings

    host = _CodexClientHost()
    monkeypatch.setattr(settings, "CODEX_MODEL_AUTODISCOVERY", True)
    monkeypatch.setattr(settings, "DEFAULT_MODEL", "gpt-5.4")
    monkeypatch.setattr(settings, "REPORTING_MODEL", "gpt-5.4-mini")
    # Isolate the runtime preset dict so the test's writes don't leak globally.
    monkeypatch.setattr(
        settings, "_provider_config", copy.deepcopy(getattr(settings, "_provider_config", {}))
    )
    monkeypatch.setattr("bugtrace.core.codex_auth.fetch_codex_models", _fake_fetch_ok)

    asyncio_run(host._maybe_refresh_codex_models())

    assert settings.DEFAULT_MODEL == "gpt-5.4"
    assert settings.REPORTING_MODEL == "gpt-5.4-mini"
    assert settings.PRIMARY_MODELS == "gpt-5.4,gpt-5.4-mini,gpt-5.4-nano"
    assert host.models == ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"]


def test_maybe_refresh_keeps_preset_on_failure(monkeypatch):
    from bugtrace.core.config import settings

    host = _CodexClientHost()
    monkeypatch.setattr(settings, "CODEX_MODEL_AUTODISCOVERY", True)
    monkeypatch.setattr(settings, "PRIMARY_MODELS", "gpt-5.4,gpt-5.4-mini")
    monkeypatch.setattr("bugtrace.core.codex_auth.fetch_codex_models", _fake_fetch_none)

    asyncio_run(host._maybe_refresh_codex_models())
    assert settings.PRIMARY_MODELS == "gpt-5.4,gpt-5.4-mini"

    # Exceptions must never propagate either.
    monkeypatch.setattr("bugtrace.core.codex_auth.fetch_codex_models", _fake_fetch_boom)
    asyncio_run(host._maybe_refresh_codex_models())
    assert settings.PRIMARY_MODELS == "gpt-5.4,gpt-5.4-mini"


def test_maybe_refresh_respects_autodiscovery_flag(monkeypatch):
    from bugtrace.core.config import settings

    host = _CodexClientHost()
    monkeypatch.setattr(settings, "CODEX_MODEL_AUTODISCOVERY", False)

    async def should_not_be_called():
        raise AssertionError("fetch must not run when autodiscovery is off")

    monkeypatch.setattr("bugtrace.core.codex_auth.fetch_codex_models", should_not_be_called)
    asyncio_run(host._maybe_refresh_codex_models())


def test_maybe_refresh_ignores_non_codex_provider(monkeypatch):
    host = _CodexClientHost()
    host.provider_id = "openrouter-v2"
    host.api_format = "openai"

    async def should_not_be_called():
        raise AssertionError("fetch must not run for non-codex providers")

    monkeypatch.setattr("bugtrace.core.codex_auth.fetch_codex_models", should_not_be_called)
    asyncio_run(host._maybe_refresh_codex_models())


def test_codex_config_loader_autodiscovery_flag():
    class Dummy(SettingsLoadersMixin):
        CODEX_AUTH_ENABLED = False
        CODEX_TOKEN_FILE = "~/.codex/auth.json"
        CODEX_MODEL_AUTODISCOVERY = True

    dummy = Dummy()
    config = configparser.ConfigParser()
    config.read_string("[CODEX]\nMODEL_AUTODISCOVERY = False\nENABLED = True\n")
    dummy._load_codex_config(config)
    assert dummy.CODEX_MODEL_AUTODISCOVERY is False
    assert dummy.CODEX_AUTH_ENABLED is True

# ───────────────────────────── runner helper ─────────────────────────────


def asyncio_run(coro):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        # Called from inside a running loop — return a task for the caller.
        return loop.create_task(coro)
    return asyncio.run(coro)