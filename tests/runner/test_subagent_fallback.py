"""Cross-harness sub-agent fallback (AB#2691).

A sub-agent's ``executor.config.fallback`` block declares where to
re-dispatch its work when a turn terminally fails with a matching error
kind (default: ``quota``). The runner intercepts the failure at the
terminal-report site, spawns a FRESH child session with the fallback
harness/model overrides, replays the original message, and delivers the
eventual result under the ORIGINAL ``work_id`` — the parent never sees
the intercepted quota failure.

Covers the config reader, the parser's structured-key handling (including
the previously stringified ``allowed_harnesses`` list), the error-kind
classifier, and the end-to-end native failure → fallback → delivery paths.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from omnigent.errors import OmnigentError
from omnigent.runner import app as runner_app
from omnigent.runner import create_runner_app
from omnigent.runner.tool_dispatch import (
    FallbackTarget,
    _consult_ppu_oracle,
    _ppu_oracle_payload_eligible,
    _sanitize_ppu_reason,
    _subagent_fallback_targets,
)
from omnigent.spec import parser as spec_parser
from omnigent.spec.types import AgentSpec, ExecutorSpec
from omnigent.turn_errors import classify_turn_error_kind

# Reuse the proven runner-turn stubs from the sessions-native suite.
from tests.runner.helpers import NullServerClient
from tests.runner.test_app_sessions_native import (
    _FakeProcessManager,
    _runner_client,
    _ScriptedHarnessClient,
)

PARENT_SESSION_ID = "conv_parent_orchestrator"
CHILD_SESSION_ID = "conv_child_reviewer"
FALLBACK_CHILD_SESSION_ID = "conv_child_reviewer_fb1"

REVIEW_MESSAGE = "Please review the diff in PR #42."
QUOTA_OUTPUT = "Error: sub-agent turn failed: You've hit your usage limit."


# ── Unit: error-kind classifier ───────────────────────────────────────


def test_classifier_quota_by_code_and_text() -> None:
    assert classify_turn_error_kind(None, 429) == "quota"
    assert classify_turn_error_kind(None, "429") == "quota"
    # Codex wording.
    assert classify_turn_error_kind("usage_limit_exceeded") == "quota"
    assert classify_turn_error_kind("You've hit your usage limit.") == "quota"
    # Anthropic/Claude wording (including the 5-hour window phrasing).
    assert classify_turn_error_kind("You've hit your limit · resets 3am") == "quota"
    assert classify_turn_error_kind("5-hour limit reached") == "quota"
    assert classify_turn_error_kind("Rate limit exceeded") == "quota"
    # Generic provider wording.
    assert classify_turn_error_kind("HTTP 429 Too Many Requests") == "quota"
    assert classify_turn_error_kind("quota exceeded for project") == "quota"
    assert classify_turn_error_kind("request was rate-limited") == "quota"


def test_classifier_monthly_spend_limit_is_quota() -> None:
    """Anthropic's subscription monthly-spend hard-stop is a quota failure."""
    # API error body wording (rate_limit_error).
    assert (
        classify_turn_error_kind(
            "This request would exceed your account's monthly spend limit. Please try again later."
        )
        == "quota"
    )
    # Claude Code surface wording.
    assert (
        classify_turn_error_kind(
            "You've hit your org's monthly spend limit · run /usage-credits to raise it"
        )
        == "quota"
    )
    assert classify_turn_error_kind("monthly spend limit reached") == "quota"


def test_classifier_spend_adjacent_prose_stays_generic() -> None:
    """Anchor check: spend wording without the full phrase is not quota."""
    assert classify_turn_error_kind("spending review") == "generic"
    assert classify_turn_error_kind("spend limit discussion in budget doc") == "generic"


def test_classifier_auth_by_code_and_text() -> None:
    assert classify_turn_error_kind(None, 401) == "auth"
    assert classify_turn_error_kind(None, 403) == "auth"
    assert classify_turn_error_kind("please run codex login") == "auth"
    assert classify_turn_error_kind("token expired, re-authenticate") == "auth"


def test_classifier_does_not_false_positive_on_code_like_text() -> None:
    """Digits that merely contain 429, and token counts, are not quota.

    A false quota positive replays the message on a fresh child and
    re-executes side effects, so "429" only matches the structured code.
    """
    assert classify_turn_error_kind("request id abc429def") == "generic"
    assert classify_turn_error_kind("14290 tokens") == "generic"
    assert classify_turn_error_kind("processed 14290 tokens in 4290ms") == "generic"
    assert classify_turn_error_kind("error 429") == "generic"
    # Anchored phrases still match, including quoted tool output.
    assert classify_turn_error_kind("GitHub API rate limit exceeded") == "quota"


def test_classifier_generic_otherwise() -> None:
    assert classify_turn_error_kind(None) == "generic"
    assert classify_turn_error_kind("") == "generic"
    assert classify_turn_error_kind("disk full") == "generic"
    assert classify_turn_error_kind("model stream broke", 500) == "generic"


# ── Unit: fallback config reader ──────────────────────────────────────


def _spec_with_fallback(fallback: object) -> SimpleNamespace:
    config: dict[str, Any] = {"harness": "codex-native"}
    if fallback is not None:
        config["fallback"] = fallback
    return SimpleNamespace(name="reviewer", executor=SimpleNamespace(config=config))


def test_fallback_targets_single_mapping_defaults_to_quota() -> None:
    spec = _spec_with_fallback({"harness": "pi", "model": "google/gemini-2.5-pro"})
    assert _subagent_fallback_targets(spec) == (
        FallbackTarget(harness="pi", model="google/gemini-2.5-pro", on=frozenset({"quota"})),
    )


def test_fallback_targets_ordered_list_with_on() -> None:
    spec = _spec_with_fallback(
        [
            {"harness": "pi", "model": "google/gemini-2.5-pro", "on": ["quota", "auth"]},
            {"harness": "claude-native", "on": "generic"},
        ]
    )
    targets = _subagent_fallback_targets(spec)
    assert targets == (
        FallbackTarget(
            harness="pi", model="google/gemini-2.5-pro", on=frozenset({"quota", "auth"})
        ),
        FallbackTarget(harness="claude-native", model=None, on=frozenset({"generic"})),
    )


def test_fallback_targets_malformed_entries_skipped_never_raise() -> None:
    spec = _spec_with_fallback(
        [
            "not-a-mapping",
            {"model": "gpt-5"},  # missing harness
            {"harness": ""},  # empty harness
            {"harness": "pi", "on": ["bogus-kind"]},  # no recognized kind
            {"harness": "pi", "on": 42},  # non-list on
            {"harness": "pi", "on": ["quota"]},  # the one valid entry
        ]
    )
    assert _subagent_fallback_targets(spec) == (
        FallbackTarget(harness="pi", model=None, on=frozenset({"quota"})),
    )


def test_fallback_targets_absent_or_stringified_yield_empty() -> None:
    assert _subagent_fallback_targets(None) == ()
    assert _subagent_fallback_targets(_spec_with_fallback(None)) == ()
    # A pre-fix parser stringified the block; the reader must not crash on it.
    assert _subagent_fallback_targets(_spec_with_fallback("{'harness': 'pi'}")) == ()


def test_fallback_targets_consult_ppu_parsed_defaults_false_non_bool_skipped() -> None:
    spec = _spec_with_fallback(
        [
            {"harness": "pi", "consult_ppu": True},
            {"harness": "claude-native"},  # absent -> False
            {"harness": "codex-native", "consult_ppu": "yes"},  # non-bool -> skipped
        ]
    )
    assert _subagent_fallback_targets(spec) == (
        FallbackTarget(harness="pi", on=frozenset({"quota"}), consult_ppu=True),
        FallbackTarget(harness="claude-native", on=frozenset({"quota"}), consult_ppu=False),
    )


# ── Unit: parser keeps the structured executor.config keys ────────────


def test_parse_executor_keeps_fallback_and_allowed_harnesses_structured() -> None:
    executor = spec_parser._parse_executor(
        {
            "type": "omnigent",
            "config": {
                "harness": "codex-native",
                "allowed_harnesses": ["pi", "claude-native"],
                "fallback": [{"harness": "pi", "model": "google/gemini-2.5-pro", "on": ["quota"]}],
                "some_flag": True,
            },
        }
    )
    # The previously latent bug: this list was stringified, so
    # _subagent_allowed_harnesses always saw a str and returned frozenset().
    assert executor.config["allowed_harnesses"] == ["pi", "claude-native"]
    assert executor.config["fallback"] == [
        {"harness": "pi", "model": "google/gemini-2.5-pro", "on": ["quota"]}
    ]
    # Everything else keeps the string coercion.
    assert executor.config["some_flag"] == "True"
    assert executor.config["harness"] == "codex-native"


def test_parse_yaml_fallback_block_survives_end_to_end(tmp_path: Any) -> None:
    """A real config.yaml ``fallback:`` block round-trips through parse().

    Also exercises the loader's YAML-1.2 bool narrowing: the ``on:`` key
    must stay the string ``"on"`` (not the YAML 1.1 boolean ``True``).
    """
    (tmp_path / "config.yaml").write_text(
        "spec_version: 1\n"
        "name: reviewer\n"
        "executor:\n"
        "  type: omnigent\n"
        "  config:\n"
        "    harness: codex-native\n"
        "    allowed_harnesses:\n"
        "      - pi\n"
        "    fallback:\n"
        "      - harness: pi\n"
        "        model: google/gemini-2.5-pro\n"
        "        on: [quota, auth]\n"
        "        consult_ppu: true\n"
    )
    spec = spec_parser.parse(tmp_path)
    assert spec.executor.config["allowed_harnesses"] == ["pi"]
    assert spec.executor.config["fallback"] == [
        {
            "harness": "pi",
            "model": "google/gemini-2.5-pro",
            "on": ["quota", "auth"],
            "consult_ppu": True,
        }
    ]
    assert _subagent_fallback_targets(spec) == (
        FallbackTarget(
            harness="pi",
            model="google/gemini-2.5-pro",
            on=frozenset({"quota", "auth"}),
            consult_ppu=True,
        ),
    )


def test_parse_executor_rejects_non_bool_consult_ppu() -> None:
    """consult_ppu gates real spend, so a mistyped value fails at parse time."""
    for bad in ("yes", 1, None, [True]):
        with pytest.raises(OmnigentError, match="consult_ppu must be a boolean"):
            spec_parser._parse_executor(
                {
                    "type": "omnigent",
                    "config": {"fallback": {"harness": "pi", "consult_ppu": bad}},
                }
            )
    # List form is validated entry-by-entry.
    with pytest.raises(OmnigentError, match="consult_ppu must be a boolean"):
        spec_parser._parse_executor(
            {
                "type": "omnigent",
                "config": {
                    "fallback": [
                        {"harness": "pi", "consult_ppu": True},
                        {"harness": "claude-native", "consult_ppu": "false"},
                    ]
                },
            }
        )


# ── Unit: PPU consent oracle (consult_ppu gate) ───────────────────────


def _eligible_ppu_payload() -> dict[str, Any]:
    return {
        "ppu": {
            "anthropic": {
                "enabled": True,
                "health": {"status": "up"},
                "budget": {"budget_ok": True},
            }
        }
    }


def _ppu_payload_with(**anthropic_overrides: Any) -> dict[str, Any]:
    payload = _eligible_ppu_payload()
    payload["ppu"]["anthropic"].update(anthropic_overrides)
    return payload


def test_ppu_payload_eligible_only_on_exact_contract() -> None:
    assert _ppu_oracle_payload_eligible(_eligible_ppu_payload()) == (True, "eligible")


@pytest.mark.parametrize(
    ("payload", "reason_fragment"),
    [
        (["not", "a", "dict"], "not a JSON object"),
        ({}, "ppu.anthropic missing"),
        ({"ppu": "anthropic"}, "ppu.anthropic missing"),
        ({"ppu": {"anthropic": "yes"}}, "ppu.anthropic missing"),
        (_ppu_payload_with(enabled=False), "enabled is not true"),
        # Exact-type checks: truthy non-bools are not consent.
        (_ppu_payload_with(enabled=1), "enabled is not true"),
        (_ppu_payload_with(enabled="true"), "enabled is not true"),
        (_ppu_payload_with(health={"status": "down"}), "health.status is not 'up'"),
        (_ppu_payload_with(health="up"), "health.status is not 'up'"),
        (_ppu_payload_with(health={}), "health.status is not 'up'"),
        (_ppu_payload_with(budget={"budget_ok": False}), "budget_ok is not true"),
        (_ppu_payload_with(budget={"budget_ok": "true"}), "budget_ok is not true"),
        (_ppu_payload_with(budget=None), "budget_ok is not true"),
    ],
)
def test_ppu_payload_ineligible_shapes(payload: Any, reason_fragment: str) -> None:
    """Any missing/malformed shape is ineligible — consent is never assumed."""
    eligible, reason = _ppu_oracle_payload_eligible(payload)
    assert eligible is False
    assert reason_fragment in reason


def _patch_oracle_http(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status_code: int = 200,
    payload: Any = None,
    invalid_json: bool = False,
    exc: Exception | None = None,
) -> list[str]:
    """Replace httpx.AsyncClient with a canned oracle; returns requested URLs."""
    requested: list[str] = []

    class _Response:
        def __init__(self) -> None:
            self.status_code = status_code

        def json(self) -> Any:
            if invalid_json:
                raise ValueError("not json")
            return payload

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def get(self, url: str) -> _Response:
            requested.append(url)
            if exc is not None:
                raise exc
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return requested


@pytest.mark.asyncio
async def test_consult_ppu_oracle_eligible_uses_default_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIGENT_PPU_ORACLE_URL", raising=False)
    requested = _patch_oracle_http(monkeypatch, payload=_eligible_ppu_payload())
    assert await _consult_ppu_oracle() == (True, "eligible")
    assert requested == ["http://localhost:5151/quota"]


@pytest.mark.asyncio
async def test_consult_ppu_oracle_honors_env_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIGENT_PPU_ORACLE_URL", "https://127.0.0.1:9/quota")
    requested = _patch_oracle_http(monkeypatch, payload=_eligible_ppu_payload())
    assert await _consult_ppu_oracle() == (True, "eligible")
    assert requested == ["https://127.0.0.1:9/quota"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://evil.example/quota",
        "http://oracle.test:9999/quota",
        "http://127.0.0.2:5151/quota",  # loopback range but not the exact allowed host
        "not a url at all",
        "http://[::1]:5151/quota",  # IPv6 loopback intentionally excluded
    ],
)
async def test_consult_ppu_oracle_denies_non_loopback_url_without_request(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    """A spoofed/redirected oracle URL denies consent before any HTTP I/O."""
    monkeypatch.setenv("OMNIGENT_PPU_ORACLE_URL", url)
    requested = _patch_oracle_http(monkeypatch, payload=_eligible_ppu_payload())
    eligible, reason = await _consult_ppu_oracle()
    assert eligible is False
    assert reason == "oracle URL not loopback — consent denied"
    assert requested == [], "a non-loopback oracle URL must never be fetched"


def test_sanitize_ppu_reason_strips_control_chars_and_caps() -> None:
    raw = "denied\r\nreason\x1b[31m\x00" + "x" * 500
    cleaned = _sanitize_ppu_reason(raw)
    assert "\n" not in cleaned
    assert "\r" not in cleaned
    assert "\x1b" not in cleaned
    assert "\x00" not in cleaned
    assert cleaned.startswith("deniedreason[31mxxx")
    assert len(cleaned) == 200


@pytest.mark.asyncio
async def test_consult_ppu_oracle_sanitizes_every_returned_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sanitizer wraps the read itself, so a hostile reason never escapes."""

    async def _hostile_read() -> tuple[bool, str]:
        return (False, "oracle said:\r\nEVIL\x1b[31m " + "y" * 500)

    monkeypatch.setattr("omnigent.runner.tool_dispatch._read_ppu_oracle", _hostile_read)
    eligible, reason = await _consult_ppu_oracle()
    assert eligible is False
    assert "\n" not in reason
    assert "\x1b" not in reason
    assert len(reason) == 200


@pytest.mark.asyncio
async def test_consult_ppu_oracle_fails_closed_on_transport_and_body_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """500, timeout, and malformed JSON all deny consent, never grant it."""
    _patch_oracle_http(monkeypatch, status_code=500)
    eligible, reason = await _consult_ppu_oracle()
    assert (eligible, reason) == (False, "oracle returned HTTP 500")

    _patch_oracle_http(monkeypatch, exc=httpx.ConnectTimeout("timed out"))
    eligible, reason = await _consult_ppu_oracle()
    assert eligible is False
    assert "oracle unreachable: ConnectTimeout" in reason

    _patch_oracle_http(monkeypatch, invalid_json=True)
    eligible, reason = await _consult_ppu_oracle()
    assert (eligible, reason) == (False, "oracle response is not valid JSON")


# ── E2E: native failure → fallback re-dispatch → inbox delivery ───────


@pytest.fixture
def _clean_subagent_registry() -> Iterator[None]:
    """Snapshot and restore the process-wide sub-agent / inbox maps."""
    saved = (
        dict(runner_app._subagent_work_by_child),
        {k: set(v) for k, v in runner_app._subagent_work_by_parent.items()},
        dict(runner_app._session_inboxes_ref),
        set(runner_app._drained_delivered_subagent_children),
        dict(runner_app._child_session_parents),
        dict(runner_app._session_agent_ids_ref),
        {k: set(v) for k, v in runner_app._pending_subagent_fallbacks.items()},
        dict(runner_app._superseded_subagent_children_map),
    )
    runner_app._subagent_work_by_child.clear()
    runner_app._subagent_work_by_parent.clear()
    runner_app._session_inboxes_ref.clear()
    runner_app._drained_delivered_subagent_children.clear()
    runner_app._child_session_parents.clear()
    runner_app._session_agent_ids_ref.clear()
    runner_app._pending_subagent_fallbacks.clear()
    runner_app._superseded_subagent_children_map.clear()
    try:
        yield
    finally:
        runner_app._subagent_work_by_child.clear()
        runner_app._subagent_work_by_child.update(saved[0])
        runner_app._subagent_work_by_parent.clear()
        runner_app._subagent_work_by_parent.update(saved[1])
        runner_app._session_inboxes_ref.clear()
        runner_app._session_inboxes_ref.update(saved[2])
        runner_app._drained_delivered_subagent_children.clear()
        runner_app._drained_delivered_subagent_children.update(saved[3])
        runner_app._child_session_parents.clear()
        runner_app._child_session_parents.update(saved[4])
        runner_app._session_agent_ids_ref.clear()
        runner_app._session_agent_ids_ref.update(saved[5])
        runner_app._pending_subagent_fallbacks.clear()
        runner_app._pending_subagent_fallbacks.update(saved[6])
        runner_app._superseded_subagent_children_map.clear()
        runner_app._superseded_subagent_children_map.update(saved[7])


class _FallbackServerClient(NullServerClient):
    """Fake Omnigent server for the fallback re-dispatch flow.

    Serves session snapshots for the parent and both children, accepts the
    tombstone PATCH, records the fallback child create (returning a fixed
    new session id), and records every ``/events`` POST so the test can
    assert the original message was replayed. Failure modes are opt-in so
    each fail-safe branch of the engine can be driven.
    """

    def __init__(
        self,
        *,
        existing_children: list[dict[str, Any]] | None = None,
        fail_create: bool = False,
        fail_patch: bool = False,
        fail_message_post_to: str | None = None,
    ) -> None:
        self.created_sessions: list[dict[str, Any]] = []
        self.event_posts: list[tuple[str, dict[str, Any]]] = []
        self.policy_posts: list[tuple[str, dict[str, Any]]] = []
        self.patches: list[tuple[str, dict[str, Any]]] = []
        self._existing_children = existing_children or []
        self._fail_create = fail_create
        self._fail_patch = fail_patch
        self._fail_message_post_to = fail_message_post_to

    class _JsonResp:
        def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = ""

        def json(self) -> dict[str, Any]:
            return self._payload

        def raise_for_status(self) -> None:
            return None

    def _snapshot(self, session_id: str) -> dict[str, Any]:
        if session_id == PARENT_SESSION_ID:
            return {
                "id": session_id,
                "agent_id": "ag_orch",
                "agent_name": "orchestrator",
                "sub_agent_name": None,
                "parent_session_id": None,
                "created_at": 0,
                "workspace": None,
            }
        return {
            "id": session_id,
            "agent_id": "ag_orch",
            "agent_name": "reviewer",
            "sub_agent_name": "reviewer",
            "parent_session_id": PARENT_SESSION_ID,
            "created_at": 0,
            "workspace": None,
        }

    async def get(self, url: str, **kwargs: Any) -> Any:
        del kwargs
        path = url.rstrip("/")
        if path.endswith("/items"):
            return self._JsonResp({"data": [], "has_more": False})
        if path.endswith("/child_sessions"):
            return self._JsonResp({"data": self._existing_children})
        if "/v1/sessions/" in path:
            return self._JsonResp(self._snapshot(path.rsplit("/", 1)[-1]))
        return self._Response()

    async def post(self, url: str, **kwargs: Any) -> Any:
        path = url.rstrip("/")
        body = kwargs.get("json") or {}
        if path.endswith("/v1/sessions") or path == "/v1/sessions":
            if self._fail_create:
                raise RuntimeError("create exploded mid-flight")
            self.created_sessions.append(dict(body))
            return self._JsonResp({"id": FALLBACK_CHILD_SESSION_ID})
        if path.endswith("/policies"):
            self.policy_posts.append((path, dict(body)))
            return self._Response()
        if path.endswith("/events"):
            self.event_posts.append((path, dict(body)))
            if (
                self._fail_message_post_to is not None
                and f"/{self._fail_message_post_to}/" in path
                and body.get("type") == "message"
            ):
                return self._JsonResp({}, status_code=500)
            return self._Response()
        return self._Response()

    async def patch(self, url: str, **kwargs: Any) -> Any:
        self.patches.append((url, dict(kwargs.get("json") or {})))
        if self._fail_patch:
            return self._JsonResp({}, status_code=500)
        return self._Response()


def _reviewer_spec(fallback: object) -> AgentSpec:
    config: dict[str, Any] = {"harness": "codex-native"}
    if fallback is not None:
        config["fallback"] = fallback
    return AgentSpec(
        spec_version=1,
        name="reviewer",
        executor=ExecutorSpec(type="omnigent", config=config),
    )


def _parent_spec(reviewer: AgentSpec) -> AgentSpec:
    return AgentSpec(
        spec_version=1,
        name="orchestrator",
        executor=ExecutorSpec(type="omnigent", config={"harness": "claude-sdk"}),
        sub_agents=[reviewer],
    )


def _build_app(server: _FallbackServerClient, parent_spec: AgentSpec) -> Any:
    pm = _FakeProcessManager(_ScriptedHarnessClient([]))

    async def _resolver(agent_id: str, session_id: str | None = None) -> AgentSpec:
        del agent_id, session_id
        return parent_spec

    return create_runner_app(
        process_manager=pm,  # type: ignore[arg-type]
        spec_resolver=_resolver,
        server_client=server,  # type: ignore[arg-type]
    )


def _register_failed_dispatch(
    *,
    fallback_targets: tuple[FallbackTarget, ...],
    fallback_index: int = 0,
    fallback_history: list[str] | None = None,
    child_session_id: str = CHILD_SESSION_ID,
    cost_budget: dict[str, Any] | None = None,
) -> Any:
    """Seed the parent inbox and a dispatched work entry, as sys_session_send does."""
    runner_app._session_inboxes_ref[PARENT_SESSION_ID] = asyncio.Queue()
    return runner_app.register_subagent_work(
        parent_session_id=PARENT_SESSION_ID,
        child_session_id=child_session_id,
        agent="reviewer",
        title="review",
        message=REVIEW_MESSAGE,
        active_harness="codex-native",
        cost_budget=cost_budget,
        fallback_targets=fallback_targets,
        fallback_index=fallback_index,
        fallback_history=fallback_history,
    )


def _drain_parent_inbox() -> list[dict[str, Any]]:
    inbox = runner_app._session_inboxes_ref.get(PARENT_SESSION_ID)
    items: list[dict[str, Any]] = []
    if inbox is not None:
        while not inbox.empty():
            items.append(inbox.get_nowait())
    return items


async def _wait_for(condition: Any, *, timeout_s: float = 3.0) -> None:
    """Poll *condition* (a nullary callable) until truthy or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for background fallback task")


_PI_GEMINI_TARGETS = (
    FallbackTarget(harness="pi", model="google/gemini-2.5-pro", on=frozenset({"quota"})),
)


@pytest.mark.asyncio
async def test_quota_failure_redispatches_on_fallback_and_suppresses_delivery(
    _clean_subagent_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) A quota-classified native failure spawns the fallback child.

    The failed report must NOT reach the parent inbox; instead a fresh
    child session is created with the fallback harness+model overrides and
    the original message is replayed to it under the same work_id.
    """
    monkeypatch.setattr(
        "omnigent.onboarding.harness_install.missing_harness_cli", lambda harness: None
    )
    budget = {"max_cost_usd": 2.5}
    entry = _register_failed_dispatch(fallback_targets=_PI_GEMINI_TARGETS, cost_budget=budget)
    original_work_id = entry.work_id
    server = _FallbackServerClient()
    app = _build_app(server, _parent_spec(_reviewer_spec(None)))

    async with _runner_client(app) as client:
        resp = await client.post(
            f"/v1/sessions/{CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                # The forwarder's structured verdict: error_kind rides the payload.
                "data": {
                    "status": "failed",
                    "output": "turn ended with an error",
                    "error_kind": "quota",
                },
            },
        )
        assert resp.status_code == 204
        await _wait_for(lambda: server.created_sessions)
        await _wait_for(
            lambda: any(FALLBACK_CHILD_SESSION_ID in path for path, _body in server.event_posts)
        )

    # The intercepted failure never reached the parent inbox.
    assert _drain_parent_inbox() == []
    # Fresh child created with the fallback overrides.
    create_body = server.created_sessions[0]
    assert create_body["harness_override"] == "pi"
    assert create_body["model_override"] == "google/gemini-2.5-pro"
    assert create_body["sub_agent_name"] == "reviewer"
    assert create_body["parent_session_id"] == PARENT_SESSION_ID
    # Original message replayed verbatim to the new child.
    replayed = [
        body
        for path, body in server.event_posts
        if FALLBACK_CHILD_SESSION_ID in path and body.get("type") == "message"
    ]
    assert replayed, f"no message replayed to fallback child; posts={server.event_posts!r}"
    assert replayed[0]["data"]["content"][0]["text"] == REVIEW_MESSAGE
    # The superseded child's title slot was tombstoned (sys_session_close style).
    assert any(CHILD_SESSION_ID in url for url, _body in server.patches)
    # New work entry: same work_id, incremented index, provenance note armed.
    assert runner_app.get_subagent_work(CHILD_SESSION_ID) is None
    new_entry = runner_app.get_subagent_work(FALLBACK_CHILD_SESSION_ID)
    assert new_entry is not None
    assert new_entry.work_id == original_work_id
    assert new_entry.fallback_index == 1
    assert new_entry.message == REVIEW_MESSAGE
    assert new_entry.fallback_note == (
        "[fallback: completed on pi/google/gemini-2.5-pro after codex-native quota limit]"
    )
    assert new_entry.fallback_history == [
        "codex-native failed: quota — retrying on pi/google/gemini-2.5-pro"
    ]
    # The original cost budget travels with the work: the fallback child
    # got the same subagent_cost_budget policy attached, and the entry
    # retains it for any further hop.
    assert new_entry.cost_budget == budget
    policy_bodies = [body for path, body in server.policy_posts]
    assert policy_bodies and policy_bodies[0]["factory_params"] == budget
    # Stale handles resolve to the live replacement, and the in-flight
    # marker that kept the parent in "waiting" is gone.
    assert runner_app.resolve_superseded_subagent_child(CHILD_SESSION_ID) == (
        FALLBACK_CHILD_SESSION_ID
    )
    assert runner_app._pending_subagent_fallbacks == {}


@pytest.mark.asyncio
async def test_fallback_child_success_delivers_with_provenance_note(
    _clean_subagent_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) The fallback child's success flows to the inbox with provenance.

    Same work_id as the original dispatch, output prefixed with the
    fallback provenance note.
    """
    monkeypatch.setattr(
        "omnigent.onboarding.harness_install.missing_harness_cli", lambda harness: None
    )
    entry = _register_failed_dispatch(fallback_targets=_PI_GEMINI_TARGETS)
    original_work_id = entry.work_id
    server = _FallbackServerClient()
    app = _build_app(server, _parent_spec(_reviewer_spec(None)))

    async with _runner_client(app) as client:
        resp = await client.post(
            f"/v1/sessions/{CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                "data": {
                    "status": "failed",
                    "output": "You've hit your usage limit.",
                    # No error_kind: exercises output-text classification.
                },
            },
        )
        assert resp.status_code == 204
        await _wait_for(lambda: runner_app.get_subagent_work(FALLBACK_CHILD_SESSION_ID))

        # Now the fallback child completes its turn.
        resp2 = await client.post(
            f"/v1/sessions/{FALLBACK_CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                "data": {"status": "idle", "output": "review complete: LGTM"},
            },
        )
        assert resp2.status_code == 204

    items = _drain_parent_inbox()
    assert len(items) == 1
    payload = items[0]
    assert payload["status"] == "completed"
    assert payload["work_id"] == original_work_id
    assert payload["agent"] == "reviewer"
    assert payload["title"] == "review"
    assert payload["conversation_id"] == FALLBACK_CHILD_SESSION_ID
    assert payload["output"] == (
        "[fallback: completed on pi/google/gemini-2.5-pro after codex-native quota limit]"
        "\n\nreview complete: LGTM"
    )


@pytest.mark.asyncio
async def test_fallback_child_quota_failure_with_no_targets_left_delivers_history(
    _clean_subagent_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) A fallback child's own quota failure with no remaining target fails through.

    The loop guard: fallback_index already consumed the only target, so the
    failure is delivered (once), with the accumulated fallback history
    appended to the output.
    """
    monkeypatch.setattr(
        "omnigent.onboarding.harness_install.missing_harness_cli", lambda harness: None
    )
    history = ["codex-native failed: quota — retrying on pi/google/gemini-2.5-pro"]
    _register_failed_dispatch(
        fallback_targets=_PI_GEMINI_TARGETS,
        fallback_index=1,
        fallback_history=history,
        child_session_id=FALLBACK_CHILD_SESSION_ID,
    )
    server = _FallbackServerClient()
    app = _build_app(server, _parent_spec(_reviewer_spec(None)))

    async with _runner_client(app) as client:
        resp = await client.post(
            f"/v1/sessions/{FALLBACK_CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                "data": {"status": "failed", "output": "Rate limit exceeded"},
            },
        )
        assert resp.status_code == 204

    assert server.created_sessions == [], "no further fallback child may be spawned"
    items = _drain_parent_inbox()
    assert len(items) == 1
    assert items[0]["status"] == "failed"
    assert "Rate limit exceeded" in items[0]["output"]
    assert "[fallback history]" in items[0]["output"]
    assert "retrying on pi/google/gemini-2.5-pro" in items[0]["output"]


@pytest.mark.asyncio
async def test_generic_failure_with_quota_only_fallback_delivers_unchanged(
    _clean_subagent_registry: None,
) -> None:
    """(d) A generic failure does not trigger a quota-only fallback block."""
    _register_failed_dispatch(fallback_targets=_PI_GEMINI_TARGETS)
    server = _FallbackServerClient()
    app = _build_app(server, _parent_spec(_reviewer_spec(None)))

    async with _runner_client(app) as client:
        resp = await client.post(
            f"/v1/sessions/{CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                "data": {"status": "failed", "output": "disk full"},
            },
        )
        assert resp.status_code == 204

    assert server.created_sessions == []
    assert server.patches == []
    items = _drain_parent_inbox()
    assert len(items) == 1
    assert items[0]["status"] == "failed"
    assert items[0]["output"] == "disk full"


@pytest.mark.asyncio
async def test_missing_cli_fallback_target_is_skipped_with_history(
    _clean_subagent_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(e) A fallback target whose CLI is absent is skipped, not spawned.

    With no further targets the original failure is delivered, and the
    history records why the target was skipped.
    """

    def _missing(harness: str) -> Any:
        assert harness == "pi"
        return SimpleNamespace(binary="pi", package="@mariozechner/pi-coder", install_hint=None)

    monkeypatch.setattr("omnigent.onboarding.harness_install.missing_harness_cli", _missing)
    _register_failed_dispatch(fallback_targets=_PI_GEMINI_TARGETS)
    server = _FallbackServerClient()
    app = _build_app(server, _parent_spec(_reviewer_spec(None)))

    async with _runner_client(app) as client:
        resp = await client.post(
            f"/v1/sessions/{CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                "data": {
                    "status": "failed",
                    "output": QUOTA_OUTPUT,
                    "error_kind": "quota",
                },
            },
        )
        assert resp.status_code == 204
        inbox = runner_app._session_inboxes_ref[PARENT_SESSION_ID]
        await _wait_for(lambda: not inbox.empty())

    assert server.created_sessions == [], "an uninstallable target must not be spawned"
    assert server.patches == [], "an undispatchable block must not tombstone the child"
    items = _drain_parent_inbox()
    assert len(items) == 1
    payload = items[0]
    assert payload["status"] == "failed"
    assert QUOTA_OUTPUT in payload["output"]
    assert "[fallback history]" in payload["output"]
    assert "skipped" in payload["output"]
    assert "'pi' CLI on PATH" in payload["output"]
    assert runner_app._pending_subagent_fallbacks == {}


@pytest.mark.asyncio
async def test_fallback_dispatch_crash_still_delivers_original_failure(
    _clean_subagent_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash inside the fallback task must not vanish the work.

    The old entry is superseded before the async re-dispatch runs, so an
    unhandled exception (here: the create POST raising) would otherwise
    leave the parent waiting forever. The fail-safe delivers the ORIGINAL
    failure with a dispatch-failed note in the history.
    """
    monkeypatch.setattr(
        "omnigent.onboarding.harness_install.missing_harness_cli", lambda harness: None
    )
    _register_failed_dispatch(fallback_targets=_PI_GEMINI_TARGETS)
    server = _FallbackServerClient(fail_create=True)
    app = _build_app(server, _parent_spec(_reviewer_spec(None)))

    async with _runner_client(app) as client:
        resp = await client.post(
            f"/v1/sessions/{CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                "data": {"status": "failed", "output": QUOTA_OUTPUT, "error_kind": "quota"},
            },
        )
        assert resp.status_code == 204
        inbox = runner_app._session_inboxes_ref[PARENT_SESSION_ID]
        await _wait_for(lambda: not inbox.empty())

    items = _drain_parent_inbox()
    assert len(items) == 1
    payload = items[0]
    assert payload["status"] == "failed"
    assert QUOTA_OUTPUT in payload["output"]
    assert "[fallback dispatch failed:" in payload["output"]
    assert "RuntimeError" in payload["output"]
    assert runner_app._pending_subagent_fallbacks == {}


@pytest.mark.asyncio
async def test_continuation_send_does_not_arm_fallback(
    _clean_subagent_registry: None,
) -> None:
    """(H2) A continuation send to an existing child dispatches with fallback inert.

    The existing child holds the conversation history; replaying only the
    last message on a fresh child would present an incomplete answer as
    authoritative. So the entry stores no targets/message, and a later
    quota failure delivers as a plain failure.
    """
    from omnigent.runner.tool_dispatch import _execute_subagent_tool

    reviewer = _reviewer_spec(
        [{"harness": "pi", "model": "google/gemini-2.5-pro", "on": ["quota"]}]
    )
    parent_spec = _parent_spec(reviewer)
    server = _FallbackServerClient(
        existing_children=[
            {
                "id": CHILD_SESSION_ID,
                "tool": "reviewer",
                "session_name": "review",
                "labels": {},
                "busy": False,
            }
        ]
    )
    runner_app._session_inboxes_ref[PARENT_SESSION_ID] = asyncio.Queue()
    runner_app._session_agent_ids_ref[PARENT_SESSION_ID] = "ag_orch"

    handle = await _execute_subagent_tool(
        {"agent": "reviewer", "title": "review", "args": "continue: also check tests"},
        server_client=server,  # type: ignore[arg-type]
        conversation_id=PARENT_SESSION_ID,
        agent_spec=parent_spec,
        session_inbox=runner_app._session_inboxes_ref[PARENT_SESSION_ID],
    )
    assert not handle.startswith("Error:"), handle

    entry = runner_app.get_subagent_work(CHILD_SESSION_ID)
    assert entry is not None
    assert entry.fallback_targets == ()
    assert entry.message is None
    assert server.created_sessions == []  # continued, not re-created

    # A quota failure on the continuation turn now delivers as-is.
    app = _build_app(server, parent_spec)
    async with _runner_client(app) as client:
        resp = await client.post(
            f"/v1/sessions/{CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                "data": {"status": "failed", "output": QUOTA_OUTPUT, "error_kind": "quota"},
            },
        )
        assert resp.status_code == 204

    assert server.created_sessions == []
    items = _drain_parent_inbox()
    assert len(items) == 1
    assert items[0]["status"] == "failed"
    assert items[0]["output"] == QUOTA_OUTPUT


@pytest.mark.asyncio
async def test_fresh_spawn_arms_fallback_from_spec(
    _clean_subagent_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control for H2: a fresh spawn through sys_session_send arms fallback."""
    from omnigent.runner.tool_dispatch import _execute_subagent_tool

    monkeypatch.setattr(
        "omnigent.onboarding.harness_install.missing_harness_cli", lambda harness: None
    )
    reviewer = _reviewer_spec(
        [{"harness": "pi", "model": "google/gemini-2.5-pro", "on": ["quota"]}]
    )
    parent_spec = _parent_spec(reviewer)
    server = _FallbackServerClient()  # no existing children -> fresh create
    runner_app._session_inboxes_ref[PARENT_SESSION_ID] = asyncio.Queue()
    runner_app._session_agent_ids_ref[PARENT_SESSION_ID] = "ag_orch"

    handle = await _execute_subagent_tool(
        {"agent": "reviewer", "title": "review", "args": REVIEW_MESSAGE},
        server_client=server,  # type: ignore[arg-type]
        conversation_id=PARENT_SESSION_ID,
        agent_spec=parent_spec,
        session_inbox=runner_app._session_inboxes_ref[PARENT_SESSION_ID],
    )
    assert not handle.startswith("Error:"), handle

    entry = runner_app.get_subagent_work(FALLBACK_CHILD_SESSION_ID)
    assert entry is not None
    assert entry.message == REVIEW_MESSAGE
    assert entry.fallback_targets == _PI_GEMINI_TARGETS


@pytest.mark.asyncio
async def test_stale_handle_cancel_resolves_to_fallback_child(
    _clean_subagent_registry: None,
) -> None:
    """(M2) sys_cancel_task with the superseded handle reaches the live child."""
    from omnigent.runner.tool_dispatch import _cancel_subagent_task

    server = _FallbackServerClient()
    runner_app.register_subagent_work(
        parent_session_id=PARENT_SESSION_ID,
        child_session_id=FALLBACK_CHILD_SESSION_ID,
        agent="reviewer",
        title="review",
    )
    runner_app.record_subagent_supersession(CHILD_SESSION_ID, FALLBACK_CHILD_SESSION_ID)

    result = await _cancel_subagent_task(
        {"task_id": CHILD_SESSION_ID},
        conversation_id=PARENT_SESSION_ID,
        server_client=server,  # type: ignore[arg-type]
    )

    assert "no in-flight task" not in result
    interrupt_posts = [
        path for path, body in server.event_posts if body.get("type") == "interrupt"
    ]
    assert interrupt_posts and FALLBACK_CHILD_SESSION_ID in interrupt_posts[0]


@pytest.mark.asyncio
async def test_tombstone_failure_aborts_fallback_and_delivers_original(
    _clean_subagent_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(L1) A non-2xx tombstone PATCH aborts the fallback fail-safe.

    Creating the replacement would only hit the (parent, title) unique
    index, so no create is attempted and the original failure delivers.
    """
    monkeypatch.setattr(
        "omnigent.onboarding.harness_install.missing_harness_cli", lambda harness: None
    )
    _register_failed_dispatch(fallback_targets=_PI_GEMINI_TARGETS)
    server = _FallbackServerClient(fail_patch=True)
    app = _build_app(server, _parent_spec(_reviewer_spec(None)))

    async with _runner_client(app) as client:
        resp = await client.post(
            f"/v1/sessions/{CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                "data": {"status": "failed", "output": QUOTA_OUTPUT, "error_kind": "quota"},
            },
        )
        assert resp.status_code == 204
        inbox = runner_app._session_inboxes_ref[PARENT_SESSION_ID]
        await _wait_for(lambda: not inbox.empty())

    assert server.created_sessions == []
    items = _drain_parent_inbox()
    assert len(items) == 1
    assert QUOTA_OUTPUT in items[0]["output"]
    assert "[fallback dispatch failed:" in items[0]["output"]
    assert "tombstone" in items[0]["output"]


@pytest.mark.asyncio
async def test_failed_message_post_closes_replacement_child(
    _clean_subagent_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(L2) A replacement child whose message POST fails is closed, not leaked.

    Otherwise the zombie session squats on the (parent, title) slot and a
    later send adopts it. With no further target, the original failure
    then delivers with the skip recorded.
    """
    monkeypatch.setattr(
        "omnigent.onboarding.harness_install.missing_harness_cli", lambda harness: None
    )
    _register_failed_dispatch(fallback_targets=_PI_GEMINI_TARGETS)
    server = _FallbackServerClient(fail_message_post_to=FALLBACK_CHILD_SESSION_ID)
    app = _build_app(server, _parent_spec(_reviewer_spec(None)))

    async with _runner_client(app) as client:
        resp = await client.post(
            f"/v1/sessions/{CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                "data": {"status": "failed", "output": QUOTA_OUTPUT, "error_kind": "quota"},
            },
        )
        assert resp.status_code == 204
        inbox = runner_app._session_inboxes_ref[PARENT_SESSION_ID]
        await _wait_for(lambda: not inbox.empty())

    # Both tombstones happened: the original child (slot free) and the
    # undispatched replacement (no zombie left holding the slot).
    patched_ids = [url for url, _body in server.patches]
    assert any(CHILD_SESSION_ID in url for url in patched_ids)
    assert any(FALLBACK_CHILD_SESSION_ID in url for url in patched_ids)
    assert runner_app.get_subagent_work(FALLBACK_CHILD_SESSION_ID) is None
    items = _drain_parent_inbox()
    assert len(items) == 1
    assert items[0]["status"] == "failed"
    assert QUOTA_OUTPUT in items[0]["output"]
    assert "skipped" in items[0]["output"]


# ── E2E: consult_ppu-gated fallback targets ───────────────────────────


_PPU_GATED_PI_TARGETS = (
    FallbackTarget(
        harness="pi", model="google/gemini-2.5-pro", on=frozenset({"quota"}), consult_ppu=True
    ),
)


def _patch_consult_ppu(
    monkeypatch: pytest.MonkeyPatch, *, eligible: bool, reason: str = "eligible"
) -> list[tuple[bool, str]]:
    """Replace the oracle consult seam with a canned verdict; returns the call log."""
    calls: list[tuple[bool, str]] = []

    async def _consult() -> tuple[bool, str]:
        calls.append((eligible, reason))
        return (eligible, reason)

    monkeypatch.setattr("omnigent.runner.tool_dispatch._consult_ppu_oracle", _consult)
    return calls


@pytest.mark.asyncio
async def test_consult_ppu_eligible_oracle_allows_fallback_spawn(
    _clean_subagent_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consult_ppu target spawns normally when the oracle grants consent."""
    monkeypatch.setattr(
        "omnigent.onboarding.harness_install.missing_harness_cli", lambda harness: None
    )
    consults = _patch_consult_ppu(monkeypatch, eligible=True)
    _register_failed_dispatch(fallback_targets=_PPU_GATED_PI_TARGETS)
    server = _FallbackServerClient()
    app = _build_app(server, _parent_spec(_reviewer_spec(None)))

    async with _runner_client(app) as client:
        resp = await client.post(
            f"/v1/sessions/{CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                "data": {"status": "failed", "output": QUOTA_OUTPUT, "error_kind": "quota"},
            },
        )
        assert resp.status_code == 204
        await _wait_for(lambda: server.created_sessions)

    assert len(consults) == 1, "the oracle is consulted exactly once per candidate"
    assert _drain_parent_inbox() == []
    create_body = server.created_sessions[0]
    assert create_body["harness_override"] == "pi"
    assert create_body["model_override"] == "google/gemini-2.5-pro"


@pytest.mark.asyncio
async def test_consult_ppu_denied_skips_candidate_and_failure_stands(
    _clean_subagent_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Denied consent on the only target: no spawn, original failure delivers.

    Covers every denial flavor via the seam (enabled false / budget_ok
    false / health down / HTTP 500 / timeout / malformed body all reduce
    to an ineligible verdict — the flavors are unit-tested above).
    """
    monkeypatch.setattr(
        "omnigent.onboarding.harness_install.missing_harness_cli", lambda harness: None
    )
    consults = _patch_consult_ppu(
        monkeypatch, eligible=False, reason="ppu.anthropic.budget.budget_ok is not true"
    )
    _register_failed_dispatch(fallback_targets=_PPU_GATED_PI_TARGETS)
    server = _FallbackServerClient()
    app = _build_app(server, _parent_spec(_reviewer_spec(None)))

    async with _runner_client(app) as client:
        resp = await client.post(
            f"/v1/sessions/{CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                "data": {"status": "failed", "output": QUOTA_OUTPUT, "error_kind": "quota"},
            },
        )
        assert resp.status_code == 204
        inbox = runner_app._session_inboxes_ref[PARENT_SESSION_ID]
        await _wait_for(lambda: not inbox.empty())

    assert len(consults) == 1
    assert server.created_sessions == [], "a denied candidate must not be spawned"
    items = _drain_parent_inbox()
    assert len(items) == 1
    payload = items[0]
    assert payload["status"] == "failed"
    assert QUOTA_OUTPUT in payload["output"]
    assert "[fallback history]" in payload["output"]
    assert "PPU consent not granted" in payload["output"]
    assert "budget_ok is not true" in payload["output"]
    assert runner_app._pending_subagent_fallbacks == {}


@pytest.mark.asyncio
async def test_consult_ppu_denied_continues_to_next_candidate(
    _clean_subagent_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A denied gated target yields to the next (ungated) candidate."""
    monkeypatch.setattr(
        "omnigent.onboarding.harness_install.missing_harness_cli", lambda harness: None
    )
    consults = _patch_consult_ppu(
        monkeypatch, eligible=False, reason="ppu.anthropic.enabled is not true"
    )
    targets = (
        *_PPU_GATED_PI_TARGETS,
        FallbackTarget(harness="claude-native", on=frozenset({"quota"})),
    )
    _register_failed_dispatch(fallback_targets=targets)
    server = _FallbackServerClient()
    app = _build_app(server, _parent_spec(_reviewer_spec(None)))

    async with _runner_client(app) as client:
        resp = await client.post(
            f"/v1/sessions/{CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                "data": {"status": "failed", "output": QUOTA_OUTPUT, "error_kind": "quota"},
            },
        )
        assert resp.status_code == 204
        await _wait_for(lambda: server.created_sessions)

    # Only the gated first candidate consulted the oracle; the ungated
    # second candidate dispatched without one.
    assert len(consults) == 1
    assert _drain_parent_inbox() == []
    create_body = server.created_sessions[0]
    assert create_body["harness_override"] == "claude-native"
    new_entry = runner_app.get_subagent_work(FALLBACK_CHILD_SESSION_ID)
    assert new_entry is not None
    assert "PPU consent not granted" in "\n".join(new_entry.fallback_history)


@pytest.mark.asyncio
async def test_consult_ppu_absent_makes_no_oracle_call(
    _clean_subagent_registry: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ungated targets keep today's behavior: fallback spawns, oracle untouched."""
    monkeypatch.setattr(
        "omnigent.onboarding.harness_install.missing_harness_cli", lambda harness: None
    )
    consults = _patch_consult_ppu(monkeypatch, eligible=True)
    _register_failed_dispatch(fallback_targets=_PI_GEMINI_TARGETS)  # consult_ppu absent
    server = _FallbackServerClient()
    app = _build_app(server, _parent_spec(_reviewer_spec(None)))

    async with _runner_client(app) as client:
        resp = await client.post(
            f"/v1/sessions/{CHILD_SESSION_ID}/events",
            json={
                "type": "external_session_status",
                "data": {"status": "failed", "output": QUOTA_OUTPUT, "error_kind": "quota"},
            },
        )
        assert resp.status_code == 204
        await _wait_for(lambda: server.created_sessions)

    assert consults == [], "an ungated target must never consult the oracle"
    assert _drain_parent_inbox() == []
    assert server.created_sessions[0]["harness_override"] == "pi"
