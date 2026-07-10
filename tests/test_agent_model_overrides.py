"""Unit tests for ``omnigent/agent_model_overrides.py``.

The resolver feeds session creation, so its failure posture is the
contract under test: fail-open on the FILE (missing / unreadable /
unparseable / absent key → silent ``None``, pinned model wins) and
fail-safe on the VALUE (present-but-invalid → loud warning naming the
agent and the rejected value, then ``None``). A regression here either
silently drops operator overrides or lets a state-file typo crash a
spawn.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from omnigent.agent_model_overrides import (
    AGENT_OVERRIDES_PATH_ENV,
    agent_overrides_path,
    load_agent_overrides,
    resolve_agent_model_override,
)

_AGENT = "implementer"
_MODEL = "databricks-claude-opus-4-8"


def _write_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
) -> Path:
    """Write *content* as the state file and point the env var at it."""
    path = tmp_path / "harness-status-state.json"
    path.write_text(content, encoding="utf-8")
    monkeypatch.setenv(AGENT_OVERRIDES_PATH_ENV, str(path))
    return path


def test_agent_overrides_path_defaults_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the env var, the path is ``~/.omnigent/harness-status-state.json``."""
    monkeypatch.delenv(AGENT_OVERRIDES_PATH_ENV, raising=False)
    assert agent_overrides_path() == Path.home() / ".omnigent" / "harness-status-state.json"


def test_agent_overrides_path_honors_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-empty env var relocates the state file."""
    monkeypatch.setenv(AGENT_OVERRIDES_PATH_ENV, str(tmp_path / "alt.json"))
    assert agent_overrides_path() == tmp_path / "alt.json"


def test_resolve_returns_valid_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed entry resolves to its validated model id."""
    _write_state(tmp_path, monkeypatch, json.dumps({"agent_overrides": {_AGENT: _MODEL}}))
    assert resolve_agent_model_override(_AGENT) == _MODEL


def test_resolve_strips_whitespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The value passes through ``validate_model_override`` (which strips)."""
    _write_state(tmp_path, monkeypatch, json.dumps({"agent_overrides": {_AGENT: f"  {_MODEL}  "}}))
    assert resolve_agent_model_override(_AGENT) == _MODEL


def test_resolve_matches_per_agent_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lookup is exact per agent name — other agents keep their pinned model."""
    _write_state(
        tmp_path,
        monkeypatch,
        json.dumps({"agent_overrides": {_AGENT: _MODEL, "reviewer": "x-ai/grok-4.20"}}),
    )
    assert resolve_agent_model_override(_AGENT) == _MODEL
    assert resolve_agent_model_override("reviewer") == "x-ai/grok-4.20"
    assert resolve_agent_model_override("planner") is None


def test_missing_file_is_silent_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No state file → ``None`` with no warning (debug only)."""
    monkeypatch.setenv(AGENT_OVERRIDES_PATH_ENV, str(tmp_path / "absent.json"))
    with caplog.at_level(logging.WARNING):
        assert resolve_agent_model_override(_AGENT) is None
    assert caplog.records == []


def test_absent_key_is_silent_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A file with no entry for the agent → ``None`` with no warning."""
    _write_state(tmp_path, monkeypatch, json.dumps({"agent_overrides": {"other": _MODEL}}))
    with caplog.at_level(logging.WARNING):
        assert resolve_agent_model_override(_AGENT) is None
    assert caplog.records == []


@pytest.mark.parametrize(
    "content",
    [
        "not json at all {",
        json.dumps(["a", "list"]),
        json.dumps({"agent_overrides": "not-a-dict"}),
        json.dumps({"no_overrides_key": True}),
        "",
    ],
)
def test_unparseable_or_misshapen_file_is_silent_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    content: str,
) -> None:
    """Bad JSON / wrong shapes fail open: ``{}`` map, ``None`` resolve, no warning."""
    _write_state(tmp_path, monkeypatch, content)
    with caplog.at_level(logging.WARNING):
        assert load_agent_overrides() == {}
        assert resolve_agent_model_override(_AGENT) is None
    assert caplog.records == []


@pytest.mark.parametrize(
    "bad_value",
    [
        "--dangerously-skip-permissions",
        "model with spaces",
        "claude; rm -rf /",
        "",
        123,
        None,
        {"model": _MODEL},
    ],
)
def test_invalid_value_warns_and_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    bad_value: object,
) -> None:
    """
    A present-but-invalid value → ``None`` plus a warning that names
    the agent and the rejected value (the fail-safe half of the
    contract — an operator typo must be loud, never a crash).
    """
    _write_state(tmp_path, monkeypatch, json.dumps({"agent_overrides": {_AGENT: bad_value}}))
    with caplog.at_level(logging.WARNING):
        assert resolve_agent_model_override(_AGENT) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert _AGENT in message
    assert repr(bad_value) in message


def test_resolve_with_preloaded_overrides_skips_file_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller-supplied map is used verbatim (one consistent read per page)."""
    monkeypatch.setenv(AGENT_OVERRIDES_PATH_ENV, str(tmp_path / "absent.json"))
    assert resolve_agent_model_override(_AGENT, {_AGENT: _MODEL}) == _MODEL


def test_resolve_none_or_empty_agent_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``None``/empty agent names resolve to ``None`` without a file read."""
    _write_state(tmp_path, monkeypatch, json.dumps({"agent_overrides": {"": _MODEL}}))
    assert resolve_agent_model_override(None) is None
    assert resolve_agent_model_override("") is None


def test_fresh_read_sees_dashboard_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Every resolve re-reads the file — a dashboard Save (rewrite) is
    visible on the very next call, no restart or cache expiry needed.
    """
    path = _write_state(tmp_path, monkeypatch, json.dumps({"agent_overrides": {_AGENT: _MODEL}}))
    assert resolve_agent_model_override(_AGENT) == _MODEL
    path.write_text(
        json.dumps({"agent_overrides": {_AGENT: "databricks-claude-haiku-4-5"}}),
        encoding="utf-8",
    )
    assert resolve_agent_model_override(_AGENT) == "databricks-claude-haiku-4-5"
    path.unlink()
    assert resolve_agent_model_override(_AGENT) is None
