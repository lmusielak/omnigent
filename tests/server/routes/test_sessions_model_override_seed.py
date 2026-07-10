"""Session-create seeding of operator model overrides (state file).

``POST /v1/sessions`` is the single choke-point where the Harness
Status dashboard's per-agent overrides
(``~/.omnigent/harness-status-state.json``, relocatable via
``OMNIGENT_OVERRIDES_PATH``) enter a session: when the create names no
explicit ``model_override``, the effective agent name (sub-agent name
for dispatched children, agent name otherwise) is looked up in the
state file and the validated value is persisted as the session's
``model_override`` — from where the EXISTING delivery plumbing
(``HARNESS_<H>_MODEL`` spawn env / native ``--model`` argv) applies it.

These tests pin the seeding contract on both create paths (JSON
agent-id and multipart bundle upload), the explicit-override
precedence, the per-agent name matching for sub-agent dispatches, and
the fail-open/fail-safe fallback behavior.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from omnigent.agent_model_overrides import AGENT_OVERRIDES_PATH_ENV
from tests.server.helpers import build_agent_bundle, create_test_agent

_OVERRIDE_MODEL = "databricks-claude-opus-4-8"
_SUB_OVERRIDE_MODEL = "databricks-claude-haiku-4-5"


def _write_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
) -> Path:
    """Persist *overrides* as the state file and point the env var at it."""
    path = tmp_path / "harness-status-state.json"
    path.write_text(json.dumps({"agent_overrides": overrides}), encoding="utf-8")
    monkeypatch.setenv(AGENT_OVERRIDES_PATH_ENV, str(path))
    return path


async def _create_json_session(
    client: httpx.AsyncClient,
    agent_id: str,
    **extra: Any,
) -> dict[str, Any]:
    """Create a session via the JSON path and return its snapshot."""
    resp = await client.post("/v1/sessions", json={"agent_id": agent_id, **extra})
    assert resp.status_code == 201, f"session create failed: {resp.text}"
    return resp.json()


async def test_json_create_seeds_override_from_state_file(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching state-file entry becomes the session's model_override."""
    agent = await create_test_agent(client, name="seeded-agent")
    _write_overrides(tmp_path, monkeypatch, {"seeded-agent": _OVERRIDE_MODEL})
    session = await _create_json_session(client, agent["id"])
    assert session["model_override"] == _OVERRIDE_MODEL


async def test_explicit_model_override_beats_state_file(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit create-body override wins; the file never shadows it."""
    agent = await create_test_agent(client, name="pinned-agent")
    _write_overrides(tmp_path, monkeypatch, {"pinned-agent": _OVERRIDE_MODEL})
    session = await _create_json_session(
        client, agent["id"], model_override="databricks-claude-sonnet-4-6"
    )
    assert session["model_override"] == "databricks-claude-sonnet-4-6"


async def test_absent_entry_leaves_pinned_model(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No entry for this agent name → no session override (pinned model wins)."""
    agent = await create_test_agent(client, name="unlisted-agent")
    _write_overrides(tmp_path, monkeypatch, {"someone-else": _OVERRIDE_MODEL})
    session = await _create_json_session(client, agent["id"])
    assert session["model_override"] is None


async def test_missing_file_is_silent_fallback(
    client: httpx.AsyncClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing state file never blocks or warns — the spawn proceeds pinned."""
    agent = await create_test_agent(client, name="no-file-agent")
    with caplog.at_level(logging.WARNING, logger="omnigent.agent_model_overrides"):
        session = await _create_json_session(client, agent["id"])
    assert session["model_override"] is None
    assert caplog.records == []


async def test_invalid_value_warns_and_falls_back(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    A flag-shaped state-file value is rejected loudly (agent + value in
    the warning) and the session is still created on the pinned model —
    an operator typo must never crash a spawn.
    """
    agent = await create_test_agent(client, name="typo-agent")
    _write_overrides(tmp_path, monkeypatch, {"typo-agent": "--dangerously-skip-permissions"})
    with caplog.at_level(logging.WARNING, logger="omnigent.agent_model_overrides"):
        session = await _create_json_session(client, agent["id"])
    assert session["model_override"] is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "typo-agent" in message
    assert "--dangerously-skip-permissions" in message


async def test_sub_agent_create_matches_sub_agent_name(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A dispatched child (``sub_agent_name`` set) is keyed by ITS name,
    not the parent's — the bound ``agent_id`` resolves to the parent.
    """
    # ``build_agent_bundle`` writes sub-agent configs without an
    # ``executor`` block, which spec validation rejects for the default
    # omnigent executor — build the parent bundle inline with a
    # harnessed sub-agent (mirrors the child-sessions test builder).
    sub_config = {
        "spec_version": 1,
        "name": "override-child",
        "llm": {"model": "override-child", "connection": {"api_key": "test-key"}},
        "executor": {"config": {"harness": "claude-sdk"}},
    }
    parent_config = {
        "spec_version": 1,
        "name": "override-parent",
        "llm": {"model": "override-parent", "connection": {"api_key": "test-key"}},
        "executor": {"config": {"harness": "claude-sdk"}},
        "tools": {"agents": ["override-child"]},
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for member_name, member_config in (
            ("config.yaml", parent_config),
            ("agents/override-child/config.yaml", sub_config),
        ):
            member_bytes = yaml.dump(member_config).encode()
            info = tarfile.TarInfo(name=member_name)
            info.size = len(member_bytes)
            tf.addfile(info, io.BytesIO(member_bytes))
    bundle = buf.getvalue()
    resp = await client.post(
        "/v1/sessions",
        data={"metadata": json.dumps({})},
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201, f"parent create failed: {resp.text}"
    parent_session_id = resp.json()["session_id"]
    parent_agent = await client.get(f"/v1/sessions/{parent_session_id}/agent")
    assert parent_agent.status_code == 200
    _write_overrides(
        tmp_path,
        monkeypatch,
        {
            "override-parent": _OVERRIDE_MODEL,
            "override-child": _SUB_OVERRIDE_MODEL,
        },
    )
    child = await _create_json_session(
        client,
        parent_agent.json()["id"],
        parent_session_id=parent_session_id,
        sub_agent_name="override-child",
        title="override-child:probe",
    )
    assert child["model_override"] == _SUB_OVERRIDE_MODEL


async def test_multipart_create_seeds_override_by_spec_name(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bundled (``omnigent run``) entry path seeds by the root spec name."""
    _write_overrides(tmp_path, monkeypatch, {"bundled-entry": _OVERRIDE_MODEL})
    bundle = build_agent_bundle(name="bundled-entry")
    resp = await client.post(
        "/v1/sessions",
        data={"metadata": json.dumps({})},
        files={"bundle": ("agent.tar.gz", bundle, "application/gzip")},
    )
    assert resp.status_code == 201, f"bundled create failed: {resp.text}"
    snapshot = await client.get(f"/v1/sessions/{resp.json()['session_id']}")
    assert snapshot.status_code == 200
    assert snapshot.json()["model_override"] == _OVERRIDE_MODEL


async def test_dashboard_save_affects_next_session_without_restart(
    client: httpx.AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The file is read fresh per create — a Save applies to the next session."""
    agent = await create_test_agent(client, name="fresh-read-agent")
    first = await _create_json_session(client, agent["id"])
    assert first["model_override"] is None
    _write_overrides(tmp_path, monkeypatch, {"fresh-read-agent": _OVERRIDE_MODEL})
    second = await _create_json_session(client, agent["id"])
    assert second["model_override"] == _OVERRIDE_MODEL
