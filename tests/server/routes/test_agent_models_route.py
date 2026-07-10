"""Tests for the agent model listing route (``GET /v1/agents/models``).

The route feeds the Harness Status dashboard: one row per registered
(``session_id IS NULL``) agent pairing the bundle-pinned model (parsed
from the stored bundle tar's root ``config.yaml``) with the active
per-agent operator override from the state file. Registered agents are
seeded directly through the agent/artifact stores because the app
fixture skips the lifespan seeding, mirroring ``test_builtin_agents``.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from omnigent.agent_model_overrides import AGENT_OVERRIDES_PATH_ENV
from omnigent.db.utils import generate_agent_id
from omnigent.server.bundles import bundle_location
from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from omnigent.stores.artifact_store.local import LocalArtifactStore
from tests.server.helpers import build_agent_bundle

_PINNED_MODEL = "databricks-claude-sonnet-4-6"
_OVERRIDE_MODEL = "databricks-claude-opus-4-8"


def _register_agent(
    db_uri: str,
    tmp_path: Path,
    name: str,
    *,
    executor: dict[str, object] | None = None,
    bundle: bytes | None = None,
) -> str:
    """Seed a registered agent whose bundle lives in the app's artifact store.

    Uses the same ``tmp_path / "artifacts"`` root the app fixture wires
    into its :class:`LocalArtifactStore`, so the route's agent-cache
    load reads the exact stored bundle tar.

    :returns: The new agent id.
    """
    if bundle is None:
        bundle = build_agent_bundle(name=name, executor=executor)
    agent_id = generate_agent_id()
    location = bundle_location(agent_id, bundle)
    LocalArtifactStore(str(tmp_path / "artifacts")).put(location, bundle)
    SqlAlchemyAgentStore(db_uri).create(agent_id, name=name, bundle_location=location)
    return agent_id


async def test_list_agent_models_empty(
    client: httpx.AsyncClient,
) -> None:
    """No registered agents → an empty paginated list."""
    resp = await client.get("/v1/agents/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == []
    assert body["has_more"] is False


async def test_pinned_model_extracted_from_bundle_tar(
    client: httpx.AsyncClient,
    db_uri: str,
    tmp_path: Path,
) -> None:
    """
    The pinned model is parsed from the stored bundle's root
    ``config.yaml`` — ``executor.model`` wins, ``llm.model`` backfills.
    """
    executor_pinned = _register_agent(
        db_uri,
        tmp_path,
        "executor-pinned",
        executor={"model": _PINNED_MODEL, "config": {"harness": "claude-sdk"}},
    )
    # ``build_agent_bundle`` writes ``llm.model = <name>``; with no
    # executor.model the parser backfills executor.model from it.
    llm_pinned = _register_agent(db_uri, tmp_path, "llm-pinned")

    resp = await client.get("/v1/agents/models")
    assert resp.status_code == 200
    by_id = {row["agent_id"]: row for row in resp.json()["data"]}
    assert by_id[executor_pinned]["pinned_model"] == _PINNED_MODEL
    assert by_id[llm_pinned]["pinned_model"] == "llm-pinned"


async def test_response_shape_and_override_pairing(
    client: httpx.AsyncClient,
    db_uri: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each row carries identity + pin + the validated active override."""
    overridden = _register_agent(
        db_uri,
        tmp_path,
        "overridden-agent",
        executor={"model": _PINNED_MODEL, "config": {"harness": "claude-sdk"}},
    )
    untouched = _register_agent(db_uri, tmp_path, "untouched-agent")
    state = tmp_path / "harness-status-state.json"
    state.write_text(
        json.dumps({"agent_overrides": {"overridden-agent": _OVERRIDE_MODEL}}),
        encoding="utf-8",
    )
    monkeypatch.setenv(AGENT_OVERRIDES_PATH_ENV, str(state))

    resp = await client.get("/v1/agents/models")
    assert resp.status_code == 200
    by_id = {row["agent_id"]: row for row in resp.json()["data"]}

    row = by_id[overridden]
    assert row["object"] == "agent.model_info"
    assert row["name"] == "overridden-agent"
    assert isinstance(row["created_at"], int)
    assert row["pinned_model"] == _PINNED_MODEL
    assert row["override_model"] == _OVERRIDE_MODEL

    assert by_id[untouched]["override_model"] is None


async def test_invalid_override_reads_as_null(
    client: httpx.AsyncClient,
    db_uri: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An invalid state-file value surfaces as ``null`` — the endpoint
    mirrors exactly what session creation would apply, never the raw
    garbage.
    """
    agent_id = _register_agent(db_uri, tmp_path, "garbled-agent")
    state = tmp_path / "harness-status-state.json"
    state.write_text(
        json.dumps({"agent_overrides": {"garbled-agent": "--not-a-model"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv(AGENT_OVERRIDES_PATH_ENV, str(state))

    resp = await client.get("/v1/agents/models")
    assert resp.status_code == 200
    by_id = {row["agent_id"]: row for row in resp.json()["data"]}
    assert by_id[agent_id]["override_model"] is None


async def test_unloadable_bundle_reports_null_pin(
    client: httpx.AsyncClient,
    db_uri: str,
    tmp_path: Path,
) -> None:
    """A corrupt stored bundle yields ``pinned_model: null``, not a 500."""
    agent_id = _register_agent(db_uri, tmp_path, "broken-bundle", bundle=b"this is not a tar.gz")
    resp = await client.get("/v1/agents/models")
    assert resp.status_code == 200
    by_id = {row["agent_id"]: row for row in resp.json()["data"]}
    assert by_id[agent_id]["pinned_model"] is None
    assert by_id[agent_id]["name"] == "broken-bundle"
