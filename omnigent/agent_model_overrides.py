"""Operator-managed per-agent model overrides (Harness Status state file).

The Harness Status dashboard persists per-agent model overrides to
``~/.omnigent/harness-status-state.json`` (atomic tmp+rename writes) as::

    {"agent_overrides": {"<agent-name>": "<model-id>"}, ...}

This module is the read side of that contract. Session creation calls
:func:`resolve_agent_model_override` with the effective agent name (the
root spec name for an entry agent, ``sub_agent_name`` for a dispatched
sub-agent) and, when the create request names no explicit model, seeds
the session's persisted ``model_override`` from the matching entry.
Delivery then rides the EXISTING per-session override plumbing —
``HARNESS_<H>_MODEL`` in the SDK spawn env and ``--model`` argv at
native-CLI terminal launch — so no new delivery path exists to drift.

The file is read FRESH on every call (no module-level cache): a
dashboard Save affects the next session without any server restart.

Failure posture:

- **Fail-open on the file.** A missing, unreadable, or unparseable
  file — or an absent ``agent_overrides`` key / agent entry — resolves
  to ``None`` (the spec-pinned model wins) with a debug log only.
- **Fail-safe on the value.** A present-but-invalid model id logs a
  loud warning naming the agent and the rejected value, then resolves
  to ``None``. This function never raises — an operator typo in the
  state file must never crash a spawn.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

from omnigent.model_override import validate_model_override

_logger = logging.getLogger(__name__)

# Env var that relocates the state file (tests, non-default layouts).
AGENT_OVERRIDES_PATH_ENV = "OMNIGENT_OVERRIDES_PATH"

# The Harness Status dashboard's persisted-state file.
DEFAULT_AGENT_OVERRIDES_FILENAME = "harness-status-state.json"

# Top-level key carrying the per-agent override map.
_AGENT_OVERRIDES_KEY = "agent_overrides"


def agent_overrides_path() -> Path:
    """
    Return the state-file path, honoring :data:`AGENT_OVERRIDES_PATH_ENV`.

    :returns: The path from the env var when set and non-empty,
        otherwise ``~/.omnigent/harness-status-state.json``.
    """
    env_path = os.environ.get(AGENT_OVERRIDES_PATH_ENV, "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".omnigent" / DEFAULT_AGENT_OVERRIDES_FILENAME


def load_agent_overrides() -> Mapping[str, object]:
    """
    Read the state file fresh and return its raw ``agent_overrides`` map.

    Values are returned unvalidated (``object``) so
    :func:`resolve_agent_model_override` can distinguish "absent"
    (silent) from "present but invalid" (loud warning). Fail-open: any
    file-level problem — missing file, I/O error, bad JSON, wrong
    top-level shape — returns an empty map with a debug log only.

    :returns: The ``agent_overrides`` mapping, e.g.
        ``{"implementer": "databricks-claude-opus-4-8"}``, or ``{}``.
    """
    path = agent_overrides_path()
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _logger.debug("agent-overrides state file %s not found; no overrides", path)
        return {}
    except OSError:
        _logger.debug(
            "agent-overrides state file %s unreadable; no overrides",
            path,
            exc_info=True,
        )
        return {}
    try:
        parsed = json.loads(raw_text)
    except ValueError:
        _logger.debug(
            "agent-overrides state file %s is not valid JSON; no overrides",
            path,
            exc_info=True,
        )
        return {}
    if not isinstance(parsed, dict):
        _logger.debug(
            "agent-overrides state file %s top level is %s, expected object; no overrides",
            path,
            type(parsed).__name__,
        )
        return {}
    overrides = parsed.get(_AGENT_OVERRIDES_KEY)
    if not isinstance(overrides, dict):
        _logger.debug(
            "agent-overrides state file %s has no %r object; no overrides",
            path,
            _AGENT_OVERRIDES_KEY,
        )
        return {}
    return {str(name): value for name, value in overrides.items()}


def resolve_agent_model_override(
    agent_name: str | None,
    overrides: Mapping[str, object] | None = None,
) -> str | None:
    """
    Return the validated operator model override for *agent_name*.

    :param agent_name: Effective agent name to look up — the root spec
        name for an entry agent (e.g. ``"analytics-supervisor"``) or
        the dispatched sub-agent's name (e.g. ``"implementer"``).
        ``None``/empty resolves to ``None``.
    :param overrides: Optional pre-loaded map from
        :func:`load_agent_overrides`, for callers resolving several
        agents against one consistent file read. ``None`` reads the
        file fresh.
    :returns: The validated model id, or ``None`` when no usable
        override exists (absent → silent; invalid → warning). Never
        raises.
    """
    if not agent_name:
        return None
    if overrides is None:
        overrides = load_agent_overrides()
    if agent_name not in overrides:
        _logger.debug("no agent-override entry for %r; using pinned model", agent_name)
        return None
    value = overrides[agent_name]
    if not isinstance(value, str):
        _logger.warning(
            "agent-override for %r rejected: value %r is not a string; "
            "falling back to the pinned model",
            agent_name,
            value,
        )
        return None
    try:
        return validate_model_override(value)
    except ValueError as exc:
        _logger.warning(
            "agent-override for %r rejected: invalid model id %r (%s); "
            "falling back to the pinned model",
            agent_name,
            value,
            exc,
        )
        return None
