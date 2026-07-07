"""Cross-harness turn-error classification.

A terminally failed child turn carries only free-form error text (and
sometimes an HTTP-ish code). The cross-harness fallback engine needs a
coarse kind — ``"quota"`` / ``"auth"`` / ``"generic"`` — to decide whether
a failure should be re-dispatched on a fallback harness. This module is
dependency-free so both the runner (``omnigent.runner.app``) and the
native forwarders (e.g. ``omnigent.codex_native_forwarder``) can import
it without cycles.
"""

from __future__ import annotations

TURN_ERROR_KIND_QUOTA = "quota"
TURN_ERROR_KIND_AUTH = "auth"
TURN_ERROR_KIND_GENERIC = "generic"

# Every kind classify_turn_error_kind can return; also the vocabulary a
# fallback target's ``on:`` list is validated against.
TURN_ERROR_KINDS: frozenset[str] = frozenset(
    {TURN_ERROR_KIND_QUOTA, TURN_ERROR_KIND_AUTH, TURN_ERROR_KIND_GENERIC}
)

# Case-insensitive substrings that mark a quota / rate-limit failure.
# Covers Codex ("usage limit", "usage_limit_exceeded"), Anthropic/Claude
# ("hit your limit", the 5-hour-window wording, "rate limit"), and the
# generic provider spellings ("429", "quota exceeded", "rate-limited").
QUOTA_ERROR_FRAGMENTS: tuple[str, ...] = (
    "usage_limit_exceeded",
    "usage limit",
    "usage-limit",
    "hit your limit",
    "reached your limit",
    "rate limit",
    "rate-limit",
    "rate_limit",
    "quota exceeded",
    "quota_exceeded",
    "insufficient_quota",
    "out of quota",
    "too many requests",
    "resource_exhausted",
    "resource exhausted",
    "5-hour limit",
    "5 hour limit",
    "limit will reset",
    "limit resets",
    "429",
)

# Case-insensitive substrings that mark an authentication failure.
# Shared with the codex-native forwarder's auth classification (it was
# the original owner of this list). Surface-only recall-over-precision:
# a false positive only steers which fallback targets are considered.
AUTH_ERROR_FRAGMENTS: tuple[str, ...] = (
    "401",
    "403",
    "unauthorized",
    "authentication",
    "not logged in",
    "not authenticated",
    "log in",
    "login",
    "sign in",
    "re-authenticate",
    "reauthenticate",
    "credentials",
    "access token",
    "token expired",
    "expired token",
    "session expired",
    "api key",
)


def classify_turn_error_kind(message: str | None, code: str | int | None = None) -> str:
    """
    Classify a terminal turn failure as ``"quota"``, ``"auth"``, or ``"generic"``.

    A structured *code* wins over text matching: 429 is quota, 401/403 is
    auth. Otherwise the *message* is scanned for quota fragments first
    (a quota message like "hit your usage limit — sign in to upgrade"
    must not be mistaken for an auth failure), then auth fragments.

    :param message: Free-form error text from the failed turn, e.g.
        ``"You've hit your usage limit."``. ``None`` when unavailable.
    :param code: Optional structured error/HTTP code, e.g. ``429`` or
        ``"429"``.
    :returns: One of :data:`TURN_ERROR_KINDS`.
    """
    code_text = str(code).strip() if code is not None else ""
    if code_text == "429":
        return TURN_ERROR_KIND_QUOTA
    if code_text in ("401", "403"):
        return TURN_ERROR_KIND_AUTH
    lowered = (message or "").lower()
    if not lowered:
        return TURN_ERROR_KIND_GENERIC
    if any(fragment in lowered for fragment in QUOTA_ERROR_FRAGMENTS):
        return TURN_ERROR_KIND_QUOTA
    if any(fragment in lowered for fragment in AUTH_ERROR_FRAGMENTS):
        return TURN_ERROR_KIND_AUTH
    return TURN_ERROR_KIND_GENERIC
