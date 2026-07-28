"""
Provider routing — parse model strings and resolve adapters.

Model strings use ``"provider/model-name"`` format, e.g.
``"anthropic/claude-sonnet-4-20250514"``. If no provider prefix
is given, defaults to ``"openai"``.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnigent.errors import ErrorCode, OmnigentError

# Known providers and their default base URLs.
# API keys come from connection_params at call time (llm.connection config),
# not from environment variables. Providers that require connection_params
# for their base URL (Bedrock, Vertex, Databricks) have None here.
PROVIDER_CONFIGS: dict[str, str | None] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "bedrock": None,
    "vertex": None,
    "databricks": None,
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "xai": "https://api.x.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
    "moonshot": "https://api.moonshot.cn/v1",
    # Fireworks is OpenAI-compatible. Both spellings route: ``fireworks_ai``
    # is the catalog/onboarding id (matching FIREWORKS_AI_API_KEY), while
    # ``fireworks`` is what a model string naturally reads as
    # ("fireworks/kimi-k3").
    "fireworks_ai": "https://api.fireworks.ai/inference/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
}

_DEFAULT_PROVIDER = "openai"


@dataclass
class RoutedModel:
    """
    A parsed model string split into provider and model name.

    :param provider: The provider identifier, e.g. ``"anthropic"``.
    :param model: The model name without prefix, e.g.
        ``"claude-sonnet-4-20250514"``.
    """

    provider: str
    model: str


# Fireworks ids are fully qualified resource paths
# ("accounts/fireworks/models/kimi-k3") and its API wants that whole path in
# the request's ``model`` field. Splitting on the first "/" would read
# "accounts" as the provider, so the prefix is recognised before the generic
# split and the id is kept intact.
_FIREWORKS_MODEL_PATH_PREFIX = "accounts/"
_FIREWORKS_MODEL_PATH_MARKER = "/models/"
_FIREWORKS_PROVIDER = "fireworks_ai"


def parse_model_string(model: str) -> RoutedModel:
    """
    Parse a ``"provider/model-name"`` string into its components.

    If no ``"/"`` is present, the provider defaults to ``"openai"``
    for backward compatibility. A bare Fireworks resource path
    (``"accounts/<account>/models/<name>"``) is recognised as Fireworks and
    kept whole, since that full path is the id its API expects.

    :param model: The model string, e.g.
        ``"anthropic/claude-sonnet-4-20250514"``, ``"gpt-5.4"``, or
        ``"accounts/fireworks/models/kimi-k3"``.
    :returns: A :class:`RoutedModel` with ``provider`` and ``model``.
    :raises OmnigentError: If the provider prefix is not recognized.
    """
    if model.startswith(_FIREWORKS_MODEL_PATH_PREFIX) and (_FIREWORKS_MODEL_PATH_MARKER in model):
        return RoutedModel(provider=_FIREWORKS_PROVIDER, model=model)

    if "/" in model:
        provider, model_name = model.split("/", 1)
    else:
        provider = _DEFAULT_PROVIDER
        model_name = model

    if provider not in PROVIDER_CONFIGS:
        raise OmnigentError(
            f"Unknown provider {provider!r}. Known providers: {sorted(PROVIDER_CONFIGS)}",
            code=ErrorCode.INVALID_INPUT,
        )

    # A Fireworks model given with an explicit provider prefix still needs its
    # full resource path sent to the API, so restore what the split removed.
    if provider in ("fireworks", _FIREWORKS_PROVIDER) and model_name.startswith(
        _FIREWORKS_MODEL_PATH_PREFIX
    ):
        return RoutedModel(provider=_FIREWORKS_PROVIDER, model=model_name)

    return RoutedModel(provider=provider, model=model_name)


# Maps model-string prefixes to the harness that should run them.
# Used by :func:`infer_harness_from_model` when a spec doesn't name
# a harness explicitly.
_HARNESS_FOR_MODEL_PREFIX: dict[str, str] = {
    "databricks-claude-": "claude-sdk",
    "anthropic/claude-": "claude-sdk",
    "databricks-gpt-": "openai-agents",
    "openai/gpt-": "openai-agents",
    "gpt-": "openai-agents",
    # xAI is OpenAI-compatible; provider prefix required (bare grok- defaults to openai).
    "xai/grok-": "openai-agents",
    # Fireworks serves open-weight models over the OpenAI-compatible wire, so
    # they run on the multi-model openai-agents harness. Its own ids are fully
    # qualified ("accounts/fireworks/models/kimi-k3"); the provider-prefixed
    # spellings cover the shorthand a spec is more likely to carry.
    "accounts/fireworks/models/": "openai-agents",
    "fireworks_ai/": "openai-agents",
    "fireworks/": "openai-agents",
}


def infer_harness_from_model(model: str) -> str:
    """
    Return the harness name implied by a model string, or ``""``
    when no prefix in :data:`_HARNESS_FOR_MODEL_PREFIX` matches.

    An empty string has the same meaning as "no harness declared"
    in :class:`~omnigent.datamodel.ExecutorSpec` — callers that
    receive it propagate it unchanged and let the downstream
    validator surface a "harness required" error if one is needed.

    :param model: Model string from the spec's ``llm.model`` or
        ``executor.model`` field, e.g. ``"databricks-claude-sonnet-4"``.
    :returns: A harness name such as ``"claude-sdk"`` or
        ``"openai-agents"``, or ``""`` when *model* is unrecognised.
    """
    for prefix, harness in _HARNESS_FOR_MODEL_PREFIX.items():
        if model.startswith(prefix):
            return harness
    return ""
