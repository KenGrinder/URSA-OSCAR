"""AI proxy module — Phase 5.

The AI proxy lives inside the API container and brokers LLM calls
between the chat panel UI and a user-configured provider (Claude API
or any OpenAI-compatible endpoint). All API keys stay server-side
(Decision 1); conversations are client-state only (Decision 5);
secrets are Fernet-encrypted at rest (Decision 7).

Public surface for the rest of the API container:
  - ``build_adapter(provider_id, config_dict, api_key) -> ProviderAdapter``
  - ``execute_tool(name, args, api_base_url) -> dict`` (re-exported from tools)
  - ``render_system_prompt(...)`` (re-exported from prompt)
"""
import os

from .providers.base import (  # noqa: F401
    AiMessage,
    AiStreamEvent,
    AiToolCall,
    ProviderAdapter,
    ProviderTestResult,
)
from .providers.claude import ClaudeAdapter  # noqa: F401
from .providers.openai_compat import OpenAiCompatAdapter  # noqa: F401
from .providers.presets import (  # noqa: F401
    PRESETS,
    ProviderPreset,
    build_auth_header,
    get_preset,
)


# 1.1.11 — provider-family defaults for the HTTP read timeout on
# streaming chat calls. Local LLMs get a generous 5-minute window
# because thinking-mode models (Qwen3, DeepSeek-R1, GPT-OSS, Gemma-4)
# on CPU can spend several minutes on the chain-of-thought before
# the first content token. Cloud APIs (Claude, OpenAI, Gemini, etc.)
# stream within a few seconds of connect; longer waits usually mean
# a real network problem worth surfacing rather than hiding.
DEFAULT_TIMEOUT_SECONDS_LOCAL = 300  # 5 minutes
DEFAULT_TIMEOUT_SECONDS_CLOUD = 120  # 2 minutes


def _effective_timeout(provider_id: str, configured: int | None) -> float:
    """Return the per-request HTTP read timeout in seconds. Operator
    override wins; otherwise pick the family default."""
    if configured is not None:
        return float(configured)
    if provider_id == "local":
        return float(DEFAULT_TIMEOUT_SECONDS_LOCAL)
    return float(DEFAULT_TIMEOUT_SECONDS_CLOUD)


# 1.1.14 — provider-family default for the completion output-token cap.
# The empty-answer trap: reasoning-mode local models spend a big slice of
# max_tokens on a hidden reasoning channel BEFORE the first answer token,
# so a too-small server default truncates mid-thought and the answer is
# blank on a 200. We MUST send a generous cap to local servers. Cloud
# providers already default to a large cap; sending a small one would
# REGRESS long cloud answers, so the cloud family default is None (omit
# max_tokens and let the provider decide). Claude keeps its own 4096.
DEFAULT_MAX_TOKENS_LOCAL = 4000


def _effective_max_tokens(provider_id: str, configured: int | None) -> int | None:
    """Return the completion output-token cap. Operator override wins.
    Otherwise: local gets a generous default (the reasoning-starve fix);
    every other provider gets None so the adapter omits max_tokens and
    the provider's own (large) default applies."""
    if configured is not None:
        return int(configured)
    if provider_id == "local":
        return DEFAULT_MAX_TOKENS_LOCAL
    return None


# 1.1.18 — env var for a local inference server that has turned on auth.
# Cross-fleet convention (the Lemonade box upgrade of ~2026-10-17 can now
# require a key). Deliberately NOT prefixed URSA_OSCAR_: Kevin sets ONE
# value across every sibling app that points at the shared box.
LOCAL_API_KEY_ENV = "LEMONADE_API_KEY"


def _resolve_local_api_key(provider_id: str, api_key: str | None) -> str | None:
    """Fill in the local server's API key from the environment when the
    operator hasn't stored one in Settings.

    Scoped to ``provider_id == "local"`` ON PURPOSE: this guarantees the
    env key can NEVER be attached to a cloud provider's request. It's not
    "unlikely" to leak — it's structurally unreachable, because any non-local
    provider returns before the env is even read (the "unreachable, not
    unlikely" canon). Precedence: a key the operator stored in Settings
    always wins; the env is only a fallback. Safe to set before the box
    turns auth on — an absent/empty env changes nothing.
    """
    if provider_id != "local":
        return api_key
    if api_key:
        return api_key
    env_key = os.environ.get(LOCAL_API_KEY_ENV, "").strip()
    return env_key or None


def build_adapter(provider_id: str, config_dict: dict, api_key: str | None):
    """Pick the right adapter class for a provider id and wire it with
    the operator's stored config + decrypted API key. Returns ``None``
    when the provider isn't in the registry — caller turns that into a
    400 (better than silently using a default)."""
    preset = get_preset(provider_id)
    if preset is None:
        return None

    # 1.1.18 — local-only env fallback for a box that now requires a key.
    # Resolved here, in the one place every caller funnels through, so the
    # scoping (local-only) holds for chat AND the test probe without either
    # endpoint needing to know about it.
    api_key = _resolve_local_api_key(provider_id, api_key)

    endpoint = (
        config_dict.get("endpoint_url")
        or preset.default_endpoint
    )
    model = config_dict.get("model") or (
        preset.default_models[0] if preset.default_models else ""
    )
    headers = build_auth_header(preset, api_key)
    timeout = _effective_timeout(
        provider_id, config_dict.get("timeout_seconds"),
    )
    max_tokens = _effective_max_tokens(
        provider_id, config_dict.get("max_output_tokens"),
    )
    # 1.1.19 — resolve the per-request thinking flag. ONLY the local
    # provider gets an explicit chat_template_kwargs.enable_thinking; every
    # cloud provider gets None so the field is omitted (they don't speak it
    # and some reject unknown body fields). Same local-only scoping as the
    # api-key and max-tokens resolvers above.
    enable_thinking = (
        bool(config_dict.get("local_thinking", False))
        if provider_id == "local" else None
    )

    cls = ClaudeAdapter if preset.adapter == "claude" else OpenAiCompatAdapter
    return cls(
        api_key=api_key,
        endpoint=endpoint,
        model=model,
        extra_headers=headers,
        timeout_seconds=timeout,
        max_output_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )
