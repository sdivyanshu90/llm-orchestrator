from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, Mapping, Optional


class Provider(str, Enum):
    """Supported OpenAI-compatible backends."""

    OLLAMA = "ollama"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    HF_TGI = "hf_tgi"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single model invocation profile.

    Attributes:
        name: Provider-specific model identifier.
        provider: Backend provider enum.
        base_url: OpenAI-compatible API base URL.
        api_key: API key or placeholder for local backends.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        repeat_penalty: Repetition penalty mapped to frequency penalty.
        max_tokens: Maximum completion tokens for the request.
        context_window: Approximate context window size used for budgeting.
    """

    name: str
    provider: Provider
    base_url: str
    api_key: str
    temperature: float
    top_p: float
    repeat_penalty: float
    max_tokens: int
    context_window: int


DEFAULT_PROVIDER = Provider.OLLAMA
STAGE_NAMES = ("decomposer", "reasoner", "critic", "synthesizer")

DEFAULT_PROVIDER_BASE_URLS: Dict[Provider, str] = {
    Provider.OLLAMA: "http://localhost:11434/v1",
    Provider.GROQ: "https://api.groq.com/openai/v1",
    Provider.OPENROUTER: "https://openrouter.ai/api/v1",
    Provider.HF_TGI: "http://localhost:8080/v1",
    Provider.CUSTOM: "http://localhost:8000/v1",
}

DEFAULT_PROVIDER_API_KEYS: Dict[Provider, str] = {
    Provider.OLLAMA: "ollama",
    Provider.GROQ: "",
    Provider.OPENROUTER: "",
    Provider.HF_TGI: "hf_tgi",
    Provider.CUSTOM: "custom",
}

PROVIDER_ENV_KEYS: Dict[Provider, str] = {
    Provider.OLLAMA: "OLLAMA",
    Provider.GROQ: "GROQ",
    Provider.OPENROUTER: "OPENROUTER",
    Provider.HF_TGI: "HF_TGI",
    Provider.CUSTOM: "CUSTOM",
}


def _env_int(name: str, default: int, env: Mapping[str, str]) -> int:
    """Read an integer environment variable with fallback.

    Args:
        name: Environment variable name.
        default: Fallback value when unset or invalid.
        env: Environment mapping.

    Returns:
        Parsed integer or fallback.
    """
    raw_value = env.get(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        return default


def _parse_provider(raw_value: Optional[str]) -> Provider:
    """Parse a provider string into the Provider enum.

    Args:
        raw_value: Raw provider name.

    Returns:
        Parsed provider enum.

    Raises:
        ValueError: If the provided name is unsupported.
    """
    if raw_value is None or raw_value == "":
        return DEFAULT_PROVIDER
    return Provider(raw_value.strip().lower())


def _provider_base_url(provider: Provider, env: Mapping[str, str]) -> str:
    """Resolve the API base URL for a provider.

    Args:
        provider: Selected provider.
        env: Environment mapping.

    Returns:
        Base URL for the provider.
    """
    prefix = PROVIDER_ENV_KEYS[provider]
    return env.get(f"{prefix}_BASE_URL", DEFAULT_PROVIDER_BASE_URLS[provider])


def _provider_api_key(provider: Provider, env: Mapping[str, str]) -> str:
    """Resolve the API key for a provider.

    Args:
        provider: Selected provider.
        env: Environment mapping.

    Returns:
        API key or placeholder token.
    """
    prefix = PROVIDER_ENV_KEYS[provider]
    return env.get(f"{prefix}_API_KEY", DEFAULT_PROVIDER_API_KEYS[provider])


def _base_stage_configs() -> Dict[str, ModelConfig]:
    """Create immutable default configs for all pipeline stages.

    Returns:
        Default stage configuration mapping.
    """
    return {
        "decomposer": ModelConfig(
            name="qwen3:14b",
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.1,
            max_tokens=800,
            context_window=32768,
        ),
        "reasoner": ModelConfig(
            name="qwen3:14b",
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.4,
            top_p=0.92,
            repeat_penalty=1.05,
            max_tokens=1200,
            context_window=32768,
        ),
        "critic": ModelConfig(
            name="deepseek-r1:14b",
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.1,
            top_p=0.85,
            repeat_penalty=1.15,
            max_tokens=1000,
            context_window=32768,
        ),
        "synthesizer": ModelConfig(
            name="qwen3:14b",
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.0,
            max_tokens=2000,
            context_window=32768,
        ),
    }


STAGE_CONFIGS = _base_stage_configs()

MAX_REVISION_CYCLES = _env_int("MAX_REVISION_CYCLES", 2, os.environ)
MAX_REASONING_STEPS = 7
MIN_REASONING_STEPS = 2
MAX_TOTAL_TOKENS_PER_RUN = _env_int("MAX_TOTAL_TOKENS", 20000, os.environ)
MAX_WALL_TIME_SECONDS = _env_int("MAX_WALL_TIME_SECONDS", 300, os.environ)
TOKEN_BUDGET_WARNING_PCT = 0.80


def get_default_provider(env: Optional[Mapping[str, str]] = None) -> Provider:
    """Resolve the default provider from environment.

    Args:
        env: Optional environment mapping.

    Returns:
        Selected provider.
    """
    source = env if env is not None else os.environ
    return _parse_provider(source.get("PROVIDER"))


def load_stage_configs(
    provider_override: Optional[Provider] = None,
    model_override: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, ModelConfig]:
    """Build effective stage configs with provider and model overrides.

    Args:
        provider_override: Explicit provider to use for all stages.
        model_override: Explicit model name to use for all stages.
        env: Optional environment mapping.

    Returns:
        Fresh stage configuration mapping.
    """
    source = env if env is not None else os.environ
    selected_provider = provider_override or get_default_provider(source)
    stage_configs: Dict[str, ModelConfig] = {}

    for stage_name, base_config in STAGE_CONFIGS.items():
        resolved_model_name = model_override or source.get(
            f"{stage_name.upper()}_MODEL", base_config.name
        )
        resolved_provider = selected_provider
        stage_configs[stage_name] = replace(
            base_config,
            name=resolved_model_name,
            provider=resolved_provider,
            base_url=_provider_base_url(resolved_provider, source),
            api_key=_provider_api_key(resolved_provider, source),
        )

    return stage_configs


def describe_stage_models(stage_configs: Mapping[str, ModelConfig]) -> Dict[str, str]:
    """Create a simple stage-to-model label mapping for UI display.

    Args:
        stage_configs: Effective stage configs.

    Returns:
        Mapping of stage name to user-facing provider/model string.
    """
    return {
        stage_name: f"{config.provider.value}:{config.name}"
        for stage_name, config in stage_configs.items()
    }