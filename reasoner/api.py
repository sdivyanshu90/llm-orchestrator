from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import tiktoken
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from pydantic import BaseModel, ConfigDict

from reasoner.config import (
    MAX_TOTAL_TOKENS_PER_RUN,
    MAX_WALL_TIME_SECONDS,
    STAGE_CONFIGS,
    TOKEN_BUDGET_WARNING_PCT,
    ModelConfig,
)
from reasoner.models import ReasoningChain


LOG_PATH = Path(__file__).resolve().parent.parent / "reasoning.log"
FENCED_JSON_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# Best-effort pricing for display. Local backends default to zero cost.
MODEL_PRICING_USD_PER_MILLION: Dict[str, Tuple[float, float]] = {
    "ollama": (0.0, 0.0),
    "hf_tgi": (0.0, 0.0),
    "custom": (0.0, 0.0),
    "groq": (0.0, 0.0),
    "openrouter": (0.0, 0.0),
}


class LLMResponse(BaseModel):
    """Normalized response payload from any OpenAI-compatible backend."""

    model_config = ConfigDict(extra="forbid")

    content: str
    tokens_in: int
    tokens_out: int
    model: str
    provider: str


class LLMError(RuntimeError):
    """Raised when a provider call fails irrecoverably."""

    def __init__(self, provider: str, model: str, message: str) -> None:
        super().__init__(f"[{provider}:{model}] {message}")
        self.provider = provider
        self.model = model


class JSONExtractionError(ValueError):
    """Raised when JSON extraction fails after all recovery strategies."""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


class BudgetExceededError(RuntimeError):
    """Raised when token budget constraints are exceeded."""


class LLMCaller:
    """Provider-agnostic caller for OpenAI-compatible chat completion APIs."""

    def __init__(self, stage_configs: Optional[Mapping[str, ModelConfig]] = None) -> None:
        """Initialize the caller.

        Args:
            stage_configs: Effective stage configuration mapping.
        """
        self.stage_configs: Dict[str, ModelConfig] = dict(stage_configs or STAGE_CONFIGS)
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def call(
        self,
        config: ModelConfig,
        system: str,
        user: str,
        stream: bool = True,
        on_token: Optional[Callable[[str], None]] = None,
        stage: str = "unknown",
    ) -> LLMResponse:
        """Call an OpenAI-compatible backend and normalize the response.

        Args:
            config: Model configuration for the request.
            system: System prompt content.
            user: User prompt content.
            stream: Whether to stream tokens incrementally.
            on_token: Optional callback invoked for each streamed text chunk.
            stage: Logical pipeline stage for logging.

        Returns:
            Normalized LLM response.

        Raises:
            LLMError: If the backend fails after retries or returns a hard error.
        """
        client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        prompt_tokens = self.count_tokens(system) + self.count_tokens(user)
        messages = self._build_messages(config, system, user)

        for attempt in range(3):
            started_ns = perf_counter_ns()
            try:
                usage: Any = None
                raw_content: str

                if stream:
                    response_stream: Iterable[Any] = client.chat.completions.create(
                        model=config.name,
                        messages=messages,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        frequency_penalty=max(0.0, config.repeat_penalty - 1.0),
                        max_tokens=config.max_tokens,
                        stream=True,
                    )
                    content_parts: List[str] = []

                    for chunk in response_stream:
                        usage = getattr(chunk, "usage", usage)
                        choices = getattr(chunk, "choices", [])
                        if not choices:
                            continue
                        delta = getattr(choices[0], "delta", None)
                        token_piece = getattr(delta, "content", None)
                        if isinstance(token_piece, str) and token_piece != "":
                            content_parts.append(token_piece)
                            if on_token is not None:
                                on_token(token_piece)

                    raw_content = "".join(content_parts)
                else:
                    response: Any = client.chat.completions.create(
                        model=config.name,
                        messages=messages,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        frequency_penalty=max(0.0, config.repeat_penalty - 1.0),
                        max_tokens=config.max_tokens,
                        stream=False,
                    )
                    usage = getattr(response, "usage", None)
                    choices = getattr(response, "choices", [])
                    if not choices:
                        raise LLMError(config.provider.value, config.name, "No choices returned")
                    raw_message = getattr(choices[0], "message", None)
                    raw_content_value = getattr(raw_message, "content", "")
                    raw_content = raw_content_value if isinstance(raw_content_value, str) else ""
                    if on_token is not None and raw_content != "":
                        on_token(raw_content)

                cleaned_content, think_blocks = self._strip_think_blocks(raw_content)
                latency_ms = (perf_counter_ns() - started_ns) / 1_000_000
                tokens_in, tokens_out = self._resolve_usage(usage, prompt_tokens, cleaned_content)

                if think_blocks:
                    self._log_entry(
                        {
                            "timestamp": self._timestamp(),
                            "event": "think_block",
                            "provider": config.provider.value,
                            "model": config.name,
                            "stage": stage,
                            "think_blocks": think_blocks,
                        }
                    )

                self._log_entry(
                    {
                        "timestamp": self._timestamp(),
                        "provider": config.provider.value,
                        "model": config.name,
                        "stage": stage,
                        "tokens_in": tokens_in,
                        "tokens_out": tokens_out,
                        "latency_ms": round(latency_ms, 3),
                    }
                )

                return LLMResponse(
                    content=cleaned_content,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    model=config.name,
                    provider=config.provider.value,
                )
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                if attempt == 2:
                    failure_prefix = "Connection failure" if not isinstance(exc, RateLimitError) else "Rate limit failure"
                    raise LLMError(
                        config.provider.value,
                        config.name,
                        f"{failure_prefix} after retries: {exc}",
                    ) from exc

                backoff_seconds = 2 ** (attempt + 1)
                self._log_entry(
                    {
                        "timestamp": self._timestamp(),
                        "event": "retry",
                        "provider": config.provider.value,
                        "model": config.name,
                        "stage": stage,
                        "attempt": attempt + 1,
                        "backoff_seconds": backoff_seconds,
                        "retry_reason": "rate_limit" if isinstance(exc, RateLimitError) else "connection",
                        "error": str(exc),
                    }
                )
                time.sleep(backoff_seconds)
            except APIStatusError as exc:
                raise LLMError(
                    config.provider.value,
                    config.name,
                    f"Provider returned status {exc.status_code}: {exc}",
                ) from exc

        raise LLMError(config.provider.value, config.name, "Unreachable retry state")

    def extract_json(self, raw: str) -> dict[str, Any]:
        """Extract a JSON object using multiple recovery strategies.

        Args:
            raw: Raw model output text.

        Returns:
            Extracted JSON object.

        Raises:
            JSONExtractionError: If all strategies fail.
        """
        direct_candidate = raw.strip()
        direct_result = self._try_load_dict(direct_candidate)
        if direct_result is not None:
            return direct_result

        fenced_match = FENCED_JSON_PATTERN.search(raw)
        if fenced_match is not None:
            fenced_result = self._try_load_dict(fenced_match.group(1).strip())
            if fenced_result is not None:
                return fenced_result

        balanced_result = self._extract_balanced_object(raw)
        if balanced_result is not None:
            return balanced_result

        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            trimmed_candidate = raw[first_brace : last_brace + 1].strip()
            trimmed_result = self._try_load_dict(trimmed_candidate)
            if trimmed_result is not None:
                return trimmed_result

        raise JSONExtractionError("Unable to extract JSON object from model output", raw)

    def count_tokens(self, text: str) -> int:
        """Approximate the token count for a piece of text.

        Args:
            text: Input text.

        Returns:
            Approximate token count.
        """
        return len(self._encoding.encode(text))

    def check_budget(self, chain: ReasoningChain, upcoming_prompt_tokens: int) -> None:
        """Validate token and time budgets before another model call.

        Args:
            chain: Current reasoning chain state.
            upcoming_prompt_tokens: Approximate tokens for the next prompt.

        Raises:
            BudgetExceededError: If the token ceiling would be exceeded.
            TimeoutError: If the run exceeded the wall-time ceiling.
        """
        total_tokens_so_far = int(chain.metadata.get("total_tokens_in", 0)) + int(
            chain.metadata.get("total_tokens_out", 0)
        )
        projected_total = total_tokens_so_far + upcoming_prompt_tokens
        if projected_total > MAX_TOTAL_TOKENS_PER_RUN:
            raise BudgetExceededError(
                "Projected token usage exceeds MAX_TOTAL_TOKENS_PER_RUN"
            )

        started_at_raw = chain.metadata.get("started_at")
        if isinstance(started_at_raw, str):
            started_at = datetime.fromisoformat(started_at_raw)
            elapsed_seconds = (datetime.now(timezone.utc) - started_at).total_seconds()
            if elapsed_seconds > MAX_WALL_TIME_SECONDS:
                raise TimeoutError("Run exceeded MAX_WALL_TIME_SECONDS")

        for stage_name, config in self.stage_configs.items():
            warning_threshold = int(config.context_window * TOKEN_BUDGET_WARNING_PCT)
            if upcoming_prompt_tokens >= warning_threshold:
                self._log_entry(
                    {
                        "timestamp": self._timestamp(),
                        "event": "budget_warning",
                        "provider": config.provider.value,
                        "model": config.name,
                        "stage": stage_name,
                        "prompt_tokens": upcoming_prompt_tokens,
                        "context_window": config.context_window,
                    }
                )

    def estimate_cost_usd(
        self, config: ModelConfig, tokens_in: int, tokens_out: int
    ) -> float:
        """Estimate request cost in USD using a best-effort pricing table.

        Args:
            config: Model configuration used for the call.
            tokens_in: Prompt tokens.
            tokens_out: Completion tokens.

        Returns:
            Estimated USD cost.
        """
        input_rate, output_rate = MODEL_PRICING_USD_PER_MILLION.get(
            f"{config.provider.value}:{config.name}",
            MODEL_PRICING_USD_PER_MILLION.get(config.provider.value, (0.0, 0.0)),
        )
        return ((tokens_in / 1_000_000) * input_rate) + ((tokens_out / 1_000_000) * output_rate)

    def _build_messages(
        self, config: ModelConfig, system: str, user: str
    ) -> List[ChatCompletionMessageParam]:
        """Build chat messages, including provider-specific compatibility fallbacks.

        Args:
            config: Model configuration.
            system: System prompt content.
            user: User prompt content.

        Returns:
            OpenAI-compatible chat message payload.
        """
        if config.provider.value == "openrouter" and config.name.startswith("google/gemma-3"):
            merged_user_message: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": f"System instructions:\n{system}\n\nUser request:\n{user}",
            }
            return [merged_user_message]

        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": system,
        }
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": user,
        }
        return [system_message, user_message]

    def _resolve_usage(
        self, usage: Any, fallback_tokens_in: int, content: str
    ) -> Tuple[int, int]:
        """Resolve token counts from provider usage or fallback estimation.

        Args:
            usage: Provider-specific usage payload.
            fallback_tokens_in: Approximate prompt token count.
            content: Final response content.

        Returns:
            Tuple of prompt tokens and completion tokens.
        """
        if usage is None:
            return fallback_tokens_in, self.count_tokens(content)

        prompt_tokens_raw = getattr(usage, "prompt_tokens", fallback_tokens_in)
        completion_tokens_raw = getattr(usage, "completion_tokens", self.count_tokens(content))
        prompt_tokens = int(prompt_tokens_raw) if prompt_tokens_raw is not None else fallback_tokens_in
        completion_tokens = (
            int(completion_tokens_raw)
            if completion_tokens_raw is not None
            else self.count_tokens(content)
        )
        return prompt_tokens, completion_tokens

    def _strip_think_blocks(self, raw_content: str) -> Tuple[str, List[str]]:
        """Remove DeepSeek-style think blocks from returned content.

        Args:
            raw_content: Raw model output.

        Returns:
            Tuple of cleaned content and extracted think blocks.
        """
        think_blocks = [match.group(1).strip() for match in THINK_PATTERN.finditer(raw_content)]
        cleaned = THINK_PATTERN.sub("", raw_content).strip()
        return cleaned, think_blocks

    def _extract_balanced_object(self, raw: str) -> Optional[dict[str, Any]]:
        """Find and parse the first balanced JSON object in a string.

        Args:
            raw: Raw model output.

        Returns:
            Parsed dictionary if found, otherwise None.
        """
        for start_index, character in enumerate(raw):
            if character != "{":
                continue
            end_index = self._find_balanced_end(raw, start_index)
            if end_index is None:
                continue
            candidate = raw[start_index : end_index + 1]
            result = self._try_load_dict(candidate)
            if result is not None:
                return result
        return None

    def _find_balanced_end(self, raw: str, start_index: int) -> Optional[int]:
        """Find the matching closing brace for a JSON object.

        Args:
            raw: Source string.
            start_index: Index of the opening brace.

        Returns:
            The matching closing brace index, if found.
        """
        depth = 0
        in_string = False
        escaped = False

        for index in range(start_index, len(raw)):
            character = raw[index]

            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue

            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return index

        return None

    def _try_load_dict(self, candidate: str) -> Optional[dict[str, Any]]:
        """Attempt to load a JSON object from a candidate string.

        Args:
            candidate: Candidate JSON text.

        Returns:
            Parsed dictionary if successful, otherwise None.
        """
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None

        return parsed if isinstance(parsed, dict) else None

    def _log_entry(self, payload: Mapping[str, object]) -> None:
        """Append a structured log line to reasoning.log.

        Args:
            payload: Log payload.
        """
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(dict(payload), ensure_ascii=True, default=str))
            log_file.write("\n")

    def _timestamp(self) -> str:
        """Create an ISO-8601 UTC timestamp string.

        Returns:
            Timestamp string.
        """
        return datetime.now(timezone.utc).isoformat()