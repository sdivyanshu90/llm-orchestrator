from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from reasoner.api import BudgetExceededError, JSONExtractionError, LLMCaller
from reasoner.config import MAX_TOTAL_TOKENS_PER_RUN, ModelConfig, Provider
from reasoner.models import ReasoningChain


def build_stage_configs() -> dict[str, ModelConfig]:
    """Create a minimal stage config mapping for tests.

    Returns:
        Stage config mapping.
    """
    return {
        "reasoner": ModelConfig(
            name="test-model",
            provider=Provider.OLLAMA,
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.0,
            max_tokens=128,
            context_window=256,
        )
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"steps": [{"step_id": 1, "description": "a", "depends_on": []}]}', "steps"),
        (
            '```json\n{"issues": [], "overall_quality": 0.9, "requires_revision": false, "revision_instructions": null, "affected_step_ids": []}\n```',
            "issues",
        ),
        (
            'prefix text {"step_id": 1, "reasoning": "x", "conclusion": "y", "confidence": 0.7, "assumptions": []} suffix text',
            "step_id",
        ),
    ],
)
def test_extract_json_strategies(raw: str, expected: str) -> None:
    """extract_json should recover JSON from several common model formats.

    Args:
        raw: Raw model output.
        expected: Expected top-level JSON key.
    """
    caller = LLMCaller(build_stage_configs())

    payload = caller.extract_json(raw)

    assert expected in payload


def test_extract_json_raises_when_unrecoverable() -> None:
    """extract_json should raise when no valid JSON object can be found."""
    caller = LLMCaller(build_stage_configs())

    with pytest.raises(JSONExtractionError):
        caller.extract_json("this is not json")


def test_check_budget_raises_on_token_limit() -> None:
    """check_budget should raise when projected tokens exceed the run cap."""
    caller = LLMCaller(build_stage_configs())
    chain = ReasoningChain(query="test")
    chain.metadata["total_tokens_in"] = MAX_TOTAL_TOKENS_PER_RUN
    chain.metadata["total_tokens_out"] = 1

    with pytest.raises(BudgetExceededError):
        caller.check_budget(chain, upcoming_prompt_tokens=1)


def test_check_budget_raises_on_timeout() -> None:
    """check_budget should raise when the run exceeds wall time."""
    caller = LLMCaller(build_stage_configs())
    chain = ReasoningChain(query="test")
    chain.metadata["started_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=301)
    ).isoformat()

    with pytest.raises(TimeoutError):
        caller.check_budget(chain, upcoming_prompt_tokens=1)


def test_check_budget_allows_safe_request() -> None:
    """check_budget should not raise for a safe request."""
    caller = LLMCaller(build_stage_configs())
    chain = ReasoningChain(query="test")
    chain.metadata["started_at"] = datetime.now(timezone.utc).isoformat()
    chain.metadata["total_tokens_in"] = 100
    chain.metadata["total_tokens_out"] = 50

    caller.check_budget(chain, upcoming_prompt_tokens=25)