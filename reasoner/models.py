from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _default_metadata() -> Dict[str, Any]:
    """Build default run metadata for a reasoning chain.

    Returns:
        A metadata dictionary with initialized counters and timestamps.
    """
    return {
        "total_api_calls": 0,
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "revision_cycles": 0,
        "elapsed_seconds": 0.0,
        "models_used": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "estimated_cost_usd": 0.0,
    }


class ReasoningStep(BaseModel):
    """An atomic reasoning task produced during decomposition."""

    model_config = ConfigDict(extra="forbid")

    step_id: int
    description: str
    depends_on: List[int] = Field(default_factory=list)


class StepResult(BaseModel):
    """The result of executing one reasoning step."""

    model_config = ConfigDict(extra="forbid")

    step_id: int
    description: str
    reasoning: str
    conclusion: str
    confidence: float
    assumptions: List[str] = Field(default_factory=list)
    tokens_used: int = 0

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        """Clamp confidence into the inclusive range [0.0, 1.0].

        Args:
            value: Raw model-reported confidence.

        Returns:
            A confidence score constrained to the supported range.
        """
        return max(0.0, min(1.0, value))


class CritiqueIssue(BaseModel):
    """A single issue found during critique."""

    model_config = ConfigDict(extra="forbid")

    step_id: int
    issue: str
    severity: Literal["low", "medium", "high"]


class CritiqueResult(BaseModel):
    """Aggregate critique output across the full reasoning chain."""

    model_config = ConfigDict(extra="forbid")

    issues: List[CritiqueIssue] = Field(default_factory=list)
    overall_quality: float
    requires_revision: bool
    revision_instructions: Optional[str] = None
    affected_step_ids: List[int] = Field(default_factory=list)

    @field_validator("overall_quality")
    @classmethod
    def clamp_quality(cls, value: float) -> float:
        """Clamp quality score into the inclusive range [0.0, 1.0].

        Args:
            value: Raw quality score.

        Returns:
            A bounded quality score.
        """
        return max(0.0, min(1.0, value))


class RevisionContext(BaseModel):
    """Context describing a requested revision cycle."""

    model_config = ConfigDict(extra="forbid")

    cycle_number: int
    original_step_results: List[StepResult] = Field(default_factory=list)
    critique: CritiqueResult
    instructions: str


class ReasoningChain(BaseModel):
    """Complete state for one end-to-end reasoning run."""

    model_config = ConfigDict(extra="forbid")

    query: str
    steps: List[ReasoningStep] = Field(default_factory=list)
    step_results: List[StepResult] = Field(default_factory=list)
    critiques: List[CritiqueResult] = Field(default_factory=list)
    revision_contexts: List[RevisionContext] = Field(default_factory=list)
    final_answer: Optional[str] = None
    low_confidence_flag: bool = False
    metadata: Dict[str, Any] = Field(default_factory=_default_metadata)