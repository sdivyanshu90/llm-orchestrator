from __future__ import annotations

from time import perf_counter
from typing import Optional

from reasoner.api import BudgetExceededError, LLMCaller
from reasoner.config import MAX_REVISION_CYCLES
from reasoner.models import ReasoningChain, RevisionContext
from reasoner.stages import critique, decompose, reason, synthesize


class ReasonerExecutionError(RuntimeError):
    """Raised when a run fails and the partial chain should be preserved."""

    def __init__(self, message: str, chain: Optional[ReasoningChain]) -> None:
        super().__init__(message)
        self.chain = chain


def run(query: str, caller: LLMCaller) -> ReasoningChain:
    """Run the full decomposition, reasoning, critique, and synthesis pipeline.

    Args:
        query: Original user question.
        caller: Shared LLM caller.

    Returns:
        Final reasoning chain state.

    Raises:
        ReasonerExecutionError: If execution fails unexpectedly.
    """
    started_at = perf_counter()
    chain: Optional[ReasoningChain] = None

    try:
        chain = decompose(query, caller)
        chain = reason(chain, caller)
        chain = critique(chain, caller)
        revision_cycle = 0

        while chain.critiques[-1].requires_revision:
            if revision_cycle >= MAX_REVISION_CYCLES:
                chain.low_confidence_flag = True
                _log_engine_event(
                    caller,
                    "max_revisions_reached",
                    "Max revisions reached - flagging as LOW CONFIDENCE",
                )
                break

            latest_critique = chain.critiques[-1]
            revision_context = RevisionContext(
                cycle_number=revision_cycle + 1,
                original_step_results=list(chain.step_results),
                critique=latest_critique,
                instructions=latest_critique.revision_instructions
                or "Re-evaluate the affected steps and fix the critique issues.",
            )
            chain.revision_contexts.append(revision_context)

            affected_ids = set(latest_critique.affected_step_ids)
            chain.step_results = [
                result for result in chain.step_results if result.step_id not in affected_ids
            ]
            chain = reason(chain, caller, revision_context)
            chain = critique(chain, caller)
            revision_cycle += 1
            chain.metadata["revision_cycles"] = revision_cycle

        chain = synthesize(chain, caller)
        chain.metadata["elapsed_seconds"] = perf_counter() - started_at
        return chain
    except BudgetExceededError as exc:
        guarded_chain = _ensure_chain(chain, query)
        guarded_chain.low_confidence_flag = True
        _log_engine_event(caller, "budget_exceeded", str(exc))
        finalized_chain = _safe_finalize(guarded_chain, caller, started_at, str(exc))
        return finalized_chain
    except TimeoutError as exc:
        guarded_chain = _ensure_chain(chain, query)
        guarded_chain.low_confidence_flag = True
        _log_engine_event(caller, "timeout", str(exc))
        finalized_chain = _safe_finalize(guarded_chain, caller, started_at, str(exc))
        return finalized_chain
    except (RuntimeError, ValueError) as exc:
        if chain is not None:
            chain.metadata["elapsed_seconds"] = perf_counter() - started_at
        raise ReasonerExecutionError(f"Run failed: {exc}", chain) from exc


def _ensure_chain(chain: Optional[ReasoningChain], query: str) -> ReasoningChain:
    """Ensure a reasoning chain object exists for guarded finalization.

    Args:
        chain: Existing chain or None.
        query: Original user query.

    Returns:
        Existing chain or a minimal new chain.
    """
    return chain if chain is not None else ReasoningChain(query=query)


def _safe_finalize(
    chain: ReasoningChain,
    caller: LLMCaller,
    started_at: float,
    failure_reason: str,
) -> ReasoningChain:
    """Attempt synthesis after a guardrail failure, with a fallback answer.

    Args:
        chain: Current reasoning chain.
        caller: Shared LLM caller.
        started_at: Perf counter timestamp for the run start.
        failure_reason: Reason for guarded fallback.

    Returns:
        Finalized reasoning chain.
    """
    try:
        chain = synthesize(chain, caller)
    except (BudgetExceededError, TimeoutError, RuntimeError, ValueError):
        chain.final_answer = _build_fallback_answer(chain, failure_reason)

    chain.metadata["elapsed_seconds"] = perf_counter() - started_at
    return chain


def _build_fallback_answer(chain: ReasoningChain, failure_reason: str) -> str:
    """Build a deterministic fallback answer when synthesis cannot run.

    Args:
        chain: Current reasoning chain.
        failure_reason: Reason synthesis fallback was needed.

    Returns:
        Fallback final answer string.
    """
    summary_lines = [
        "WARNING: LOW CONFIDENCE",
        f"Run finalized early because a guardrail triggered: {failure_reason}",
        "",
        "Partial conclusions:",
    ]
    for result in sorted(chain.step_results, key=lambda item: item.step_id):
        summary_lines.append(f"- [Step {result.step_id}] {result.conclusion}")
    if not chain.step_results:
        summary_lines.append("- No completed reasoning steps were available.")
    return "\n".join(summary_lines)


def _log_engine_event(caller: LLMCaller, event: str, message: str) -> None:
    """Write an engine-level event to the shared reasoning log.

    Args:
        caller: Shared LLM caller.
        event: Event name.
        message: Event message.
    """
    caller._log_entry(
        {
            "event": event,
            "message": message,
        }
    )