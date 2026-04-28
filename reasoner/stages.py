from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from pydantic import ValidationError

from reasoner.api import JSONExtractionError, LLMCaller
from reasoner.config import MAX_REASONING_STEPS, MIN_REASONING_STEPS, ModelConfig
from reasoner.models import CritiqueResult, ReasoningChain, ReasoningStep, RevisionContext, StepResult
from reasoner.prompts import (
    CRITIC_SYSTEM,
    DECOMPOSER_SYSTEM,
    REASONER_SYSTEM,
    SYNTHESIZER_SYSTEM,
    build_revision_prompt,
)


ANSWER_BLOCK_PATTERN = re.compile(r"<answer>.*?</answer>", re.DOTALL)
STEP_FRAGMENT_PATTERN = re.compile(
    r'"step_id"\s*:\s*(\d+)\s*,\s*"description"\s*:\s*"((?:\\.|[^"\\])*)"\s*,\s*"depends_on"\s*:\s*(\[[^\]]*\])',
    re.DOTALL,
)


def decompose(query: str, caller: LLMCaller) -> ReasoningChain:
    """Break a query into atomic reasoning steps.

    Args:
        query: Original user question.
        caller: Shared LLM caller.

    Returns:
        Initialized reasoning chain with decomposition steps populated.

    Raises:
        ValueError: If decomposition fails twice or yields invalid step counts.
    """
    config = caller.stage_configs["decomposer"]
    base_user_prompt = f"Decompose the following question: {query}"
    chain = ReasoningChain(query=query)

    for attempt in range(2):
        user_prompt = base_user_prompt
        if attempt == 1:
            user_prompt = (
                f"Decompose the following question: {query}\n\n"
                f"Return between {MIN_REASONING_STEPS} and {MAX_REASONING_STEPS} steps, inclusive. "
                "Do not return fewer or more."
            )

        prompt_tokens = caller.count_tokens(DECOMPOSER_SYSTEM) + caller.count_tokens(user_prompt)
        caller.check_budget(chain, prompt_tokens)
        _emit_stage_event(caller, "decomposer", "Decomposing query")
        response = caller.call(
            config=config,
            system=DECOMPOSER_SYSTEM,
            user=user_prompt,
            stream=True,
            on_token=_build_stream_handler(caller, "decomposer"),
            stage="decomposer",
        )
        _record_usage(chain, caller, config, response.tokens_in, response.tokens_out)

        steps = _extract_reasoning_steps(caller, response.content)

        if MIN_REASONING_STEPS <= len(steps) <= MAX_REASONING_STEPS:
            chain.steps = steps
            return chain

    raise ValueError("Decomposition failed to produce a valid step count")


def reason(
    chain: ReasoningChain,
    caller: LLMCaller,
    revision_context: Optional[RevisionContext] = None,
) -> ReasoningChain:
    """Execute reasoning steps sequentially.

    Args:
        chain: Current reasoning chain.
        caller: Shared LLM caller.
        revision_context: Optional revision instructions for affected steps.

    Returns:
        Updated reasoning chain with step results populated.
    """
    config = caller.stage_configs["reasoner"]
    existing_results = {result.step_id: result for result in chain.step_results}
    target_step_ids = (
        set(revision_context.critique.affected_step_ids)
        if revision_context is not None
        else None
    )

    chain.step_results = sorted(chain.step_results, key=lambda result: result.step_id)

    for step in sorted(chain.steps, key=lambda item: item.step_id):
        if step.step_id in existing_results and (target_step_ids is None or step.step_id not in target_step_ids):
            continue

        base_prompt = _build_reasoner_prompt(chain, step)
        user_prompt = base_prompt
        if target_step_ids is not None and step.step_id in target_step_ids:
            assert revision_context is not None
            prior_result = _find_prior_result(revision_context, step.step_id)
            if prior_result is not None:
                user_prompt = build_revision_prompt(base_prompt, prior_result, revision_context)
            else:
                user_prompt = (
                    f"{base_prompt}\n\nRevision instructions: {revision_context.instructions}"
                )

        prompt_tokens = caller.count_tokens(REASONER_SYSTEM) + caller.count_tokens(user_prompt)
        caller.check_budget(chain, prompt_tokens)
        _emit_stage_event(caller, "reasoner", f"Reasoning step {step.step_id}/{len(chain.steps)}", step.step_id)
        response = caller.call(
            config=config,
            system=REASONER_SYSTEM,
            user=user_prompt,
            stream=True,
            on_token=_build_stream_handler(caller, "reasoner", step.step_id),
            stage="reasoner",
        )
        _record_usage(chain, caller, config, response.tokens_in, response.tokens_out)

        payload = caller.extract_json(response.content)
        payload["step_id"] = step.step_id
        payload["description"] = step.description
        payload["tokens_used"] = response.tokens_in + response.tokens_out

        result = StepResult.model_validate(payload)
        _upsert_step_result(chain, result)
        if result.confidence < 0.5:
            chain.low_confidence_flag = True

    chain.step_results = sorted(chain.step_results, key=lambda result: result.step_id)
    return chain


def critique(chain: ReasoningChain, caller: LLMCaller) -> ReasoningChain:
    """Audit the full reasoning chain for logical defects.

    Args:
        chain: Current reasoning chain.
        caller: Shared LLM caller.

    Returns:
        Updated reasoning chain with critique results appended.
    """
    config = caller.stage_configs["critic"]
    critique_payload = {
        "query": chain.query,
        "step_results": [
            {
                "step_id": result.step_id,
                "description": result.description,
                "reasoning": result.reasoning,
                "conclusion": result.conclusion,
                "confidence": result.confidence,
                "assumptions": result.assumptions,
            }
            for result in sorted(chain.step_results, key=lambda item: item.step_id)
        ],
        "instruction": "Identify ALL logical failures in this reasoning chain",
    }
    user_prompt = json.dumps(critique_payload, ensure_ascii=True, indent=2)

    prompt_tokens = caller.count_tokens(CRITIC_SYSTEM) + caller.count_tokens(user_prompt)
    caller.check_budget(chain, prompt_tokens)
    _emit_stage_event(caller, "critic", "Critiquing reasoning chain")
    response = caller.call(
        config=config,
        system=CRITIC_SYSTEM,
        user=user_prompt,
        stream=True,
        on_token=_build_stream_handler(caller, "critic"),
        stage="critic",
    )
    _record_usage(chain, caller, config, response.tokens_in, response.tokens_out)

    payload = caller.extract_json(response.content)
    critique_result = CritiqueResult.model_validate(payload)

    has_high_issue = any(issue.severity == "high" for issue in critique_result.issues)
    if has_high_issue or critique_result.overall_quality < 0.65:
        critique_result.requires_revision = True

    if critique_result.requires_revision and not critique_result.affected_step_ids:
        critique_result.affected_step_ids = sorted({issue.step_id for issue in critique_result.issues})

    if critique_result.requires_revision and critique_result.revision_instructions is None:
        critique_result.revision_instructions = _build_revision_instructions(critique_result)

    chain.critiques.append(critique_result)
    return chain


def synthesize(chain: ReasoningChain, caller: LLMCaller) -> ReasoningChain:
    """Produce the final answer from validated reasoning results.

    Args:
        chain: Current reasoning chain.
        caller: Shared LLM caller.

    Returns:
        Completed reasoning chain with final answer populated.
    """
    config = caller.stage_configs["synthesizer"]
    latest_critique = chain.critiques[-1] if chain.critiques else None
    synthesis_payload: Dict[str, Any] = {
        "query": chain.query,
        "validated_steps": [
            {
                "step_id": result.step_id,
                "description": result.description,
                "reasoning": result.reasoning,
                "conclusion": result.conclusion,
                "confidence": result.confidence,
                "assumptions": result.assumptions,
            }
            for result in sorted(chain.step_results, key=lambda item: item.step_id)
        ],
        "critique_summary": {
            "overall_quality": latest_critique.overall_quality if latest_critique is not None else None,
            "issues": [issue.model_dump() for issue in latest_critique.issues]
            if latest_critique is not None
            else [],
        },
    }
    if chain.low_confidence_flag:
        synthesis_payload[
            "warning"
        ] = "At least one step was low confidence or a safety guard was triggered. Be explicit about uncertainty."

    user_prompt = json.dumps(synthesis_payload, ensure_ascii=True, indent=2)
    prompt_tokens = caller.count_tokens(SYNTHESIZER_SYSTEM) + caller.count_tokens(user_prompt)
    caller.check_budget(chain, prompt_tokens)
    _emit_stage_event(caller, "synthesizer", "Synthesizing final answer")
    response = caller.call(
        config=config,
        system=SYNTHESIZER_SYSTEM,
        user=user_prompt,
        stream=True,
        on_token=_build_stream_handler(caller, "synthesizer"),
        stage="synthesizer",
    )
    _record_usage(chain, caller, config, response.tokens_in, response.tokens_out)

    answer_match = ANSWER_BLOCK_PATTERN.search(response.content)
    answer_block = answer_match.group(0).strip() if answer_match is not None else response.content.strip()
    if chain.low_confidence_flag:
        chain.final_answer = (
            "WARNING: LOW CONFIDENCE\n"
            "One or more reasoning steps remained uncertain or a guardrail was triggered.\n\n"
            f"{answer_block}"
        )
    else:
        chain.final_answer = answer_block
    return chain


def _build_reasoner_prompt(chain: ReasoningChain, step: ReasoningStep) -> str:
    """Build the user prompt for a single reasoning step.

    Args:
        chain: Current reasoning chain.
        step: The step being executed.

    Returns:
        Prompt text for the reasoner stage.
    """
    prior_conclusions = [
        {"step_id": result.step_id, "conclusion": result.conclusion}
        for result in sorted(chain.step_results, key=lambda item: item.step_id)
        if result.step_id < step.step_id
    ]
    payload = {
        "query": chain.query,
        "prior_conclusions": prior_conclusions,
        "current_step": {
            "step_id": step.step_id,
            "description": step.description,
            "depends_on": step.depends_on,
        },
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def _extract_reasoning_steps(caller: LLMCaller, raw_response: str) -> List[ReasoningStep]:
    """Extract decomposition steps from model output, including malformed JSON fallbacks.

    Args:
        caller: Shared LLM caller.
        raw_response: Raw model response text.

    Returns:
        Validated reasoning steps, or an empty list if extraction fails.
    """
    raw_steps: List[Any] = []
    try:
        payload = caller.extract_json(raw_response)
        payload_steps = payload.get("steps", [])
        if isinstance(payload_steps, list):
            raw_steps = payload_steps
    except JSONExtractionError:
        raw_steps = []

    validated_steps = _validate_reasoning_steps(raw_steps)
    if validated_steps:
        return validated_steps

    return _extract_steps_from_raw_text(raw_response)


def _validate_reasoning_steps(raw_steps: List[Any]) -> List[ReasoningStep]:
    """Validate a list of raw step dictionaries.

    Args:
        raw_steps: Raw step payloads.

    Returns:
        Validated reasoning steps, or an empty list on validation failure.
    """
    try:
        return [ReasoningStep.model_validate(raw_step) for raw_step in raw_steps]
    except ValidationError:
        return []


def _extract_steps_from_raw_text(raw_response: str) -> List[ReasoningStep]:
    """Salvage reasoning steps from malformed near-JSON decomposition output.

    Args:
        raw_response: Raw model response text.

    Returns:
        Salvaged reasoning steps sorted by step_id.
    """
    steps_by_id: Dict[int, ReasoningStep] = {}

    for step_id_text, description_text, depends_on_text in STEP_FRAGMENT_PATTERN.findall(raw_response):
        try:
            step_id = int(step_id_text)
            description = json.loads(f'"{description_text}"')
            depends_on_raw = json.loads(depends_on_text)
        except (ValueError, json.JSONDecodeError):
            continue

        if not isinstance(description, str) or not isinstance(depends_on_raw, list):
            continue
        if not all(isinstance(item, int) for item in depends_on_raw):
            continue

        try:
            steps_by_id[step_id] = ReasoningStep(
                step_id=step_id,
                description=description,
                depends_on=[int(item) for item in depends_on_raw],
            )
        except ValidationError:
            continue

    return [steps_by_id[step_id] for step_id in sorted(steps_by_id)]


def _find_prior_result(
    revision_context: RevisionContext, step_id: int
) -> Optional[StepResult]:
    """Look up a prior step result within a revision context.

    Args:
        revision_context: Revision metadata.
        step_id: Step identifier.

    Returns:
        Matching prior step result if present.
    """
    for prior_result in revision_context.original_step_results:
        if prior_result.step_id == step_id:
            return prior_result
    return None


def _upsert_step_result(chain: ReasoningChain, result: StepResult) -> None:
    """Insert or replace a step result in the chain.

    Args:
        chain: Current reasoning chain.
        result: Step result to insert.
    """
    chain.step_results = [
        existing for existing in chain.step_results if existing.step_id != result.step_id
    ]
    chain.step_results.append(result)


def _record_usage(
    chain: ReasoningChain,
    caller: LLMCaller,
    config: ModelConfig,
    tokens_in: int,
    tokens_out: int,
) -> None:
    """Record aggregate usage and cost metadata on the reasoning chain.

    Args:
        chain: Current reasoning chain.
        caller: Shared LLM caller.
        config: Model config used for the request.
        tokens_in: Prompt token count.
        tokens_out: Completion token count.
    """
    chain.metadata["total_api_calls"] = int(chain.metadata.get("total_api_calls", 0)) + 1
    chain.metadata["total_tokens_in"] = int(chain.metadata.get("total_tokens_in", 0)) + tokens_in
    chain.metadata["total_tokens_out"] = int(chain.metadata.get("total_tokens_out", 0)) + tokens_out
    chain.metadata["estimated_cost_usd"] = float(chain.metadata.get("estimated_cost_usd", 0.0)) + caller.estimate_cost_usd(
        config,
        tokens_in,
        tokens_out,
    )

    model_label = f"{config.provider.value}:{config.name}"
    models_used = chain.metadata.setdefault("models_used", [])
    if isinstance(models_used, list) and model_label not in models_used:
        models_used.append(model_label)


def _build_revision_instructions(critique_result: CritiqueResult) -> str:
    """Create fallback revision instructions from critique issues.

    Args:
        critique_result: Critique output.

    Returns:
        Combined revision instructions.
    """
    if not critique_result.issues:
        return "Re-evaluate the affected steps and correct any logical failures."
    issue_lines = [
        f"Step {issue.step_id}: {issue.issue} ({issue.severity})" for issue in critique_result.issues
    ]
    return "Fix the following issues:\n" + "\n".join(issue_lines)


def _build_stream_handler(
    caller: LLMCaller,
    stage_name: str,
    step_id: Optional[int] = None,
) -> Optional[Callable[[str], None]]:
    """Build a token callback wrapper if the caller exposes a UI hook.

    Args:
        caller: Shared LLM caller.
        stage_name: Stage name for the callback.
        step_id: Optional step identifier.

    Returns:
        Token callback or None.
    """
    token_callback = getattr(caller, "token_callback", None)
    if not callable(token_callback):
        return None

    def handler(token: str) -> None:
        token_callback(stage_name, token, step_id)

    return handler


def _emit_stage_event(
    caller: LLMCaller,
    stage_name: str,
    message: str,
    step_id: Optional[int] = None,
) -> None:
    """Emit a stage-level event if the caller exposes a UI hook.

    Args:
        caller: Shared LLM caller.
        stage_name: Stage name.
        message: Human-readable event message.
        step_id: Optional step identifier.
    """
    stage_callback = getattr(caller, "stage_callback", None)
    if callable(stage_callback):
        stage_callback(stage_name, message, step_id)