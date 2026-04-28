from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Callable, Dict, List, Mapping, Optional

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from reasoner.api import LLMCaller
from reasoner.config import ModelConfig, Provider, describe_stage_models, load_stage_configs
from reasoner.engine import ReasonerExecutionError, run as run_engine
from reasoner.models import CritiqueResult, ReasoningChain, StepResult


class UICallbackCaller(LLMCaller):
    """LLM caller with UI callbacks for live Rich updates."""

    def __init__(self, stage_configs: Mapping[str, ModelConfig]) -> None:
        """Initialize the callback-enabled caller.

        Args:
            stage_configs: Effective stage configurations.
        """
        super().__init__(stage_configs)
        self.token_callback: Optional[Callable[[str, str, Optional[int]], None]] = None
        self.stage_callback: Optional[Callable[[str, str, Optional[int]], None]] = None


class RichReasonerUI:
    """Stateful Rich renderer for live reasoner execution."""

    def __init__(self, query: str, stage_configs: Mapping[str, ModelConfig], console: Console) -> None:
        """Initialize the UI state.

        Args:
            query: Original user query.
            stage_configs: Effective stage configs.
            console: Rich console instance.
        """
        self.query = query
        self.stage_configs = dict(stage_configs)
        self.console = console
        self.started_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.live: Optional[Live] = None
        self.last_refresh_started = perf_counter()
        self.current_stream_stage = "idle"
        self.current_stream_title = "Waiting for model output"
        self.stream_buffer = ""
        self.stage_statuses: Dict[str, str] = {
            "decomposer": "[Stage 1] Decomposing...",
            "reasoner": "[Stage 2] Waiting to reason...",
            "critic": "[Stage 3] Waiting to critique...",
            "synthesizer": "[Stage 4] Waiting to synthesize...",
        }
        self.stage_states: Dict[str, str] = {
            "decomposer": "pending",
            "reasoner": "pending",
            "critic": "pending",
            "synthesizer": "pending",
        }
        self.step_progress: Dict[int, str] = {}
        self.step_results: List[StepResult] = []
        self.critique_result: Optional[CritiqueResult] = None
        self.final_answer: Optional[str] = None
        self.stats: Dict[str, str] = {}
        self.error_message: Optional[str] = None

    def attach_live(self, live: Live) -> None:
        """Attach a Live context to enable refreshes.

        Args:
            live: Active Rich Live instance.
        """
        self.live = live

    def handle_stage_event(
        self, stage_name: str, message: str, step_id: Optional[int] = None
    ) -> None:
        """Handle a stage progress event from the engine.

        Args:
            stage_name: Stage emitting the event.
            message: User-facing progress message.
            step_id: Optional step identifier.
        """
        self.current_stream_stage = stage_name
        self.current_stream_title = message
        self.stage_states[stage_name] = "running"

        if stage_name == "reasoner" and step_id is not None:
            self.step_progress[step_id] = f"[Stage 2] Reasoning step {step_id}..."
        else:
            self.stage_statuses[stage_name] = message

        self.stream_buffer = ""
        self._refresh(force=True)

    def handle_token(self, stage_name: str, token: str, step_id: Optional[int] = None) -> None:
        """Handle a streamed token update from the caller.

        Args:
            stage_name: Stage producing the token.
            token: Token or chunk text.
            step_id: Optional step identifier.
        """
        if stage_name != self.current_stream_stage:
            self.current_stream_stage = stage_name
            self.stream_buffer = ""
        if step_id is not None and stage_name == "reasoner":
            self.current_stream_title = f"Reasoning step {step_id}"
        self.stream_buffer += token
        self.stream_buffer = self.stream_buffer[-6000:]
        self._refresh(force=False)

    def finalize(self, chain: ReasoningChain) -> None:
        """Update final UI state after a completed run.

        Args:
            chain: Final reasoning chain.
        """
        self.step_results = sorted(chain.step_results, key=lambda result: result.step_id)
        self.critique_result = chain.critiques[-1] if chain.critiques else None
        self.final_answer = chain.final_answer
        self.stage_states["decomposer"] = "done"
        self.stage_states["reasoner"] = "done"
        self.stage_states["critic"] = "done"
        self.stage_states["synthesizer"] = "done"
        self.stage_statuses["decomposer"] = (
            f"[Stage 1] Decomposing...          ✓ ({len(chain.steps)} steps identified)"
        )
        self.stage_statuses["reasoner"] = (
            f"[Stage 2] Reasoning...            ✓ ({len(chain.step_results)} steps completed)"
        )
        if self.critique_result is not None and self.critique_result.issues:
            issue_count = len(self.critique_result.issues)
            self.stage_statuses["critic"] = (
                f"[Stage 3] Critiquing...           ⚠ {issue_count} issues found"
            )
        else:
            self.stage_statuses["critic"] = "[Stage 3] Critiquing...           ✓"
        self.stage_statuses["synthesizer"] = "[Stage 4] Synthesizing...        ✓"
        self.stats = {
            "Total API calls": str(chain.metadata.get("total_api_calls", 0)),
            "Tokens in": f"{int(chain.metadata.get('total_tokens_in', 0)):,}",
            "Tokens out": f"{int(chain.metadata.get('total_tokens_out', 0)):,}",
            "Revision cycles": str(chain.metadata.get("revision_cycles", 0)),
            "Time": f"{float(chain.metadata.get('elapsed_seconds', 0.0)):.1f}s",
            "Estimated cost (USD)": f"{float(chain.metadata.get('estimated_cost_usd', 0.0)):.6f}",
        }
        self.current_stream_title = "Run complete"
        self._refresh(force=True)

    def finalize_error(self, message: str, chain: Optional[ReasoningChain]) -> None:
        """Update UI state for an error.

        Args:
            message: Error message.
            chain: Partial reasoning chain, if available.
        """
        self.error_message = message
        if chain is not None:
            self.step_results = sorted(chain.step_results, key=lambda result: result.step_id)
            self.critique_result = chain.critiques[-1] if chain.critiques else None
            self.final_answer = chain.final_answer
        self._refresh(force=True)

    def render(self) -> RenderableType:
        """Render the current UI state.

        Returns:
            Combined Rich renderable.
        """
        renderables: List[RenderableType] = [
            self._build_header_panel(),
            self._build_stage_progress_table(),
            self._build_stream_panel(),
        ]

        step_table = self._build_step_confidence_table()
        if step_table is not None:
            renderables.append(step_table)

        critique_table = self._build_critique_table()
        if critique_table is not None:
            renderables.append(critique_table)

        if self.final_answer is not None:
            renderables.append(self._build_final_answer_panel())

        if self.stats:
            renderables.append(self._build_stats_table())

        if self.error_message is not None:
            renderables.append(Panel(self.error_message, title="Error", border_style="red"))

        return Group(*renderables)

    def _build_header_panel(self) -> Panel:
        """Render the header panel.

        Returns:
            Header panel.
        """
        grid = Table.grid(expand=True)
        grid.add_column(ratio=2)
        grid.add_column(ratio=3)
        stage_models = describe_stage_models(self.stage_configs)
        grid.add_row("Query", self.query)
        grid.add_row("Timestamp", self.started_at)
        grid.add_row("Decomposer", stage_models["decomposer"])
        grid.add_row("Reasoner", stage_models["reasoner"])
        grid.add_row("Critic", stage_models["critic"])
        grid.add_row("Synthesizer", stage_models["synthesizer"])
        return Panel(grid, title="CoT Reasoner", border_style="cyan")

    def _build_stage_progress_table(self) -> Table:
        """Render the stage progress table.

        Returns:
            Stage progress table.
        """
        table = Table(title="Stage Progress", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("Stage", style="bold")
        table.add_column("Status")
        table.add_row("Stage 1", self.stage_statuses["decomposer"])

        if self.step_progress:
            for step_id in sorted(self.step_progress):
                table.add_row(f"Step {step_id}", self.step_progress[step_id])
        else:
            table.add_row("Stage 2", self.stage_statuses["reasoner"])

        table.add_row("Stage 3", self.stage_statuses["critic"])
        table.add_row("Stage 4", self.stage_statuses["synthesizer"])
        return table

    def _build_stream_panel(self) -> Panel:
        """Render the live streaming output panel.

        Returns:
            Streaming panel.
        """
        content = self.stream_buffer if self.stream_buffer else "Waiting for streamed tokens..."
        return Panel(content, title=f"Streaming: {self.current_stream_title}", border_style="blue")

    def _build_step_confidence_table(self) -> Optional[Table]:
        """Render a step confidence table once step results exist.

        Returns:
            Confidence table or None.
        """
        if not self.step_results:
            return None

        table = Table(title="Step Confidence", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("Step", justify="right")
        table.add_column("Conclusion")
        table.add_column("Confidence", justify="right")
        for result in self.step_results:
            confidence_style = _confidence_style(result.confidence)
            table.add_row(
                str(result.step_id),
                result.conclusion,
                Text(f"{result.confidence:.2f}", style=confidence_style),
            )
        return table

    def _build_critique_table(self) -> Optional[Table]:
        """Render the critique issue table, if any.

        Returns:
            Critique table or None.
        """
        if self.critique_result is None or not self.critique_result.issues:
            return None

        table = Table(title="Critique Issues", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("Step", justify="right")
        table.add_column("Severity")
        table.add_column("Issue")
        for issue in self.critique_result.issues:
            table.add_row(
                str(issue.step_id),
                issue.severity,
                issue.issue,
                style=_severity_style(issue.severity),
            )
        return table

    def _build_final_answer_panel(self) -> Panel:
        """Render the final answer panel.

        Returns:
            Final answer panel.
        """
        assert self.final_answer is not None
        low_confidence = self.final_answer.startswith("WARNING: LOW CONFIDENCE")
        title = "⚠ LOW CONFIDENCE" if low_confidence else "Final Answer"
        border_style = "red" if low_confidence else "green"
        return Panel(self.final_answer, title=title, border_style=border_style)

    def _build_stats_table(self) -> Table:
        """Render the footer stats table.

        Returns:
            Stats table.
        """
        table = Table(title="Run Stats", box=box.SIMPLE_HEAVY, expand=True)
        for column_name in self.stats.keys():
            table.add_column(column_name, justify="center")
        table.add_row(*self.stats.values())
        return table

    def _refresh(self, force: bool) -> None:
        """Refresh the live UI, optionally throttled.

        Args:
            force: Whether to bypass refresh throttling.
        """
        if self.live is None:
            return

        now = perf_counter()
        if not force and (now - self.last_refresh_started) < 0.05:
            return
        self.last_refresh_started = now
        self.live.update(self.render(), refresh=True)


def run_cli(
    query: str,
    provider_override: Optional[Provider] = None,
    model_override: Optional[str] = None,
) -> ReasoningChain:
    """Run the reasoner with a Rich live terminal UI.

    Args:
        query: Original user query.
        provider_override: Optional provider override.
        model_override: Optional model override applied to all stages.

    Returns:
        Final reasoning chain.

    Raises:
        ReasonerExecutionError: If the underlying engine fails.
    """
    stage_configs = load_stage_configs(
        provider_override=provider_override,
        model_override=model_override,
    )
    console = Console()
    ui = RichReasonerUI(query, stage_configs, console)
    caller = UICallbackCaller(stage_configs)
    caller.token_callback = ui.handle_token
    caller.stage_callback = ui.handle_stage_event

    with Live(ui.render(), console=console, refresh_per_second=12, transient=False) as live:
        ui.attach_live(live)
        try:
            chain = run_engine(query, caller)
        except ReasonerExecutionError as exc:
            ui.finalize_error(str(exc), exc.chain)
            raise
        ui.finalize(chain)
        live.update(ui.render(), refresh=True)

    return chain


def render_benchmark_panel(label: str, query: str, chain: ReasoningChain) -> Panel:
    """Render a compact benchmark summary panel.

    Args:
        label: Query label such as Q1.
        query: Original benchmark query text.
        chain: Completed reasoning chain.

    Returns:
        Rich panel summarizing the benchmark run.
    """
    latest_critique = chain.critiques[-1] if chain.critiques else None
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    if latest_critique is not None:
        for issue in latest_critique.issues:
            severity_counts[issue.severity] += 1

    quality = latest_critique.overall_quality if latest_critique is not None else 0.0
    issue_count = len(latest_critique.issues) if latest_critique is not None else 0
    body = Table.grid(expand=True)
    body.add_column()
    body.add_row(f"Query: {label}")
    body.add_row(query)
    body.add_row(f"Steps decomposed: {len(chain.steps)}")
    body.add_row(f"Revision cycles: {int(chain.metadata.get('revision_cycles', 0))}")
    body.add_row(
        "Issues found: "
        f"{issue_count} ({severity_counts['high']} high, {severity_counts['medium']} medium, {severity_counts['low']} low)"
    )
    body.add_row(f"Overall quality: {quality:.2f}")
    body.add_row(
        "Tokens in: "
        f"{int(chain.metadata.get('total_tokens_in', 0)):,} / out: {int(chain.metadata.get('total_tokens_out', 0)):,}"
    )
    body.add_row(f"Wall time: {float(chain.metadata.get('elapsed_seconds', 0.0)):.1f}s")
    body.add_row(f"Low confidence: {'Yes' if chain.low_confidence_flag else 'No'}")
    return Panel.fit(body, border_style="red" if chain.low_confidence_flag else "cyan")


def _confidence_style(confidence: float) -> str:
    """Resolve a Rich style for a confidence score.

    Args:
        confidence: Confidence score.

    Returns:
        Rich style name.
    """
    if confidence >= 0.8:
        return "green"
    if confidence >= 0.5:
        return "yellow"
    return "red"


def _severity_style(severity: str) -> str:
    """Resolve a Rich row style for critique severity.

    Args:
        severity: Severity string.

    Returns:
        Rich style string.
    """
    if severity == "high":
        return "white on red"
    if severity == "medium":
        return "black on yellow"
    return "dim"