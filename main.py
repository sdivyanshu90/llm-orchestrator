from __future__ import annotations

import argparse
from typing import Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from reasoner.api import LLMCaller
from reasoner.cli import render_benchmark_panel, run_cli
from reasoner.config import Provider, load_stage_configs
from reasoner.engine import ReasonerExecutionError, run as run_engine


BENCHMARK_QUERIES = {
    "Q1": (
        "Should a new startup building a B2B SaaS product choose microservices "
        "or a monolithic architecture, and why?"
    ),
    "Q2": (
        "The trolley problem has a runaway trolley heading toward five people. "
        "You can pull a lever to divert it to a track with one person. "
        "What does each major ethical framework say you should do?"
    ),
    "Q3": (
        "Quicksort has O(n²) worst-case but O(n log n) average-case complexity. "
        "Explain precisely why the worst case occurs and under what input "
        "conditions, then explain the average-case proof intuition."
    ),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Production-ready CoT reasoner")
    parser.add_argument("query", nargs="?", help="Question to reason about")
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in Provider],
        help="Provider override for all stages",
    )
    parser.add_argument(
        "--model",
        help="Model override for all stages, e.g. llama-3.3-70b-versatile",
    )
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Run the three benchmark queries and print summary panels",
    )
    return parser


def parse_provider(raw_provider: Optional[str]) -> Optional[Provider]:
    """Parse an optional provider override.

    Args:
        raw_provider: Raw provider string.

    Returns:
        Provider enum value or None.
    """
    if raw_provider is None:
        return None
    return Provider(raw_provider)


def run_benchmarks(
    provider_override: Optional[Provider],
    model_override: Optional[str],
) -> int:
    """Run the benchmark suite and print a summary panel for each query.

    Args:
        provider_override: Optional provider override.
        model_override: Optional model override.

    Returns:
        Process exit code.
    """
    console = Console()
    exit_code = 0

    for label, query in BENCHMARK_QUERIES.items():
        stage_configs = load_stage_configs(
            provider_override=provider_override,
            model_override=model_override,
        )
        caller = LLMCaller(stage_configs)
        try:
            chain = run_engine(query, caller)
        except ReasonerExecutionError as exc:
            exit_code = 1
            console.print(
                Panel(
                    f"Benchmark {label} failed.\n\n{exc}",
                    title="Benchmark Error",
                    border_style="red",
                )
            )
            continue

        console.print(render_benchmark_panel(label, query, chain))

    return exit_code


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    provider_override = parse_provider(args.provider)

    if args.benchmarks:
        return run_benchmarks(provider_override, args.model)

    if args.query is None:
        parser.error("Either provide a query or pass --benchmarks")

    try:
        run_cli(args.query, provider_override=provider_override, model_override=args.model)
    except ReasonerExecutionError as exc:
        Console().print(
            Panel(
                f"Reasoner execution failed.\n\n{exc}",
                title="Execution Error",
                border_style="red",
            )
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())