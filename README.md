# llm-orchestrator

A production-ready Chain-of-Thought (CoT) reasoning orchestrator. It routes a query through a four-stage LLM pipeline — **Decompose → Reason → Critique → Synthesize** — with automatic revision cycles and a live Rich terminal UI.

---

## How it works

```
Query ──► Decomposer ──► Reasoner ──► Critic ──► Synthesizer ──► Answer
                              ▲______________|
                              (revision loop, up to MAX_REVISION_CYCLES)
```

| Stage | Role | Default model |
|---|---|---|
| **Decomposer** | Breaks the query into atomic reasoning steps | `qwen3:14b` |
| **Reasoner** | Executes each step sequentially | `qwen3:14b` |
| **Critic** | Reviews the reasoning and flags issues | `deepseek-r1:14b` |
| **Synthesizer** | Produces the final answer from validated steps | `qwen3:14b` |

If the critic requires a revision, the reasoner re-runs on the affected steps (up to `MAX_REVISION_CYCLES` times). If the limit is hit, the result is flagged as **LOW CONFIDENCE**.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

---

## Quick start

```bash
# Ask a single question (uses Ollama by default)
python main.py "Should a startup choose microservices or a monolith?"

# Override provider and model for all stages
python main.py "Explain quicksort worst-case complexity" --provider groq --model llama-3.3-70b-versatile

# Run the built-in benchmark suite (three reference queries)
python main.py --benchmarks
```

---

## Configuration

Copy or create a `.env` file in the project root. All variables are optional — defaults use a local Ollama instance.

### Provider settings

Each provider uses a `{PREFIX}_BASE_URL` and `{PREFIX}_API_KEY` pair.

| Provider | Prefix | Default base URL |
|---|---|---|
| Ollama | `OLLAMA` | `http://localhost:11434/v1` |
| Groq | `GROQ` | `https://api.groq.com/openai/v1` |
| OpenRouter | `OPENROUTER` | `https://openrouter.ai/api/v1` |
| HF TGI | `HF_TGI` | `http://localhost:8080/v1` |
| Custom | `CUSTOM` | `http://localhost:8000/v1` |

Example `.env`:

```dotenv
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-...

# Optional runtime limits
MAX_REVISION_CYCLES=2
MAX_TOTAL_TOKENS=20000
MAX_WALL_TIME_SECONDS=300
```

### Runtime limits

| Variable | Default | Description |
|---|---|---|
| `MAX_REVISION_CYCLES` | `2` | Maximum critic→reasoner revision loops per run |
| `MAX_TOTAL_TOKENS` | `20000` | Token budget across all stages for a single run |
| `MAX_WALL_TIME_SECONDS` | `300` | Wall-clock timeout for a single run |

> **Note:** Stale shell exports (`MODEL_PROVIDER`, `OPENAI_BASE_URL`, etc.) override `.env` values. Unset them or use a fresh terminal if settings look inconsistent.

---

## CLI reference

```
python main.py [query] [--provider PROVIDER] [--model MODEL] [--benchmarks]
```

| Argument | Description |
|---|---|
| `query` | Question to reason about |
| `--provider` | Override the provider for all stages (`ollama`, `groq`, `openrouter`, `hf_tgi`, `custom`) |
| `--model` | Override the model name for all stages |
| `--benchmarks` | Run the three built-in benchmark queries and print summary panels |

---

## Project structure

```
main.py                  # CLI entry point and benchmark runner
reasoner/
  config.py              # Providers, ModelConfig, stage defaults, env vars
  engine.py              # Pipeline orchestration and revision loop
  stages.py              # Decompose / Reason / Critique / Synthesize logic
  api.py                 # LLMCaller, token counting, retry, logging
  models.py              # Pydantic data models (ReasoningChain, etc.)
  prompts.py             # System and user prompt templates
  cli.py                 # Rich terminal UI and benchmark panel renderer
tests/
  conftest.py
  test_api.py
  test_stages.py
```

---

## Running tests

```bash
pytest
```

Type-check with:

```bash
mypy reasoner/ main.py
```
