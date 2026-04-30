"""
Phase 1 - Causal Exploration

Runs 10 independent calls to Claude (Sonnet 4.6 by default), each generating
a candidate list of undirected causal edges over the 20 features + 1 target.
Aggregates results across runs to produce a single edge list with frequency
counts (how many of the 10 runs proposed each edge).

Inputs:
    LLM Context/prompt.txt                       core system instructions
    LLM Context/verbose_feature_definitions.docx feature definitions

Outputs:
    outputs/system_prompt_used.txt     full assembled prompt (for reproducibility)
    outputs/exploration_run_NN.json    one file per run, raw edges + meta
    outputs/aggregated_edges.json      unioned edges with run counts
    outputs/aggregation_report.txt     human-readable summary
"""

import json
import os
import time
from pathlib import Path

from anthropic import Anthropic
from docx import Document
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model + sampling settings - paper used 10 runs at temp=1 with GPT-4
MODEL = "claude-sonnet-4-6"
NUM_RUNS = 10
TEMPERATURE = 1.0
MAX_TOKENS = 8000

# Paths - all relative to the script location (project root)
SCRIPT_DIR = Path(__file__).resolve().parent
CONTEXT_DIR = SCRIPT_DIR / "LLM Context"
PROMPT_FILE = CONTEXT_DIR / "prompt.txt"
FEATURES_FILE = CONTEXT_DIR / "verbose_feature_definitions.docx"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

# Standard guardrails appended to the user-editable core prompt
# Keeps prompt.txt as the editable source of truth while ensuring
# anti-hallucination guidance, naming discipline, and format example
# are always present
APPENDED_GUARDRAILS = """

=== ADDITIONAL GUIDANCE ===

Do not invent relationships that have no plausible mechanism. "These two variables both relate to inflation" is NOT a mechanism unless one variable has a direct causal effect on the other. Each candidate edge must be supported by a mechanism grounded in the feature definitions or in standard financial economics.

Use the exact feature names provided in the feature definitions (e.g., `real_10y`, `fed_funds`, `gold_price`) for node_a and node_b. Do not use display names, abbreviations, or paraphrases.

=== EXAMPLE OUTPUT FORMAT ===

[
  {"node_a": "real_10y", "node_b": "gold_price", "justification": "Real yields represent the opportunity cost of holding non-yielding assets. When real yields rise, gold becomes relatively less attractive."},
  {"node_a": "fed_funds", "node_b": "real_10y", "justification": "Fed Funds is the policy rate that anchors the front of the yield curve and influences longer-dated real yields through the expectations channel."}
]"""


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_system_prompt(path: Path) -> str:
    # Read core system instructions from prompt.txt and append guardrails
    core = path.read_text(encoding="utf-8").strip()
    return core + APPENDED_GUARDRAILS


def load_feature_definitions(path: Path) -> str:
    # Extract all text from the docx feature definitions file
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def build_user_message(feature_text: str) -> str:
    # Construct the message containing feature definitions + final ask
    return (
        "Below are the feature definitions for the gold price DAG. The target "
        "variable is `gold_price`. The 20 input features span monetary policy, "
        "currency, equity/risk sentiment, fiscal stress, commodities, supply, "
        "demand, uncertainty, consumer sentiment, substitutes, and labor markets.\n\n"
        "=== FEATURE DEFINITIONS ===\n\n"
        f"{feature_text}\n\n"
        "=== TASK ===\n\n"
        "Identify all plausible undirected causal relationships among these "
        "features (including relationships involving the target `gold_price`). "
        "For each candidate edge, provide a brief 1-2 sentence justification "
        "grounded in the feature mechanisms. Output a JSON array as specified "
        "in your instructions."
    )


# ---------------------------------------------------------------------------
# API calling - one call per run
# ---------------------------------------------------------------------------

def parse_edges_from_response(raw_text: str, run_id: int) -> list | None:
    # Parse the model's JSON response, stripping markdown code fences if present.
    # Strip leading/trailing whitespace
    text = raw_text.strip()

    # Strip markdown code fences if present - handles ```json...``` and ```...```
    if text.startswith("```"):
        # Remove opening fence (could be ``` or ```json or ```JSON etc)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Remove closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    # Try to parse what's left
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as exc:
        print(f"  [run {run_id}] WARNING: malformed JSON, error: {exc}")
        return None

def call_exploration(
    client: Anthropic,
    system_prompt: str,
    feature_text: str,
    run_id: int,
) -> dict:
    # Make one API call asking for candidate edges.
    user_msg = build_user_message(feature_text)

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_msg,
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ],
    )

    # Pull the raw text out of the response content blocks
    raw_text = response.content[0].text

    # Try to parse as JSON, after stripping markdown code fences if present
    # Claude sometimes wraps JSON in ```json ... ``` despite prompt instructions
    edges = parse_edges_from_response(raw_text, run_id)

    # Pull usage stats - cache fields are present in response.usage
    usage = response.usage
    return {
        "run_id": run_id,
        "model": MODEL,
        "raw_text": raw_text,
        "edges": edges,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_creation_input_tokens": getattr(
                usage, "cache_creation_input_tokens", 0
            ),
            "cache_read_input_tokens": getattr(
                usage, "cache_read_input_tokens", 0
            ),
        },
    }


# ---------------------------------------------------------------------------
# Aggregation across runs
# ---------------------------------------------------------------------------


def canonical_edge(node_a: str, node_b: str) -> tuple[str, str]:
    # Sort node names so (a,b) and (b,a) match in the count dict
    return tuple(sorted([node_a.strip(), node_b.strip()]))


def aggregate_runs(run_results: list[dict]) -> dict:
    # Combine edges from all successful runs.
    edge_data: dict[tuple, dict] = {}

    for result in run_results:
        if result["edges"] is None:
            continue

        for edge in result["edges"]:
            # Skip malformed edge entries (missing keys)
            if not all(k in edge for k in ("node_a", "node_b")):
                continue

            key = canonical_edge(edge["node_a"], edge["node_b"])
            if key not in edge_data:
                edge_data[key] = {"count": 0, "justifications": []}
            edge_data[key]["count"] += 1
            edge_data[key]["justifications"].append(
                {
                    "run_id": result["run_id"],
                    "text": edge.get("justification", ""),
                }
            )

    return edge_data


def write_aggregation_report(
    edge_data: dict, run_results: list[dict], path: Path
) -> None:
    # Write a plain-text summary to eyeball the aggregation.
    successful = [r for r in run_results if r["edges"] is not None]
    edges_per_run = [len(r["edges"]) for r in successful]

    lines = []
    lines.append("=" * 70)
    lines.append("Causal Exploration - Aggregation Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Model: {MODEL}")
    lines.append(f"Runs attempted: {len(run_results)}")
    lines.append(f"Runs succeeded: {len(successful)}")
    lines.append(f"Runs with malformed JSON: {len(run_results) - len(successful)}")
    lines.append("")

    if edges_per_run:
        lines.append(
            f"Edges per run: min={min(edges_per_run)}, "
            f"max={max(edges_per_run)}, "
            f"mean={sum(edges_per_run) / len(edges_per_run):.1f}"
        )
    lines.append(f"Unique edges across all runs: {len(edge_data)}")
    lines.append("")

    # Show distribution of edge frequencies
    freq_dist = {}
    for data in edge_data.values():
        c = data["count"]
        freq_dist[c] = freq_dist.get(c, 0) + 1

    lines.append("Edge frequency distribution (count: # of edges with that count):")
    for c in sorted(freq_dist.keys(), reverse=True):
        lines.append(
            f"  appeared in {c}/{len(successful)} runs: {freq_dist[c]} edges"
        )
    lines.append("")

    # Sort edges by frequency for readability
    sorted_edges = sorted(
        edge_data.items(), key=lambda kv: (-kv[1]["count"], kv[0])
    )

    lines.append("=" * 70)
    lines.append("All edges (sorted by frequency, then alphabetical)")
    lines.append("=" * 70)
    for (a, b), data in sorted_edges:
        lines.append(f"  [{data['count']:2d}/{len(successful)}] {a} <-> {b}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("Token usage summary")
    lines.append("=" * 70)
    total_input = sum(r["usage"]["input_tokens"] for r in run_results)
    total_output = sum(r["usage"]["output_tokens"] for r in run_results)
    total_cache_create = sum(
        r["usage"]["cache_creation_input_tokens"] for r in run_results
    )
    total_cache_read = sum(
        r["usage"]["cache_read_input_tokens"] for r in run_results
    )
    lines.append(f"Total uncached input tokens: {total_input:,}")
    lines.append(f"Total cache creation tokens: {total_cache_create:,}")
    lines.append(f"Total cache read tokens: {total_cache_read:,}")
    lines.append(f"Total output tokens: {total_output:,}")

    # Cost estimate for Sonnet 4.6 ($3/M input, $15/M output)
    # Cache writes ~1.25x input rate, cache reads ~0.1x input rate
    cost_input = total_input * 3.0 / 1_000_000
    cost_cache_create = total_cache_create * 3.75 / 1_000_000
    cost_cache_read = total_cache_read * 0.30 / 1_000_000
    cost_output = total_output * 15.0 / 1_000_000
    total_cost = cost_input + cost_cache_create + cost_cache_read + cost_output

    lines.append("")
    lines.append("Estimated cost (Sonnet 4.6 pricing):")
    lines.append(f"  Uncached input: ${cost_input:.4f}")
    lines.append(f"  Cache creation: ${cost_cache_create:.4f}")
    lines.append(f"  Cache reads:    ${cost_cache_read:.4f}")
    lines.append(f"  Output:         ${cost_output:.4f}")
    lines.append(f"  TOTAL:          ${total_cost:.4f}")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    # Load env vars and verify the API key exists before doing anything
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ERROR: ANTHROPIC_API_KEY not found in environment")

    # Verify input files exist before any API spend
    if not PROMPT_FILE.exists():
        raise SystemExit(f"ERROR: prompt file not found at {PROMPT_FILE}")
    if not FEATURES_FILE.exists():
        raise SystemExit(f"ERROR: feature file not found at {FEATURES_FILE}")

    # Make sure outputs/ exists
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Load and assemble the inputs once - same content for all 10 runs
    system_prompt = load_system_prompt(PROMPT_FILE)
    feature_text = load_feature_definitions(FEATURES_FILE)
    print(f"Loaded system prompt: {len(system_prompt):,} chars")
    print(f"Loaded feature definitions: {len(feature_text):,} chars")

    # Save the full assembled system prompt for reproducibility
    # If runs go sideways, this file shows exactly what the LLM saw
    prompt_snapshot_path = OUTPUTS_DIR / "system_prompt_used.txt"
    prompt_snapshot_path.write_text(system_prompt, encoding="utf-8")
    print(f"Saved assembled system prompt to {prompt_snapshot_path}")

    client = Anthropic()
    run_results: list[dict] = []

    # Run the 10 calls sequentially - each must complete within ~5 min total
    # to keep the prompt cache warm
    print(f"\nRunning {NUM_RUNS} exploration passes with {MODEL}...")
    start = time.time()

    for run_id in range(1, NUM_RUNS + 1):
        run_start = time.time()
        print(f"\n[run {run_id}/{NUM_RUNS}] sending request...")
        try:
            result = call_exploration(client, system_prompt, feature_text, run_id)
        except Exception as exc:
            # Log error and keep going - one bad run shouldn't kill the rest
            print(f"  [run {run_id}] ERROR: {exc}")
            continue

        run_results.append(result)
        elapsed = time.time() - run_start
        n_edges = len(result["edges"]) if result["edges"] else 0
        cache_read = result["usage"]["cache_read_input_tokens"]
        print(
            f"  [run {run_id}] done in {elapsed:.1f}s, "
            f"{n_edges} edges parsed, "
            f"cache_read={cache_read} tokens"
        )

        # Save raw output immediately so we don't lose it on a later crash
        out_path = OUTPUTS_DIR / f"exploration_run_{run_id:02d}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    total_elapsed = time.time() - start
    print(f"\nAll runs complete in {total_elapsed:.1f}s")

    # Aggregate edges across all successful runs
    edge_data = aggregate_runs(run_results)
    print(f"Aggregated {len(edge_data)} unique edges across runs")

    # Save aggregated edges as JSON for the next phase to consume
    aggregated_path = OUTPUTS_DIR / "aggregated_edges.json"
    serializable = [
        {
            "node_a": a,
            "node_b": b,
            "count": data["count"],
            "justifications": data["justifications"],
        }
        for (a, b), data in sorted(
            edge_data.items(), key=lambda kv: (-kv[1]["count"], kv[0])
        )
    ]
    aggregated_path.write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    print(f"Wrote aggregated edges to {aggregated_path}")

    # Write the report
    report_path = OUTPUTS_DIR / "aggregation_report.txt"
    write_aggregation_report(edge_data, run_results, report_path)
    print(f"Wrote summary report to {report_path}")


if __name__ == "__main__":
    main()