"""
Phase 2 - Causal Inference

Takes the aggregated undirected edges from Phase 1 and asks Claude to assign
a direction to each one, producing a directed graph (intended to be a DAG).
For edges where both directions seem plausible, the LLM is instructed to
commit to a single direction and flag the edge as a bidirectional candidate
for later scrutiny.

Inputs:
    LLM Context/inference_prompt.txt   core system instructions for inference
    LLM Context/verbose_feature_definitions.docx  feature definitions
    outputs/aggregated_edges.json      undirected edges from Phase 1

Outputs:
    outputs/inference_prompt_used.txt  full assembled prompt (for reproducibility)
    outputs/inference_response.json    raw API response with directed edges
    outputs/directed_edges.json        cleaned directed edge list for Phase 3
    outputs/inference_report.txt       summary
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

# Paper uses a single inference call per group at temp=1
# Keeping temp=1 for consistency
MODEL = "claude-sonnet-4-6"
TEMPERATURE = 1.0
MAX_TOKENS = 16000  # higher than Phase 1 because output includes many directed edges with longer justifications

# Filter edges >= MIN_RUN_COUNT runs
# This drops the noisiest tail (edges in 1-2 runs) while preserving variance
MIN_RUN_COUNT = 3

# Paths - all relative to the script location (project root)
SCRIPT_DIR = Path(__file__).resolve().parent
CONTEXT_DIR = SCRIPT_DIR / "LLM Context"
PROMPT_FILE = CONTEXT_DIR / "inference_prompt.txt"
FEATURES_FILE = CONTEXT_DIR / "verbose_feature_definitions.docx"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
EDGES_INPUT = OUTPUTS_DIR / "aggregated_edges.json"

# Standard guardrails appended to user-editable core prompt
# Enforces DAG structure, output format, and naming discipline
APPENDED_GUARDRAILS = """

=== CYCLE-BREAKING REQUIREMENT ===

A Directed Acyclic Graph (DAG) cannot contain cycles. Even if both directions of causality seem plausible for an edge (e.g., high gold prices reduce jewelry demand AND high jewelry demand supports gold prices), you MUST commit to a single direction per edge. Choose the direction with the stronger or more dominant causal mechanism.

When both directions have plausible mechanisms, mark the edge with `"bidirectional_candidate": true` so that this edge can be flagged for additional scrutiny in the validation phase. Otherwise mark `"bidirectional_candidate": false`.

=== ADDITIONAL GUIDANCE ===

For each edge in the input list, you must produce exactly one directed edge in the output. Do not add new edges that were not in the input list. Do not omit edges from the input list - even if you believe an edge is implausible, assign a direction and explain in the justification why the edge is weak.

Use the exact feature names provided in the feature definitions (e.g., `real_10y`, `fed_funds`, `gold_price`) for source and target. Do not use display names, abbreviations, or paraphrases.

The justification should explain WHY you chose this direction, especially for edges where reverse causation is plausible. Keep justifications to 2-3 sentences.

=== EXAMPLE OUTPUT FORMAT ===

[
  {"source": "real_10y", "target": "gold_price", "justification": "Real yields are determined by Treasury market dynamics and Fed expectations independent of gold prices, while gold's value responds to changes in real opportunity cost. The dominant causal direction runs from real yields to gold.", "bidirectional_candidate": false},
  {"source": "gold_price", "target": "jewelry_demand", "justification": "Gold prices are set in liquid global markets primarily by macro and investment flows, while jewelry demand is a consumer response to those prices. Although high jewelry demand can also support prices, the price-to-quantity direction is the stronger and more immediate channel.", "bidirectional_candidate": true}
]"""


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_system_prompt(path: Path) -> str:
    # Read core inference instructions and append guardrails
    core = path.read_text(encoding="utf-8").strip()
    return core + APPENDED_GUARDRAILS


def load_feature_definitions(path: Path) -> str:
    # Extract all paragraph text from the docx feature definitions
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_filtered_edges(path: Path, min_count: int) -> list[dict]:
    # Load aggregated edges from Phase 1, filtering to those that appeared
    # in at least min_count runs. Drops the noisiest tail of low-frequency edges.

    all_edges = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in all_edges if e["count"] >= min_count]


def format_edges_for_prompt(edges: list[dict]) -> str:
    # Render the edge list as a readable list for the LLM prompt.
    lines = []
    for i, edge in enumerate(edges, start=1):
        # Use the first justification as a representative summary
        # Multiple justifications were provided per edge; one is enough context
        sample_just = edge["justifications"][0]["text"] if edge["justifications"] else ""
        lines.append(
            f"{i}. {edge['node_a']} <-> {edge['node_b']} "
            f"(consensus: {edge['count']}/10 runs)\n"
            f"   Sample mechanism: {sample_just}"
        )
    return "\n\n".join(lines)


def build_user_message(feature_text: str, edges: list[dict]) -> str:
    # Construct the user message: feature definitions + edges to direct + task.
    edge_block = format_edges_for_prompt(edges)
    return (
        "You are completing the second phase of constructing a causal DAG for "
        "gold price modeling. The undirected edge list below was produced by "
        "running Phase 1 (Causal Exploration) ten times with sampling and "
        "aggregating the results. Edges with higher consensus counts were "
        "proposed in more of the 10 independent runs.\n\n"
        "=== FEATURE DEFINITIONS ===\n\n"
        f"{feature_text}\n\n"
        f"=== UNDIRECTED EDGES TO DIRECT ({len(edges)} total) ===\n\n"
        f"{edge_block}\n\n"
        "=== TASK ===\n\n"
        "For each of the edges above, assign a direction (source -> target) "
        "and provide a brief justification. Mark edges as bidirectional "
        "candidates where both directions have plausible mechanisms. Output "
        "a JSON array as specified in your instructions."
    )


# ---------------------------------------------------------------------------
# API call - single inference pass
# ---------------------------------------------------------------------------


def parse_edges_from_response(raw_text: str) -> list | None:
    # Parse the model's JSON response, stripping markdown code fences if present.
    text = raw_text.strip()

    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as exc:
        print(f"  WARNING: malformed JSON, error: {exc}")
        return None


def call_inference(
    client: Anthropic,
    system_prompt: str,
    feature_text: str,
    edges: list[dict],
) -> dict:
    # Make the single inference API call.
    user_msg = build_user_message(feature_text, edges)

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

    raw_text = response.content[0].text
    directed_edges = parse_edges_from_response(raw_text)

    usage = response.usage
    return {
        "model": MODEL,
        "raw_text": raw_text,
        "directed_edges": directed_edges,
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
# Validation and reporting
# ---------------------------------------------------------------------------


def validate_directed_edges(
    directed_edges: list[dict], input_edges: list[dict]
) -> dict:
    # Sanity-check the LLM output against the input edges.
    # Build set of canonical input edge pairs for comparison
    input_pairs = set()
    for e in input_edges:
        input_pairs.add(tuple(sorted([e["node_a"], e["node_b"]])))

    output_pairs = set()
    bidirectional_count = 0
    self_loops = []
    for e in directed_edges:
        if "source" not in e or "target" not in e:
            continue
        if e["source"] == e["target"]:
            self_loops.append(e)
            continue
        output_pairs.add(tuple(sorted([e["source"], e["target"]])))
        if e.get("bidirectional_candidate"):
            bidirectional_count += 1

    # Check for missing edges (in input but not output)
    missing = input_pairs - output_pairs
    extra = output_pairs - input_pairs

    return {
        "n_input_edges": len(input_edges),
        "n_output_edges": len(directed_edges),
        "n_unique_output_pairs": len(output_pairs),
        "n_missing": len(missing),
        "n_extra": len(extra),
        "n_self_loops": len(self_loops),
        "n_bidirectional_candidates": bidirectional_count,
        "missing_edges": [f"{a} <-> {b}" for a, b in missing],
        "extra_edges": [f"{a} <-> {b}" for a, b in extra],
        "self_loops": self_loops,
    }


def write_inference_report(
    directed_edges: list[dict],
    input_edges: list[dict],
    validation: dict,
    usage: dict,
    path: Path,
) -> None:
    # Write a human-readable summary of the inference results.
    lines = []
    lines.append("=" * 70)
    lines.append("Causal Inference - Direction Assignment Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Model: {MODEL}")
    lines.append(f"Min run count filter: >= {MIN_RUN_COUNT}/10 runs")
    lines.append(f"Input edges (after filter): {validation['n_input_edges']}")
    lines.append(f"Output directed edges: {validation['n_output_edges']}")
    lines.append(
        f"Unique output edge pairs: {validation['n_unique_output_pairs']}"
    )
    lines.append("")

    lines.append("=" * 70)
    lines.append("Validation checks")
    lines.append("=" * 70)
    lines.append(f"Missing edges (input but not output): {validation['n_missing']}")
    if validation["missing_edges"]:
        for m in validation["missing_edges"]:
            lines.append(f"  - {m}")
    lines.append(f"Extra edges (output but not input): {validation['n_extra']}")
    if validation["extra_edges"]:
        for e in validation["extra_edges"]:
            lines.append(f"  - {e}")
    lines.append(f"Self-loops: {validation['n_self_loops']}")
    lines.append(
        f"Bidirectional candidates flagged: {validation['n_bidirectional_candidates']}"
    )
    lines.append("")

    # List bidirectional candidates separately for visibility
    if validation["n_bidirectional_candidates"] > 0:
        lines.append("=" * 70)
        lines.append("Bidirectional candidate edges (flagged for validation)")
        lines.append("=" * 70)
        for e in directed_edges:
            if e.get("bidirectional_candidate"):
                lines.append(f"  {e['source']} -> {e['target']}")
                lines.append(f"    Reason: {e.get('justification', '')[:150]}")
        lines.append("")

    # Full directed edge list
    lines.append("=" * 70)
    lines.append("All directed edges")
    lines.append("=" * 70)
    for e in directed_edges:
        marker = " [BIDIR]" if e.get("bidirectional_candidate") else ""
        lines.append(f"  {e['source']} -> {e['target']}{marker}")

    # Token usage and cost
    lines.append("")
    lines.append("=" * 70)
    lines.append("Token usage and cost")
    lines.append("=" * 70)
    lines.append(f"Uncached input tokens: {usage['input_tokens']:,}")
    lines.append(
        f"Cache creation tokens: {usage['cache_creation_input_tokens']:,}"
    )
    lines.append(f"Cache read tokens: {usage['cache_read_input_tokens']:,}")
    lines.append(f"Output tokens: {usage['output_tokens']:,}")

    # Cost calculation - same Sonnet 4.6 pricing as Phase 1
    cost_input = usage["input_tokens"] * 3.0 / 1_000_000
    cost_cache_create = (
        usage["cache_creation_input_tokens"] * 3.75 / 1_000_000
    )
    cost_cache_read = usage["cache_read_input_tokens"] * 0.30 / 1_000_000
    cost_output = usage["output_tokens"] * 15.0 / 1_000_000
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
    # Verify env and inputs before doing anything that costs money
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ERROR: ANTHROPIC_API_KEY not found in environment")

    if not PROMPT_FILE.exists():
        raise SystemExit(f"ERROR: prompt file not found at {PROMPT_FILE}")
    if not FEATURES_FILE.exists():
        raise SystemExit(f"ERROR: feature file not found at {FEATURES_FILE}")
    if not EDGES_INPUT.exists():
        raise SystemExit(
            f"ERROR: aggregated edges not found at {EDGES_INPUT}. "
            "Run 01_exploration.py first."
        )

    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Load and assemble inputs
    system_prompt = load_system_prompt(PROMPT_FILE)
    feature_text = load_feature_definitions(FEATURES_FILE)
    filtered_edges = load_filtered_edges(EDGES_INPUT, MIN_RUN_COUNT)

    # Show what's about to happen so user can sanity-check before API spend
    all_edges = json.loads(EDGES_INPUT.read_text(encoding="utf-8"))
    print(f"Loaded system prompt: {len(system_prompt):,} chars")
    print(f"Loaded feature definitions: {len(feature_text):,} chars")
    print(f"Loaded edges: {len(all_edges)} total, "
          f"{len(filtered_edges)} after >= {MIN_RUN_COUNT}/10 filter")

    # Save the assembled prompt for reproducibility
    prompt_snapshot_path = OUTPUTS_DIR / "inference_prompt_used.txt"
    prompt_snapshot_path.write_text(system_prompt, encoding="utf-8")
    print(f"Saved assembled inference prompt to {prompt_snapshot_path}")

    # Make the single inference call
    client = Anthropic()
    print(f"\nCalling {MODEL} for direction assignment...")
    start = time.time()
    result = call_inference(client, system_prompt, feature_text, filtered_edges)
    elapsed = time.time() - start
    print(f"  done in {elapsed:.1f}s")

    # Save raw response immediately
    response_path = OUTPUTS_DIR / "inference_response.json"
    response_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved raw response to {response_path}")

    if result["directed_edges"] is None:
        print("ERROR: failed to parse directed edges from response")
        print("Check inference_response.json for the raw text")
        return

    directed_edges = result["directed_edges"]
    print(f"Parsed {len(directed_edges)} directed edges")

    # Validate and report
    validation = validate_directed_edges(directed_edges, filtered_edges)
    print(f"Validation: {validation['n_missing']} missing, "
          f"{validation['n_extra']} extra, "
          f"{validation['n_self_loops']} self-loops, "
          f"{validation['n_bidirectional_candidates']} bidirectional candidates")

    # Save the cleaned directed edges as the artifact for Phase 3
    directed_path = OUTPUTS_DIR / "directed_edges.json"
    directed_path.write_text(
        json.dumps(directed_edges, indent=2), encoding="utf-8"
    )
    print(f"Saved directed edges to {directed_path}")

    # Write the human-readable report
    report_path = OUTPUTS_DIR / "inference_report.txt"
    write_inference_report(
        directed_edges, filtered_edges, validation, result["usage"], report_path
    )
    print(f"Wrote summary report to {report_path}")


if __name__ == "__main__":
    main()