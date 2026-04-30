"""
Phase 3 - Causal Validation

Iteratively reviews the directed edges from Phase 2, applying:
1. LLM self-correction (false positives, false negatives, wrong directions)
2. Deterministic cycle detection via networkx
3. Targeted cycle-breaking calls when cycles remain
4. Action-history tracking to prevent oscillation between iterations

Loops until the graph is a clean DAG with no LLM-proposed corrections,
OR until iteration cap, cost cap, or 2 consecutive parse failures abort
the run.

Inputs:
    LLM Context/validation_prompt.txt           core system instructions
    LLM Context/verbose_feature_definitions.docx feature definitions
    outputs/directed_edges.json                  directed edges from Phase 2

Outputs:
    outputs/validation_prompt_used.txt           full assembled prompt
    outputs/validation_iter_NN.json              raw API response per iteration
    outputs/validated_edges.json                 final DAG edge list
    outputs/validation_report.txt                human-readable summary
"""

import json
import os
import time
from pathlib import Path

import networkx as nx
from anthropic import Anthropic
from docx import Document
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"
TEMPERATURE = 1.0
MAX_TOKENS = 16000

# Stopping criteria - whichever hits first
MAX_ITERATIONS = 8                  # hard cap on total LLM calls
MAX_COST_USD = 2.00                 # hard cap on cumulative cost
MAX_CONSECUTIVE_PARSE_FAILURES = 2  # abort if model returns broken JSON twice in a row

# Sonnet 4.6 pricing for cost tracking
PRICE_INPUT_PER_M = 3.0
PRICE_CACHE_CREATE_PER_M = 3.75
PRICE_CACHE_READ_PER_M = 0.30
PRICE_OUTPUT_PER_M = 15.0

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
CONTEXT_DIR = SCRIPT_DIR / "LLM Context"
PROMPT_FILE = CONTEXT_DIR / "validation_prompt.txt"
FEATURES_FILE = CONTEXT_DIR / "verbose_feature_definitions.docx"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
EDGES_INPUT = OUTPUTS_DIR / "directed_edges.json"

# Standard guardrails for the full-validation pass
# Tightened to forbid prose before JSON
FULL_VALIDATION_GUARDRAILS = """

=== ADDITIONAL GUIDANCE ===

The three types of mistakes you must look for, as described in the source methodology paper:
1. False negative / missing causal relationship that should be in the graph
2. False positive / spurious causal relationship that should be removed
3. Correct relationship with the wrong causal direction

You must apply special scrutiny to:
- Edges marked as bidirectional candidates (where reverse causation is plausible)
- Edges between variables related by accounting identities (e.g., the Fisher decomposition of nominal yields, where real_10y and breakeven_10y together produce the nominal yield - the causal direction depends on which variable is the directly priced market quantity)
- Edges between variables that are co-determined by shared upstream drivers (the edge may not exist; both may be effects of the same cause)
- Duplicate edges (the same source-target pair listed more than once)

=== CRITICAL OUTPUT REQUIREMENTS ===

- Respond with ONLY a single JSON object
- Do NOT include any analysis, explanation, or prose BEFORE the JSON
- Do NOT wrap the JSON in markdown code fences
- Put your full reasoning INSIDE the "justification" field of each correction
- Your response must START with the character "{" and END with the character "}"

The JSON must have this exact structure:

{
  "no_corrections_needed": <true or false>,
  "corrections": [
    {
      "type": "remove" | "add" | "reverse",
      "source": "<feature_name>",
      "target": "<feature_name>",
      "justification": "<2-3 sentence reasoning>"
    }
  ]
}

Set "no_corrections_needed": true and provide an empty corrections array if you believe the graph is correct as-is.

For type="remove": specify the source and target of the edge to remove.
For type="add": specify the source and target of the new edge to add.
For type="reverse": specify the CURRENT source and target (the script will swap them)."""

# Targeted prompt used when cycles remain
CYCLE_BREAKING_GUARDRAILS = """

=== CYCLE-BREAKING TASK ===

A directed cycle has been detected in the graph. A DAG cannot contain cycles, so you must break this cycle by EITHER removing one edge OR reversing one edge from the cycle.

You will be shown the cycle as an ordered list of edges. Choose the action that produces the most causally defensible final structure.

If the prompt notes that certain actions have already been applied (history of prior cycle-breaks), do NOT propose those same actions again. Choose a different edge or a different action type.

=== CRITICAL OUTPUT REQUIREMENTS ===

- Respond with ONLY a single JSON object
- Do NOT include any analysis, explanation, or prose BEFORE the JSON
- Do NOT wrap the JSON in markdown code fences
- Put your full reasoning INSIDE the "justification" field
- Your response must START with the character "{" and END with the character "}"

The JSON must have this exact structure:

{
  "action": "remove" | "reverse",
  "source": "<feature_name>",
  "target": "<feature_name>",
  "justification": "<reasoning explaining why this edge was chosen and why this action is the right cycle-breaker>"
}"""


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def load_validation_prompt(path: Path) -> str:
    # Read core validation prompt and append full-validation guardrails.
    core = path.read_text(encoding="utf-8").strip()
    return core + FULL_VALIDATION_GUARDRAILS


def load_cycle_breaking_prompt(path: Path) -> str:
    # Read core validation prompt and append cycle-breaking guardrails.
    core = path.read_text(encoding="utf-8").strip()
    return core + CYCLE_BREAKING_GUARDRAILS


def load_feature_definitions(path: Path) -> str:
    # Extract paragraph text from the docx feature definitions.
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_directed_edges(path: Path) -> list[dict]:
    # Load directed edges from Phase 2 output.
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Edge list manipulation
# ---------------------------------------------------------------------------


def deduplicate_edges(edges: list[dict]) -> list[dict]:
    # Remove duplicate edges (same source-target pair); keep the first
    seen = set()
    deduped = []
    for e in edges:
        key = (e["source"], e["target"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    return deduped


def apply_corrections(
    edges: list[dict], corrections: list[dict]
) -> tuple[list[dict], dict]:
    # Apply LLM-proposed corrections to the edge list.
    stats = {"removed": 0, "added": 0, "reversed": 0, "skipped": 0}

    # Build a lookup from (source, target) to edge dict for fast modification
    edge_lookup = {(e["source"], e["target"]): e for e in edges}

    for c in corrections:
        ctype = c.get("type")
        src = c.get("source")
        tgt = c.get("target")
        just = c.get("justification", "")

        if ctype == "remove":
            if (src, tgt) in edge_lookup:
                del edge_lookup[(src, tgt)]
                stats["removed"] += 1
            else:
                stats["skipped"] += 1

        elif ctype == "add":
            if (src, tgt) in edge_lookup:
                stats["skipped"] += 1
            else:
                edge_lookup[(src, tgt)] = {
                    "source": src,
                    "target": tgt,
                    "justification": just,
                    "bidirectional_candidate": False,
                }
                stats["added"] += 1

        elif ctype == "reverse":
            if (src, tgt) in edge_lookup:
                old_edge = edge_lookup.pop((src, tgt))
                edge_lookup[(tgt, src)] = {
                    "source": tgt,
                    "target": src,
                    "justification": just,
                    "bidirectional_candidate": old_edge.get(
                        "bidirectional_candidate", False
                    ),
                }
                stats["reversed"] += 1
            else:
                stats["skipped"] += 1

        else:
            stats["skipped"] += 1

    return list(edge_lookup.values()), stats


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def build_graph(edges: list[dict]) -> nx.DiGraph:
    # Build a networkx directed graph from the edge list
    g = nx.DiGraph()
    for e in edges:
        g.add_edge(e["source"], e["target"])
    return g


def find_cycles(edges: list[dict]) -> list[list[str]]:
    # Find all simple directed cycles, returned as lists of node names
    g = build_graph(edges)
    return list(nx.simple_cycles(g))


def format_cycle_for_prompt(cycle: list[str]) -> str:
    # Render a cycle as a readable arrow-chain for LLM display
    chain = cycle + [cycle[0]]
    return " -> ".join(chain)


def cycle_to_edge_pairs(cycle: list[str]) -> set[tuple[str, str]]:
    # Convert a cycle node sequence to its set of directed edge pairs
    return set(zip(cycle, cycle[1:] + [cycle[0]]))


# ---------------------------------------------------------------------------
# Action history tracking - prevents oscillation
# ---------------------------------------------------------------------------


def _is_action_repeat(
    action: str, source: str, target: str, applied_actions: list[tuple]
) -> bool:
    # Check if the exact (action, source, target) has been applied before
    return (action, source, target) in applied_actions


def _is_oscillation(
    action: str, source: str, target: str, applied_actions: list[tuple]
) -> bool:
    # Check whether this action would undo a previous action
    if action != "reverse":
        return False
    # If we previously reversed (target, source), this proposal undoes that
    return ("reverse", target, source) in applied_actions


def format_history_for_prompt(applied_actions: list[tuple]) -> str:
    # Render the action history as text the LLM can read in the prompt
    if not applied_actions:
        return "No prior cycle-breaks have been applied."
    lines = ["Prior cycle-breaks already applied (do NOT repeat any of these):"]
    for i, (action, src, tgt) in enumerate(applied_actions, start=1):
        lines.append(f"  {i}. {action} {src} -> {tgt}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON parsing - robust to prose, fences, and prose around fences
# ---------------------------------------------------------------------------


def _extract_first_json_block(text: str) -> str | None:
    # Find the first balanced JSON object or array in a string.
    start_obj = text.find("{")
    start_arr = text.find("[")
    if start_obj == -1 and start_arr == -1:
        return None
    if start_obj == -1:
        start, open_char, close_char = start_arr, "[", "]"
    elif start_arr == -1:
        start, open_char, close_char = start_obj, "{", "}"
    elif start_obj < start_arr:
        start, open_char, close_char = start_obj, "{", "}"
    else:
        start, open_char, close_char = start_arr, "[", "]"

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        char = text[i]
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_json_response(raw_text: str):
    # Parse LLM JSON response: handles clean JSON, fences, and prose+JSON
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
    extracted = _extract_first_json_block(text)
    if extracted is not None:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError as exc:
            print(f"  WARNING: extracted JSON block failed to parse: {exc}")
            return None
    print("  WARNING: no JSON object/array found in response")
    return None


# ---------------------------------------------------------------------------
# API calling
# ---------------------------------------------------------------------------


def calc_cost(usage: dict) -> float:
    # Compute USD cost from usage dict using Sonnet 4.6 pricing
    return (
        usage["input_tokens"] * PRICE_INPUT_PER_M / 1_000_000
        + usage["cache_creation_input_tokens"] * PRICE_CACHE_CREATE_PER_M / 1_000_000
        + usage["cache_read_input_tokens"] * PRICE_CACHE_READ_PER_M / 1_000_000
        + usage["output_tokens"] * PRICE_OUTPUT_PER_M / 1_000_000
    )


def usage_dict(usage_obj) -> dict:
    # Extract usage stats from API response into a plain dict
    return {
        "input_tokens": usage_obj.input_tokens,
        "output_tokens": usage_obj.output_tokens,
        "cache_creation_input_tokens": getattr(
            usage_obj, "cache_creation_input_tokens", 0
        ),
        "cache_read_input_tokens": getattr(
            usage_obj, "cache_read_input_tokens", 0
        ),
    }


def call_full_validation(
    client: Anthropic,
    system_prompt: str,
    feature_text: str,
    edges: list[dict],
) -> dict:
    # Run a full-graph validation pass: find all three mistake types
    edge_block = "\n".join(
        f"{i}. {e['source']} -> {e['target']}"
        + (" [bidirectional candidate]" if e.get("bidirectional_candidate") else "")
        for i, e in enumerate(edges, start=1)
    )

    user_msg = (
        "You are validating a directed causal graph for gold price modeling. "
        "Review the edges below and identify any errors (false positives, "
        "false negatives, or wrong directions).\n\n"
        "=== FEATURE DEFINITIONS ===\n\n"
        f"{feature_text}\n\n"
        f"=== CURRENT DIRECTED EDGES ({len(edges)} total) ===\n\n"
        f"{edge_block}\n\n"
        "=== TASK ===\n\n"
        "Identify any corrections needed. Output a JSON object as specified."
    )

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

    raw = response.content[0].text
    parsed = parse_json_response(raw)
    return {
        "raw_text": raw,
        "parsed": parsed,
        "usage": usage_dict(response.usage),
    }


def call_cycle_breaking(
    client: Anthropic,
    system_prompt: str,
    feature_text: str,
    cycle: list[str],
    cycle_edges: list[dict],
    applied_actions: list[tuple],
) -> dict:
    # Make a targeted call to break a single cycle
    cycle_str = format_cycle_for_prompt(cycle)
    edge_lines = []
    for e in cycle_edges:
        marker = " [bidirectional candidate]" if e.get("bidirectional_candidate") else ""
        edge_lines.append(
            f"  {e['source']} -> {e['target']}{marker}\n"
            f"    Justification: {e.get('justification', '')[:200]}"
        )
    edge_block = "\n".join(edge_lines)
    history_block = format_history_for_prompt(applied_actions)

    user_msg = (
        "A directed cycle has been detected in the causal graph. You must "
        "break this cycle by removing or reversing exactly ONE edge from the "
        "cycle.\n\n"
        "=== FEATURE DEFINITIONS ===\n\n"
        f"{feature_text}\n\n"
        f"=== CYCLE ===\n\n"
        f"{cycle_str}\n\n"
        "=== EDGES IN THIS CYCLE ===\n\n"
        f"{edge_block}\n\n"
        "=== ACTION HISTORY ===\n\n"
        f"{history_block}\n\n"
        "=== TASK ===\n\n"
        "Choose one edge to remove or reverse. Do not propose any action "
        "from the action history above. Output a JSON object as specified."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        temperature=TEMPERATURE,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text
    parsed = parse_json_response(raw)
    return {
        "raw_text": raw,
        "parsed": parsed,
        "usage": usage_dict(response.usage),
    }


def is_valid_cycle_action(action: dict) -> bool:
    # Check that the cycle-breaking action has all required fields
    if not isinstance(action, dict):
        return False
    if action.get("action") not in ("remove", "reverse"):
        return False
    if not action.get("source") or not action.get("target"):
        return False
    return True


def apply_cycle_break(edges: list[dict], action: dict) -> list[dict]:
    # Apply a single cycle-breaking action to the edge list
    correction = {
        "type": action.get("action"),
        "source": action.get("source"),
        "target": action.get("target"),
        "justification": action.get("justification", ""),
    }
    new_edges, _ = apply_corrections(edges, [correction])
    return new_edges


def deterministic_tiebreaker(
    cycle: list[str], cycle_edges: list[dict]
) -> dict:
    # Last-resort cycle break when LLM keeps oscillating
    # Sort edges in the cycle by justification length, longest first
    edges_sorted = sorted(
        cycle_edges,
        key=lambda e: len(e.get("justification", "")),
        reverse=True,
    )
    target_edge = edges_sorted[0]
    return {
        "action": "remove",
        "source": target_edge["source"],
        "target": target_edge["target"],
        "justification": (
            "Deterministic tiebreaker applied after LLM oscillation. Removed "
            "the edge with the longest justification text in the cycle as a "
            "proxy for least-confident edge."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_validation_report(
    edges: list[dict],
    iterations_log: list[dict],
    final_status: str,
    total_cost: float,
    applied_actions: list[tuple],
    path: Path,
) -> None:
    # Write a summary of the validation run
    g = build_graph(edges)
    is_dag = nx.is_directed_acyclic_graph(g)
    remaining_cycles = list(nx.simple_cycles(g)) if not is_dag else []

    lines = []
    lines.append("=" * 70)
    lines.append("Causal Validation Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Model: {MODEL}")
    lines.append(f"Final status: {final_status}")
    lines.append(f"Total iterations: {len(iterations_log)}")
    lines.append(f"Total cost: ${total_cost:.4f}")
    lines.append(f"Cost cap: ${MAX_COST_USD:.2f}")
    lines.append(f"Final edge count: {len(edges)}")
    lines.append(f"Final graph is DAG: {is_dag}")
    if remaining_cycles:
        lines.append(f"Remaining cycles: {len(remaining_cycles)}")
        for c in remaining_cycles:
            lines.append(f"  - {format_cycle_for_prompt(c)}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("Iteration log")
    lines.append("=" * 70)
    for it in iterations_log:
        lines.append(f"  Iter {it['iteration']:2d} ({it['type']:20s}): "
                     f"{it['summary']}, cost ${it['cost']:.4f}")
    lines.append("")

    if applied_actions:
        lines.append("=" * 70)
        lines.append("Applied cycle-break actions (in order)")
        lines.append("=" * 70)
        for i, (action, src, tgt) in enumerate(applied_actions, start=1):
            lines.append(f"  {i}. {action} {src} -> {tgt}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("Final directed edges")
    lines.append("=" * 70)
    for e in edges:
        marker = " [BIDIR]" if e.get("bidirectional_candidate") else ""
        lines.append(f"  {e['source']} -> {e['target']}{marker}")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ERROR: ANTHROPIC_API_KEY not found in environment")

    if not PROMPT_FILE.exists():
        raise SystemExit(f"ERROR: prompt file not found at {PROMPT_FILE}")
    if not FEATURES_FILE.exists():
        raise SystemExit(f"ERROR: feature file not found at {FEATURES_FILE}")
    if not EDGES_INPUT.exists():
        raise SystemExit(
            f"ERROR: directed edges not found at {EDGES_INPUT}. "
            "Run 02_inference.py first."
        )

    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Load inputs
    full_validation_prompt = load_validation_prompt(PROMPT_FILE)
    cycle_break_prompt = load_cycle_breaking_prompt(PROMPT_FILE)
    feature_text = load_feature_definitions(FEATURES_FILE)
    edges = load_directed_edges(EDGES_INPUT)

    # Always deduplicate at the start since Phase 2 produced a duplicate
    initial_count = len(edges)
    edges = deduplicate_edges(edges)
    if len(edges) < initial_count:
        print(f"Removed {initial_count - len(edges)} duplicate edges at startup")

    print(f"Loaded {len(edges)} unique directed edges")

    # Save the full assembled prompt for reproducibility
    prompt_snapshot = OUTPUTS_DIR / "validation_prompt_used.txt"
    prompt_snapshot.write_text(full_validation_prompt, encoding="utf-8")

    client = Anthropic()
    iterations_log: list[dict] = []
    applied_actions: list[tuple] = []  # tracks (action, source, target) tuples
    total_cost = 0.0
    iteration = 0
    consecutive_parse_failures = 0
    final_status = "unknown"

    # Main loop: alternate between full validation and cycle breaking
    while iteration < MAX_ITERATIONS and total_cost < MAX_COST_USD:
        iteration += 1
        cycles = find_cycles(edges)

        if cycles:
            # Cycles present - try LLM cycle breaks, fall back to tiebreaker
            print(f"\n[iter {iteration}] {len(cycles)} cycle(s) detected")

            # Sort cycles longest first - if longest is unbreakable by LLM (proposes repeats)
            cycles_sorted = sorted(cycles, key=len, reverse=True)
            applied_break = False
            iter_cost = 0.0
            tried_cycles_count = 0

            for cycle in cycles_sorted:
                tried_cycles_count += 1
                cycle_edge_pairs = cycle_to_edge_pairs(cycle)
                cycle_edges = [
                    e for e in edges
                    if (e["source"], e["target"]) in cycle_edge_pairs
                ]

                print(f"  Trying cycle: {format_cycle_for_prompt(cycle)}")

                try:
                    result = call_cycle_breaking(
                        client, cycle_break_prompt, feature_text,
                        cycle, cycle_edges, applied_actions,
                    )
                except Exception as exc:
                    print(f"  ERROR: cycle break call failed: {exc}")
                    final_status = f"error: {exc}"
                    break

                cost = calc_cost(result["usage"])
                total_cost += cost
                iter_cost += cost

                # Save raw response for debugging
                raw_path = OUTPUTS_DIR / (
                    f"validation_iter_{iteration:02d}_"
                    f"cycle{tried_cycles_count:02d}.json"
                )
                raw_path.write_text(
                    json.dumps(result, indent=2), encoding="utf-8"
                )

                parsed = result["parsed"]
                if parsed is None or not is_valid_cycle_action(parsed):
                    consecutive_parse_failures += 1
                    reason = "parse error" if parsed is None else "missing fields"
                    print(f"  Invalid response ({reason}), consecutive: "
                          f"{consecutive_parse_failures}/"
                          f"{MAX_CONSECUTIVE_PARSE_FAILURES}")
                    if consecutive_parse_failures >= MAX_CONSECUTIVE_PARSE_FAILURES:
                        final_status = (
                            f"aborted: {MAX_CONSECUTIVE_PARSE_FAILURES} "
                            f"consecutive parse failures"
                        )
                        break
                    continue  # try next cycle in the sorted list

                # Reset failure counter on any successful parse
                consecutive_parse_failures = 0

                # Check action history for repeats and oscillation
                action = parsed["action"]
                src = parsed["source"]
                tgt = parsed["target"]

                if _is_action_repeat(action, src, tgt, applied_actions):
                    print(f"  REPEAT detected: {action} {src} -> {tgt} "
                          f"already applied - trying next cycle")
                    continue

                if _is_oscillation(action, src, tgt, applied_actions):
                    print(f"  OSCILLATION detected: {action} {src} -> {tgt} "
                          f"would undo prior action - trying next cycle")
                    continue

                # Action is novel - apply it
                edges = apply_cycle_break(edges, parsed)
                applied_actions.append((action, src, tgt))
                action_str = f"{action} {src} -> {tgt}"
                print(f"  Applied: {action_str}, iter cost ${iter_cost:.4f}")
                iterations_log.append({
                    "iteration": iteration,
                    "type": "cycle_break",
                    "summary": action_str,
                    "cost": iter_cost,
                })
                applied_break = True
                break  # exit the cycles_sorted loop, move to next iteration

            # If all cycles were processed and none yielded a novel LLM action,
            # apply the deterministic tiebreaker on the longest cycle
            if not applied_break and final_status == "unknown":
                if consecutive_parse_failures >= MAX_CONSECUTIVE_PARSE_FAILURES:
                    pass  # status already set, will exit loop below
                else:
                    print(f"  All {len(cycles_sorted)} cycles produced "
                          f"repeat/oscillation actions - applying "
                          f"deterministic tiebreaker")
                    longest = cycles_sorted[0]
                    longest_edge_pairs = cycle_to_edge_pairs(longest)
                    longest_cycle_edges = [
                        e for e in edges
                        if (e["source"], e["target"]) in longest_edge_pairs
                    ]
                    tiebreak_action = deterministic_tiebreaker(
                        longest, longest_cycle_edges
                    )
                    edges = apply_cycle_break(edges, tiebreak_action)
                    applied_actions.append((
                        tiebreak_action["action"],
                        tiebreak_action["source"],
                        tiebreak_action["target"],
                    ))
                    action_str = (
                        f"TIEBREAK {tiebreak_action['action']} "
                        f"{tiebreak_action['source']} -> "
                        f"{tiebreak_action['target']}"
                    )
                    print(f"  {action_str}")
                    iterations_log.append({
                        "iteration": iteration,
                        "type": "cycle_break_tiebreak",
                        "summary": action_str,
                        "cost": iter_cost,
                    })

            if final_status != "unknown":
                break

        else:
            # No cycles - run full LLM validation pass
            print(f"\n[iter {iteration}] graph is acyclic, running full "
                  f"validation pass...")

            try:
                result = call_full_validation(
                    client, full_validation_prompt, feature_text, edges
                )
            except Exception as exc:
                print(f"  ERROR: full validation call failed: {exc}")
                final_status = f"error: {exc}"
                break

            cost = calc_cost(result["usage"])
            total_cost += cost

            raw_path = OUTPUTS_DIR / f"validation_iter_{iteration:02d}_full.json"
            raw_path.write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )

            if result["parsed"] is None:
                consecutive_parse_failures += 1
                print(f"  ERROR: failed to parse, consecutive failures: "
                      f"{consecutive_parse_failures}/"
                      f"{MAX_CONSECUTIVE_PARSE_FAILURES}")
                iterations_log.append({
                    "iteration": iteration,
                    "type": "full_validation (failed)",
                    "summary": "JSON parse error",
                    "cost": cost,
                })
                if consecutive_parse_failures >= MAX_CONSECUTIVE_PARSE_FAILURES:
                    final_status = (
                        f"aborted: {MAX_CONSECUTIVE_PARSE_FAILURES} "
                        f"consecutive parse failures"
                    )
                    break
                continue

            consecutive_parse_failures = 0
            parsed = result["parsed"]
            corrections = parsed.get("corrections", [])
            no_corrections = parsed.get("no_corrections_needed", False)

            if no_corrections or not corrections:
                print(f"  No corrections needed - validation complete, "
                      f"cost ${cost:.4f}")
                iterations_log.append({
                    "iteration": iteration,
                    "type": "full_validation",
                    "summary": "no corrections needed (DONE)",
                    "cost": cost,
                })
                final_status = "complete: clean DAG, no corrections needed"
                break

            edges, stats = apply_corrections(edges, corrections)
            edges = deduplicate_edges(edges)
            summary = (f"removed={stats['removed']}, added={stats['added']}, "
                       f"reversed={stats['reversed']}, skipped={stats['skipped']}")
            print(f"  Applied corrections: {summary}, cost ${cost:.4f}")

            iterations_log.append({
                "iteration": iteration,
                "type": "full_validation",
                "summary": summary,
                "cost": cost,
            })

        print(f"  Cumulative cost: ${total_cost:.4f} / ${MAX_COST_USD:.2f}")

    # Determine final status if not already set by an exit condition
    if final_status == "unknown":
        if iteration >= MAX_ITERATIONS:
            final_status = f"stopped: hit max iterations ({MAX_ITERATIONS})"
        elif total_cost >= MAX_COST_USD:
            final_status = f"stopped: hit cost cap (${MAX_COST_USD:.2f})"
        else:
            final_status = "stopped: unknown reason"

    print(f"\n{final_status}")
    print(f"Total cost: ${total_cost:.4f}")

    # Save final edge list - this is the artifact for Phase 4
    final_edges_path = OUTPUTS_DIR / "validated_edges.json"
    final_edges_path.write_text(json.dumps(edges, indent=2), encoding="utf-8")
    print(f"Saved final DAG to {final_edges_path}")

    # Write report
    report_path = OUTPUTS_DIR / "validation_report.txt"
    write_validation_report(
        edges, iterations_log, final_status, total_cost,
        applied_actions, report_path,
    )
    print(f"Wrote summary report to {report_path}")


if __name__ == "__main__":
    main()