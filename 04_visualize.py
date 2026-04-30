"""
Phase 4 - Visualize the validated DAG

Renders the validated edges as a node-link graph PNG, similar in style
to Exhibits 6/7 in the source methodology paper (credit spread DAG).

Inputs:
    outputs/validated_edges.json   final DAG from Phase 3

Outputs:
    outputs/dag_visualization.png  PNG visualization of the DAG

"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
EDGES_INPUT = OUTPUTS_DIR / "validated_edges.json"
OUTPUT_PNG = OUTPUTS_DIR / "dag_visualization.png"

# Visual styling - large figure for readability with many nodes
FIG_WIDTH_INCHES = 18
FIG_HEIGHT_INCHES = 14
DPI = 200

# Target node gets a distinctive color so it's easy to spot 
TARGET_NODE = "gold_price"
TARGET_COLOR = "#ffd166"   # warm yellow for gold target
NORMAL_COLOR = "#a8dadc"   # light teal for other nodes
BIDIR_EDGE_COLOR = "#e63946"  # red for bidirectional candidates
NORMAL_EDGE_COLOR = "#1d3557"  # dark navy for normal edges


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_validated_edges(path: Path) -> list[dict]:
    # Load the final DAG edges from Phase 3 output
    return json.loads(path.read_text(encoding="utf-8"))


def build_graph(edges: list[dict]) -> nx.DiGraph:
    # Build a networkx DiGraph from the edge list, preserving edge metadata
    g = nx.DiGraph()
    for e in edges:
        g.add_edge(
            e["source"],
            e["target"],
            bidirectional_candidate=e.get("bidirectional_candidate", False),
        )
    return g


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_dag(g: nx.DiGraph, output_path: Path) -> None:
    # Render the DAG using a hierarchical layout.
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))

    # Try graphviz hierarchical layout with explicit top-to-bottom orientation
    # so root causes (gpr, debt_gdp) sit at top and sinks (gold_price) at bottom
    try:
        import pydot
        # Build pydot graph manually so we can set rankdir=TB
        pdot = pydot.Dot(graph_type="digraph", rankdir="TB")
        for node in g.nodes():
            pdot.add_node(pydot.Node(str(node)))
        for u, v in g.edges():
            pdot.add_edge(pydot.Edge(str(u), str(v)))
        # Get positions from graphviz dot layout
        layout_str = pdot.create_dot(prog="dot").decode("utf-8")
        layout_graph = pydot.graph_from_dot_data(layout_str)[0]
        pos = {}
        for node in layout_graph.get_nodes():
            name = node.get_name().strip('"')
            if "pos" in node.get_attributes():
                x, y = node.get_attributes()["pos"].strip('"').split(",")
                pos[name] = (float(x), float(y))
        # Sanity check we got positions for all nodes
        if len(pos) < g.number_of_nodes():
            raise ValueError("graphviz did not return positions for all nodes")
    except Exception as exc:
        print(f"Note: graphviz layout failed ({exc}), falling back to spring layout")
        pos = nx.spring_layout(g, k=2.0, iterations=200, seed=42)

    # Color nodes - target node gets distinct color for easy ID
    node_colors = [
        TARGET_COLOR if node == TARGET_NODE else NORMAL_COLOR
        for node in g.nodes()
    ]

    # Color edges - bidirectional candidates get red to flag visually
    edge_colors = [
        BIDIR_EDGE_COLOR if g[u][v].get("bidirectional_candidate")
        else NORMAL_EDGE_COLOR
        for u, v in g.edges()
    ]

    # Draw nodes
    nx.draw_networkx_nodes(
        g, pos,
        node_color=node_colors,
        node_size=2400,
        edgecolors="black",
        linewidths=1.5,
        ax=ax,
    )

    # Draw edges with arrows
    nx.draw_networkx_edges(
        g, pos,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=25,
        arrowstyle="-|>",
        width=1.6,
        alpha=0.75,
        connectionstyle="arc3,rad=0.08",
        node_size=2400,
        min_source_margin=15,
        min_target_margin=15,
        ax=ax,
    )

    # Draw node labels
    nx.draw_networkx_labels(
        g, pos,
        font_size=9,
        font_weight="bold",
        ax=ax,
    )

    # Build legend so the colors mean something to the reader
    legend_elements = [
        plt.scatter([], [], s=200, c=TARGET_COLOR, edgecolors="black",
                    label=f"Target: {TARGET_NODE}"),
        plt.scatter([], [], s=200, c=NORMAL_COLOR, edgecolors="black",
                    label="Feature node"),
        plt.Line2D([0], [0], color=NORMAL_EDGE_COLOR, lw=2,
                   label="Directed edge"),
        plt.Line2D([0], [0], color=BIDIR_EDGE_COLOR, lw=2,
                   label="Bidirectional candidate"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11,
              frameon=True, facecolor="white", edgecolor="gray")

    # Title with node and edge counts so the reader knows scale at a glance
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    n_bidir = sum(
        1 for _, _, d in g.edges(data=True)
        if d.get("bidirectional_candidate")
    )
    is_dag = nx.is_directed_acyclic_graph(g)
    dag_status = "DAG verified" if is_dag else "NOT A DAG (cycles present)"

    ax.set_title(
        f"Gold Price Causal DAG\n"
        f"{n_nodes} nodes, {n_edges} edges "
        f"({n_bidir} bidirectional candidates) | {dag_status}",
        fontsize=14, fontweight="bold", pad=20,
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not EDGES_INPUT.exists():
        raise SystemExit(
            f"ERROR: validated edges not found at {EDGES_INPUT}. "
            "Run 03_validation.py first."
        )

    edges = load_validated_edges(EDGES_INPUT)
    g = build_graph(edges)

    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    is_dag = nx.is_directed_acyclic_graph(g)

    print(f"Loaded {n_edges} edges across {n_nodes} nodes")
    print(f"Graph is DAG: {is_dag}")

    if not is_dag:
        cycles = list(nx.simple_cycles(g))
        print(f"WARNING: {len(cycles)} cycle(s) present, visualization will "
              f"still render but DAG structure is invalid")

    render_dag(g, OUTPUT_PNG)
    print(f"Saved visualization to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()