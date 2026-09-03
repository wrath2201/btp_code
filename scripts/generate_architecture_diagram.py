import os
import subprocess

dot_source = """
digraph G {
    fontname="Helvetica,Arial,sans-serif"
    node [fontname="Helvetica,Arial,sans-serif", shape=box, style="rounded,filled", fillcolor="#f8f9fa", color="#dee2e6", penwidth=1.5]
    edge [fontname="Helvetica,Arial,sans-serif", color="#6c757d", penwidth=1.5]
    rankdir=TB
    nodesep=0.8
    ranksep=0.8
    splines=ortho

    // Input
    Input [label="Input waveform\\n(1280 samples)", shape=plaintext, fillcolor=none, color=none, fontweight=bold]

    subgraph cluster_dasnet {
        label="Stage-1-trained DASNet representation\\n(Frozen in Stage 2)"
        style="rounded,dashed"
        color="#0d6efd"
        fontcolor="#0d6efd"
        fontweight=bold
        bgcolor="#f8f9fa"
        margin=20

        ST [label="Differentiable / Adaptive\\nStockwell Transform"]
        SNR [label="SNR Estimation\\n& Conditioning"]
        CNN [label="CNN Stages with\\nFiLM Conditioning"]
        GAP [label="Global Average\\nPooling"]

        ST -> CNN
        SNR -> CNN [style=dashed, label=" condition"]
        CNN -> GAP
    }

    subgraph cluster_classical {
        label="Classical Branch\\n(Trained in Stage 2)"
        style="rounded,dashed"
        color="#198754"
        fontcolor="#198754"
        fontweight=bold
        bgcolor="#f8f9fa"
        margin=20

        Feat [label="191 Classical Features\\n(Pre-computed)"]
        MLP [label="Classical Expert\\n(MLP)"]

        Feat -> MLP
    }

    Emb [label="256-d embedding", shape=plaintext, fillcolor=none, color=none]
    
    Fusion [label="Feature Fusion Head\\n(Concat + MLP)"]
    Output [label="29-class output", shape=plaintext, fillcolor=none, color=none, fontweight=bold]

    // Connections
    Input -> ST
    Input -> SNR
    Input -> Feat [style=dotted, label=" feature extraction"]
    
    GAP -> Emb
    Emb -> Fusion [color="#0d6efd", penwidth=2, label=" FROZEN"]
    
    MLP -> Fusion [color="#198754", penwidth=2]
    
    Fusion -> Output
}
"""

os.makedirs("docs/figures", exist_ok=True)

with open("docs/figures/frozen_dualpq_architecture.dot", "w") as f:
    f.write(dot_source)

try:
    subprocess.run(["dot", "-Tpng", "docs/figures/frozen_dualpq_architecture.dot", "-o", "docs/figures/frozen_dualpq_architecture.png"], check=True)
    subprocess.run(["dot", "-Tpdf", "docs/figures/frozen_dualpq_architecture.dot", "-o", "docs/figures/frozen_dualpq_architecture.pdf"], check=True)
    print("Successfully generated PNG and PDF diagrams.")
except subprocess.CalledProcessError as e:
    print(f"Error running dot: {e}")
except FileNotFoundError:
    print("Graphviz 'dot' executable not found.")
