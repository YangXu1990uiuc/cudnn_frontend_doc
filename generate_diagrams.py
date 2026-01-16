#!/usr/bin/env python3
"""
Generate explanatory diagrams for cuDNN Frontend documentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# Create output directory
os.makedirs("docs/images", exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12


def create_hero_image():
    """Create the main hero image for the documentation."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(6, 5.5, 'cuDNN Frontend', fontsize=32, ha='center', fontweight='bold',
            color='#76b900')
    ax.text(6, 4.8, 'GPU-Accelerated Deep Learning Made Simple', fontsize=14, ha='center',
            color='#555555')

    # Flow: Your Code -> cuDNN Frontend -> GPU
    boxes = [
        (1.5, 2.5, 'Your Code', '#e3f2fd'),
        (5, 2.5, 'cuDNN Frontend', '#c8e6c9'),
        (8.5, 2.5, 'NVIDIA GPU', '#76b900'),
    ]

    for x, y, label, color in boxes:
        box = FancyBboxPatch((x-1, y-0.8), 2, 1.6, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='#333333', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrows
    for start_x, end_x in [(2.5, 4), (7, 7.5)]:
        ax.annotate('', xy=(end_x, 2.5), xytext=(start_x, 2.5),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

    # Labels below arrows
    ax.text(3.25, 1.8, 'Simple API', ha='center', fontsize=10, color='#666666')
    ax.text(7.25, 1.8, 'Optimized Kernels', ha='center', fontsize=10, color='#666666')

    # Features at bottom
    features = ['Graph API', 'Auto-Tuning', 'Operation Fusion', 'FP8 Support']
    for i, feat in enumerate(features):
        x = 2 + i * 2.7
        ax.text(x, 0.8, f'✓ {feat}', ha='center', fontsize=10, color='#333333')

    plt.tight_layout()
    plt.savefig('docs/images/cudnn-frontend-hero.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_graph_workflow():
    """Create a diagram showing the graph building workflow."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(5, 7.5, 'cuDNN Graph Workflow', fontsize=18, ha='center', fontweight='bold')

    steps = [
        (5, 6.5, '1. Create Graph', '#e3f2fd', 'cudnn.Graph()'),
        (5, 5.3, '2. Add Operations', '#fff9c4', 'graph.conv_fprop(), graph.sdpa()'),
        (5, 4.1, '3. Mark Outputs', '#ffe0b2', 'tensor.set_output(True)'),
        (5, 2.9, '4. Build Plans', '#f3e5f5', 'Automatic optimization'),
        (5, 1.7, '5. Execute', '#c8e6c9', 'graph(inputs, handle=handle)'),
    ]

    for x, y, label, color, desc in steps:
        box = FancyBboxPatch((x-2.5, y-0.4), 5, 0.8, boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x+3.5, y, desc, ha='left', va='center', fontsize=9, color='#666666',
                style='italic')

    # Arrows
    for i in range(len(steps) - 1):
        ax.annotate('', xy=(5, steps[i+1][1] + 0.4), xytext=(5, steps[i][1] - 0.4),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    plt.tight_layout()
    plt.savefig('docs/images/graph-workflow.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()


def create_sdpa_diagram():
    """Create a diagram explaining Scaled Dot-Product Attention."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(6, 6.5, 'Scaled Dot-Product Attention (SDPA)', fontsize=16, ha='center',
            fontweight='bold')
    ax.text(6, 6, 'softmax(Q·K^T / √d) · V', fontsize=12, ha='center',
            style='italic', color='#666666')

    # Input tensors
    inputs = [
        (1.5, 4.5, 'Q\n[B,H,N,D]', '#bbdefb'),
        (1.5, 3, 'K\n[B,H,N,D]', '#bbdefb'),
        (1.5, 1.5, 'V\n[B,H,N,D]', '#bbdefb'),
    ]

    for x, y, label, color in inputs:
        box = FancyBboxPatch((x-0.6, y-0.5), 1.2, 1, boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10)

    # Operations
    ops = [
        (4, 3.75, 'MatMul\nQ·K^T', '#fff9c4'),
        (6, 3.75, 'Scale\n÷√d', '#fff9c4'),
        (8, 3.75, 'Softmax', '#fff9c4'),
        (10, 3, 'MatMul\n·V', '#fff9c4'),
    ]

    for x, y, label, color in ops:
        box = FancyBboxPatch((x-0.6, y-0.6), 1.2, 1.2, boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=9)

    # Output
    box = FancyBboxPatch((10-0.6, 1.5-0.5), 1.2, 1, boxstyle="round,pad=0.02",
                         facecolor='#c8e6c9', edgecolor='#333333', linewidth=1.5)
    ax.add_patch(box)
    ax.text(10, 1.5, 'Output\n[B,H,N,D]', ha='center', va='center', fontsize=10)

    # Arrows
    arrows = [
        (2.1, 4.5, 3.4, 4.1),   # Q to MatMul
        (2.1, 3, 3.4, 3.4),     # K to MatMul
        (4.6, 3.75, 5.4, 3.75), # MatMul to Scale
        (6.6, 3.75, 7.4, 3.75), # Scale to Softmax
        (8.6, 3.75, 9.4, 3.4),  # Softmax to MatMul2
        (2.1, 1.5, 9.4, 2.6),   # V to MatMul2
        (10, 2.4, 10, 2),       # MatMul2 to Output
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # cuDNN optimization note
    ax.add_patch(FancyBboxPatch((3.5, 0.3), 5, 0.6, boxstyle="round,pad=0.02",
                                facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=1,
                                linestyle='--'))
    ax.text(6, 0.6, 'cuDNN fuses all operations into one optimized kernel!',
            ha='center', va='center', fontsize=10, color='#2e7d32')

    plt.tight_layout()
    plt.savefig('docs/images/sdpa-diagram.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()


def create_memory_hierarchy():
    """Create a GPU memory hierarchy diagram."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(4, 7.5, 'GPU Memory Hierarchy', fontsize=16, ha='center', fontweight='bold')

    # Layers (from fastest to slowest)
    layers = [
        (4, 6.2, 3, 0.8, 'Registers (~1 cycle)', '#76ff03'),
        (4, 5, 4, 0.8, 'Shared Memory (~20 cycles)', '#c8e6c9'),
        (4, 3.8, 5, 0.8, 'L2 Cache (~200 cycles)', '#fff9c4'),
        (4, 2.2, 6, 1.2, 'HBM/Global Memory (~400+ cycles)', '#ffcdd2'),
    ]

    for x, y, w, h, label, color in layers:
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10)

    # Labels
    ax.text(0.5, 6.2, 'Fastest\nSmallest', ha='center', va='center', fontsize=9, color='#666666')
    ax.text(0.5, 2.2, 'Slowest\nLargest', ha='center', va='center', fontsize=9, color='#666666')

    # Arrow showing speed gradient
    ax.annotate('', xy=(0.5, 5.5), xytext=(0.5, 2.8),
                arrowprops=dict(arrowstyle='<->', color='#666666', lw=2))

    # Optimization tip
    ax.add_patch(FancyBboxPatch((1, 0.3), 6, 0.8, boxstyle="round,pad=0.02",
                                facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=1))
    ax.text(4, 0.7, 'cuDNN Frontend keeps data in fast memory\nthrough operation fusion!',
            ha='center', va='center', fontsize=10, color='#2e7d32')

    plt.tight_layout()
    plt.savefig('docs/images/memory-hierarchy.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()


def create_transformer_block():
    """Create a diagram of a transformer block."""
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 10)
    ax.axis('off')

    ax.text(4, 9.5, 'Transformer Block (Pre-Norm)', fontsize=16, ha='center',
            fontweight='bold')

    # Components
    components = [
        (4, 8.5, 'Input x', '#e3f2fd', 1.8, 0.5),
        (4, 7.5, 'RMSNorm', '#fff9c4', 2.5, 0.6),
        (4, 6.3, 'Self-Attention\n(cuDNN SDPA)', '#c8e6c9', 2.5, 0.8),
        (6.5, 6.9, '+', '#ffffff', 0.5, 0.5),  # Residual add
        (4, 5, 'RMSNorm', '#fff9c4', 2.5, 0.6),
        (4, 3.8, 'FFN (SwiGLU)', '#f3e5f5', 2.5, 0.8),
        (6.5, 4.4, '+', '#ffffff', 0.5, 0.5),  # Residual add
        (4, 2.5, 'Output', '#c8e6c9', 1.8, 0.5),
    ]

    for x, y, label, color, w, h in components:
        if label == '+':
            circle = plt.Circle((x, y), 0.25, facecolor=color, edgecolor='#333333', linewidth=1.5)
            ax.add_patch(circle)
            ax.text(x, y, '+', ha='center', va='center', fontsize=14, fontweight='bold')
        else:
            box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor='#333333', linewidth=1.5)
            ax.add_patch(box)
            ax.text(x, y, label, ha='center', va='center', fontsize=9)

    # Main flow arrows
    main_flow = [(4, 8.2), (4, 7.8), (4, 7.2), (4, 6.7), (4, 5.9), (4, 5.3),
                 (4, 4.7), (4, 4.2), (4, 3.4), (4, 2.8)]
    for i in range(len(main_flow) - 1):
        if i not in [3, 7]:  # Skip where residuals merge
            ax.annotate('', xy=main_flow[i+1], xytext=main_flow[i],
                        arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # Residual connections
    # First residual (around attention)
    ax.plot([1.5, 1.5], [8.5, 6.9], 'k-', lw=1.5)
    ax.plot([1.5, 6.25], [6.9, 6.9], 'k-', lw=1.5)
    ax.annotate('', xy=(6.25, 6.9), xytext=(5.25, 6.3),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # Second residual (around FFN)
    ax.plot([1.5, 1.5], [5.9, 4.4], 'k-', lw=1.5)
    ax.plot([1.5, 6.25], [4.4, 4.4], 'k-', lw=1.5)
    ax.annotate('', xy=(6.25, 4.4), xytext=(5.25, 3.8),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # Labels for residuals
    ax.text(0.8, 7.7, 'residual', fontsize=8, color='#666666', rotation=90, va='center')
    ax.text(0.8, 5.2, 'residual', fontsize=8, color='#666666', rotation=90, va='center')

    plt.tight_layout()
    plt.savefig('docs/images/transformer-block.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()


def create_llm_vs_vlm():
    """Create a comparison diagram for LLM vs VLM architectures."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, title in zip(axes, ['LLM Architecture', 'VLM Architecture']):
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 8)
        ax.axis('off')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    # LLM Architecture
    ax = axes[0]
    llm_components = [
        (3.5, 7, 'Text Input', '#e3f2fd', 3, 0.6),
        (3.5, 6, 'Embedding', '#fff9c4', 3, 0.6),
        (3.5, 4.5, 'Transformer\nBlocks\n(N layers)', '#c8e6c9', 3, 2),
        (3.5, 2.5, 'LM Head', '#f3e5f5', 3, 0.6),
        (3.5, 1.5, 'Output Tokens', '#e3f2fd', 3, 0.6),
    ]

    for x, y, label, color, w, h in llm_components:
        box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor='#333333', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10)

    # Arrows
    for y1, y2 in [(6.7, 6.3), (5.7, 5.5), (3.5, 2.8), (2.2, 1.8)]:
        ax.annotate('', xy=(3.5, y2), xytext=(3.5, y1),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # VLM Architecture
    ax = axes[1]

    # Image branch
    ax.add_patch(FancyBboxPatch((0.5, 6.5), 2, 0.6, boxstyle="round,pad=0.02",
                                facecolor='#e3f2fd', edgecolor='#333333', linewidth=1.5))
    ax.text(1.5, 6.8, 'Image', ha='center', va='center', fontsize=10)

    ax.add_patch(FancyBboxPatch((0.5, 5.3), 2, 0.8, boxstyle="round,pad=0.02",
                                facecolor='#fff9c4', edgecolor='#333333', linewidth=1.5))
    ax.text(1.5, 5.7, 'Vision\nEncoder', ha='center', va='center', fontsize=10)

    # Text branch
    ax.add_patch(FancyBboxPatch((4.5, 6.5), 2, 0.6, boxstyle="round,pad=0.02",
                                facecolor='#e3f2fd', edgecolor='#333333', linewidth=1.5))
    ax.text(5.5, 6.8, 'Text', ha='center', va='center', fontsize=10)

    ax.add_patch(FancyBboxPatch((4.5, 5.5), 2, 0.6, boxstyle="round,pad=0.02",
                                facecolor='#fff9c4', edgecolor='#333333', linewidth=1.5))
    ax.text(5.5, 5.8, 'Embedding', ha='center', va='center', fontsize=10)

    # Merge
    ax.add_patch(FancyBboxPatch((2, 4), 3, 0.8, boxstyle="round,pad=0.02",
                                facecolor='#ffe0b2', edgecolor='#333333', linewidth=1.5))
    ax.text(3.5, 4.4, 'Projection +\nConcatenate', ha='center', va='center', fontsize=10)

    # Shared transformer
    ax.add_patch(FancyBboxPatch((1.5, 1.8), 4, 1.6, boxstyle="round,pad=0.02",
                                facecolor='#c8e6c9', edgecolor='#333333', linewidth=1.5))
    ax.text(3.5, 2.6, 'Transformer\nBlocks', ha='center', va='center', fontsize=10)

    # Output
    ax.add_patch(FancyBboxPatch((2, 0.5), 3, 0.6, boxstyle="round,pad=0.02",
                                facecolor='#e3f2fd', edgecolor='#333333', linewidth=1.5))
    ax.text(3.5, 0.8, 'Output', ha='center', va='center', fontsize=10)

    # VLM Arrows
    ax.annotate('', xy=(1.5, 5.9), xytext=(1.5, 6.5),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))
    ax.annotate('', xy=(5.5, 5.9), xytext=(5.5, 6.5),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))
    ax.annotate('', xy=(2.5, 4.4), xytext=(1.5, 5.3),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))
    ax.annotate('', xy=(4.5, 4.4), xytext=(5.5, 5.5),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))
    ax.annotate('', xy=(3.5, 3.4), xytext=(3.5, 4),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))
    ax.annotate('', xy=(3.5, 1.1), xytext=(3.5, 1.8),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    plt.tight_layout()
    plt.savefig('docs/images/llm-vs-vlm.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()


def main():
    """Generate all diagrams."""
    print("Generating documentation diagrams...")

    create_hero_image()
    print("  ✓ Hero image")

    create_graph_workflow()
    print("  ✓ Graph workflow")

    create_sdpa_diagram()
    print("  ✓ SDPA diagram")

    create_memory_hierarchy()
    print("  ✓ Memory hierarchy")

    create_transformer_block()
    print("  ✓ Transformer block")

    create_llm_vs_vlm()
    print("  ✓ LLM vs VLM comparison")

    print("\nAll diagrams generated successfully!")
    print(f"Output directory: docs/images/")


if __name__ == "__main__":
    main()
