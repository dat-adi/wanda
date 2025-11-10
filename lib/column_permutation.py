"""
Column permutation module for Wanda pruning system.

This module performs exhaustive sampling without replacement to group and reorder
matrix columns by Hamming distance similarity. It creates groups of 8 columns where
each group contains features that are spatially close in Hamming space, enabling
efficient kernel processing with banded workload patterns.

Usage:
    After weight pruning (line 206 in lib/prune.py), call:

    permute_and_visualize(
        weight_matrix=subset[name].weight.data,
        layer_idx=i,
        layer_name=name,
        output_dirs={'metrics': './metrics', 'images': './images'}
    )
"""

import torch
import random
from pathlib import Path
from typing import Dict, Tuple, List, Set
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def compute_hamming_distance_batch(vec: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Compute Hamming distances between one vector and all columns of a matrix."""
    binary_vec = (vec != 0).int().unsqueeze(1)
    binary_matrix = (matrix != 0).int()
    distances = (binary_vec != binary_matrix).float().mean(dim=0)
    return distances


def find_nearest_neighbors_excluding(
    matrix: torch.Tensor,
    feature_idx: int,
    n_neighbors: int,
    excluded_indices: Set[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find n nearest neighbors to a feature, excluding already-used features."""
    reference_vec = matrix[:, feature_idx]
    distances = compute_hamming_distance_batch(reference_vec, matrix)

    # Exclude already-used indices
    for idx in excluded_indices:
        distances[idx] = float('inf')

    sorted_indices = torch.argsort(distances)
    valid_indices = sorted_indices[distances[sorted_indices] != float('inf')]
    top_indices = valid_indices[:n_neighbors + 1]

    return top_indices, distances[top_indices]


def compute_group_metrics(matrix: torch.Tensor, indices: List[int]) -> Dict:
    """Compute detailed metrics for a group of columns."""
    subset = matrix[:, indices]
    binary_subset = (subset != 0).int()

    # Row-wise metrics
    zero_rows = (binary_subset == 0).all(dim=1).sum().item()
    one_rows = (binary_subset == 1).all(dim=1).sum().item()

    # Density
    density = binary_subset.float().mean().item()

    # Unique rows
    unique_rows = torch.unique(binary_subset, dim=0).shape[0]

    return {
        'zero_rows': zero_rows,
        'one_rows': one_rows,
        'density': density,
        'unique_rows': unique_rows,
        'total_rows': subset.shape[0]
    }


def sample_groups_exhaustive(
    matrix: torch.Tensor,
    group_size: int = 8
) -> List[Dict]:
    """
    Sample matrix columns into groups without replacement using Hamming distance.

    Args:
        matrix: Weight matrix [D, N] where N is number of features
        group_size: Size of each group (default: 8)

    Returns:
        List of group dictionaries containing indices and metrics
    """
    n_features = matrix.shape[1]
    n_complete_groups = n_features // group_size

    excluded_indices = set()
    groups = []

    for group_idx in range(n_complete_groups):
        # Select random seed from remaining features
        available_features = [i for i in range(n_features) if i not in excluded_indices]

        if len(available_features) < group_size:
            break

        seed_idx = random.choice(available_features)

        # Find nearest neighbors
        indices, distances = find_nearest_neighbors_excluding(
            matrix, seed_idx, group_size - 1, excluded_indices
        )

        # Mark as used
        for idx in indices.tolist():
            excluded_indices.add(idx)

        mean_dist = distances.mean().item()

        # Compute detailed metrics for this group
        metrics = compute_group_metrics(matrix, indices.tolist())

        groups.append({
            'group_idx': group_idx,
            'seed_idx': seed_idx,
            'indices': indices.tolist(),
            'mean_distance': mean_dist,
            'size': len(indices),
            'metrics': metrics
        })

    return groups


def create_permutation_matrix(
    groups: List[Dict],
    n_features: int
) -> Tuple[torch.Tensor, List[int]]:
    """
    Create permutation mapping from sampling groups.

    Args:
        groups: List of group dictionaries
        n_features: Total number of features

    Returns:
        permutation: Permutation tensor mapping old->new indices [N]
        permuted_indices: List of column indices in new order
    """
    permuted_indices = []
    for group in groups:
        permuted_indices.extend(group['indices'])

    # Add any remaining columns that weren't grouped
    used_indices = set(permuted_indices)
    remaining = [i for i in range(n_features) if i not in used_indices]
    permuted_indices.extend(remaining)

    # Create inverse mapping (new position -> old index)
    permutation = torch.tensor(permuted_indices, dtype=torch.long)

    return permutation, permuted_indices


def visualize_permuted_matrix(
    matrix: torch.Tensor,
    permutation: torch.Tensor,
    groups: List[Dict],
    layer_idx: int,
    layer_name: str,
    output_path: Path,
    max_display_cols: int = 512
):
    """
    Visualize the permuted binary matrix with group boundaries.

    Args:
        matrix: Original weight matrix [D, N]
        permutation: Permutation indices [N]
        groups: List of group dictionaries
        layer_idx: Layer index
        layer_name: Layer name
        output_path: Path to save visualization
        max_display_cols: Maximum columns to display (default: 512 for 64 groups)
    """
    # Permute columns
    permuted_matrix = matrix[:, permutation]
    binary_matrix = (permuted_matrix != 0).int().cpu().numpy()

    # Limit display size
    n_cols = min(binary_matrix.shape[1], max_display_cols)
    display_matrix = binary_matrix[:, :n_cols]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot binary heatmap
    im = ax.imshow(display_matrix, cmap='binary', aspect='auto', interpolation='nearest')

    # Add group boundaries (every 8 columns)
    group_size = 8
    for i in range(1, n_cols // group_size):
        ax.axvline(x=i * group_size - 0.5, color='red', linewidth=0.5, alpha=0.6)

    # Labels and title
    ax.set_xlabel('Column Index (Permuted)', fontsize=12)
    ax.set_ylabel('Row Index', fontsize=12)
    ax.set_title(
        f'Layer {layer_idx} - {layer_name}\n'
        f'Permuted Matrix (showing {n_cols}/{binary_matrix.shape[1]} columns)\n'
        f'Groups: {len(groups)}, Group size: {group_size}',
        fontsize=13,
        fontweight='bold'
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Weight Active (0=pruned, 1=active)', fontsize=10)

    # Add legend for group boundaries
    red_patch = mpatches.Patch(color='red', alpha=0.6, label='Group boundary')
    ax.legend(handles=[red_patch], loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_group_metrics(
    groups: List[Dict],
    layer_idx: int,
    layer_name: str,
    output_path: Path
):
    """Create comprehensive visualization of group metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Layer {layer_idx} - {layer_name}\nGroup-wise Metrics',
                 fontsize=16, fontweight='bold')

    group_indices = [g['group_idx'] for g in groups]
    mean_distances = [g['mean_distance'] for g in groups]
    zero_rows = [g['metrics']['zero_rows'] for g in groups]
    one_rows = [g['metrics']['one_rows'] for g in groups]
    densities = [g['metrics']['density'] for g in groups]
    unique_rows = [g['metrics']['unique_rows'] for g in groups]

    # Plot 1: Mean Hamming Distance
    axes[0, 0].plot(group_indices, mean_distances, linewidth=1.5, alpha=0.7)
    axes[0, 0].set_xlabel('Group Index', fontsize=11)
    axes[0, 0].set_ylabel('Mean Hamming Distance', fontsize=11)
    axes[0, 0].set_title('Mean Hamming Distance per Group', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=sum(mean_distances)/len(mean_distances),
                       color='r', linestyle='--', alpha=0.5, label='Average')
    axes[0, 0].legend()

    # Plot 2: Zero Rows
    axes[0, 1].bar(group_indices, zero_rows, alpha=0.7, color='steelblue', width=1.0)
    axes[0, 1].set_xlabel('Group Index', fontsize=11)
    axes[0, 1].set_ylabel('Number of Zero Rows', fontsize=11)
    axes[0, 1].set_title('All-Zero Rows per Group', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: One Rows
    axes[0, 2].bar(group_indices, one_rows, alpha=0.7, color='coral', width=1.0)
    axes[0, 2].set_xlabel('Group Index', fontsize=11)
    axes[0, 2].set_ylabel('Number of One Rows', fontsize=11)
    axes[0, 2].set_title('All-One Rows per Group', fontsize=12, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # Plot 4: Density
    axes[1, 0].plot(group_indices, densities, linewidth=1.5, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Group Index', fontsize=11)
    axes[1, 0].set_ylabel('Density', fontsize=11)
    axes[1, 0].set_title('Density per Group', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=sum(densities)/len(densities),
                       color='r', linestyle='--', alpha=0.5, label='Average')
    axes[1, 0].legend()

    # Plot 5: Unique Rows
    axes[1, 1].plot(group_indices, unique_rows, linewidth=1.5, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Group Index', fontsize=11)
    axes[1, 1].set_ylabel('Unique Rows', fontsize=11)
    axes[1, 1].set_title('Unique Rows per Group', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Summary Statistics Table
    axes[1, 2].axis('off')
    summary_stats = [
        ['Metric', 'Mean', 'Min', 'Max'],
        ['Hamming Dist', f'{sum(mean_distances)/len(mean_distances):.4f}',
         f'{min(mean_distances):.4f}', f'{max(mean_distances):.4f}'],
        ['Zero Rows', f'{sum(zero_rows)/len(zero_rows):.1f}',
         f'{min(zero_rows)}', f'{max(zero_rows)}'],
        ['One Rows', f'{sum(one_rows)/len(one_rows):.1f}',
         f'{min(one_rows)}', f'{max(one_rows)}'],
        ['Density', f'{sum(densities)/len(densities):.4f}',
         f'{min(densities):.4f}', f'{max(densities):.4f}'],
        ['Unique Rows', f'{sum(unique_rows)/len(unique_rows):.1f}',
         f'{min(unique_rows)}', f'{max(unique_rows)}']
    ]
    table = axes[1, 2].table(cellText=summary_stats, cellLoc='center', loc='center',
                             colWidths=[0.35, 0.25, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    axes[1, 2].set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics(
    groups: List[Dict],
    layer_idx: int,
    layer_name: str,
    matrix_shape: Tuple[int, int],
    output_path: Path
):
    """Save group metrics to text file."""
    with open(output_path, 'w') as f:
        f.write(f"Layer {layer_idx}: {layer_name}\n")
        f.write(f"Matrix shape: {matrix_shape}\n")
        f.write(f"Total groups: {len(groups)}\n")
        f.write(f"="*60 + "\n\n")

        # Summary statistics
        mean_distances = [g['mean_distance'] for g in groups]
        zero_rows = [g['metrics']['zero_rows'] for g in groups]
        one_rows = [g['metrics']['one_rows'] for g in groups]
        densities = [g['metrics']['density'] for g in groups]
        unique_rows = [g['metrics']['unique_rows'] for g in groups]

        f.write("SUMMARY STATISTICS:\n")
        f.write(f"  Mean Hamming Distance: avg={sum(mean_distances)/len(mean_distances):.4f}, "
                f"min={min(mean_distances):.4f}, max={max(mean_distances):.4f}\n")
        f.write(f"  Zero Rows: avg={sum(zero_rows)/len(zero_rows):.2f}, "
                f"min={min(zero_rows)}, max={max(zero_rows)}\n")
        f.write(f"  One Rows: avg={sum(one_rows)/len(one_rows):.2f}, "
                f"min={min(one_rows)}, max={max(one_rows)}\n")
        f.write(f"  Density: avg={sum(densities)/len(densities):.4f}, "
                f"min={min(densities):.4f}, max={max(densities):.4f}\n")
        f.write(f"  Unique Rows: avg={sum(unique_rows)/len(unique_rows):.2f}, "
                f"min={min(unique_rows)}, max={max(unique_rows)}\n")
        f.write(f"\n" + "="*60 + "\n\n")

        # Group details (first 10 and last 5)
        f.write("GROUP DETAILS (first 10):\n")
        for group in groups[:10]:
            m = group['metrics']
            f.write(
                f"  Group {group['group_idx']:3d}: "
                f"seed={group['seed_idx']:4d}, size={group['size']}, "
                f"mean_dist={group['mean_distance']:.4f}, "
                f"zeros={m['zero_rows']}, ones={m['one_rows']}, "
                f"density={m['density']:.3f}, unique={m['unique_rows']}\n"
            )

        if len(groups) > 15:
            f.write(f"  ... ({len(groups) - 15} groups omitted) ...\n")

        if len(groups) > 10:
            f.write("\nGROUP DETAILS (last 5):\n")
            for group in groups[-5:]:
                m = group['metrics']
                f.write(
                    f"  Group {group['group_idx']:3d}: "
                    f"seed={group['seed_idx']:4d}, size={group['size']}, "
                    f"mean_dist={group['mean_distance']:.4f}, "
                    f"zeros={m['zero_rows']}, ones={m['one_rows']}, "
                    f"density={m['density']:.3f}, unique={m['unique_rows']}\n"
                )


def permute_and_visualize(
    weight_matrix: torch.Tensor,
    layer_idx: int,
    layer_name: str,
    output_dirs: Dict[str, str],
    group_size: int = 8,
    seed: int = 42
) -> Dict:
    """
    Main entry point: perform column permutation analysis and visualization.

    Args:
        weight_matrix: Pruned weight matrix [D, N]
        layer_idx: Layer index in the model
        layer_name: Name of the layer (e.g., 'self_attn.q_proj')
        output_dirs: Dictionary with 'metrics' and 'images' output directories
        group_size: Size of each group (default: 8)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - permutation: Permutation tensor [N]
            - groups: List of group dictionaries
            - metrics_file: Path to saved metrics file
            - image_file: Path to saved visualization
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Create output directories
    metrics_dir = Path(output_dirs['metrics'])
    images_dir = Path(output_dirs['images'])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Sample groups
    groups = sample_groups_exhaustive(weight_matrix, group_size=group_size)

    # Create permutation
    permutation, _ = create_permutation_matrix(groups, weight_matrix.shape[1])

    # Generate sanitized layer name for filenames
    sanitized_name = layer_name.replace('.', '_')

    # Save metrics
    metrics_file = metrics_dir / f"layer_{layer_idx:02d}_{sanitized_name}_metrics.txt"
    save_metrics(groups, layer_idx, layer_name, weight_matrix.shape, metrics_file)

    # Visualize permuted matrix
    image_file = images_dir / f"layer_{layer_idx:02d}_{sanitized_name}_permuted.png"
    visualize_permuted_matrix(
        weight_matrix, permutation, groups, layer_idx, layer_name, image_file
    )

    # Visualize group metrics
    metrics_viz_file = images_dir / f"layer_{layer_idx:02d}_{sanitized_name}_group_metrics.png"
    visualize_group_metrics(groups, layer_idx, layer_name, metrics_viz_file)

    return {
        'permutation': permutation,
        'groups': groups,
        'metrics_file': str(metrics_file),
        'image_file': str(image_file),
        'metrics_viz_file': str(metrics_viz_file),
        'n_groups': len(groups),
        'avg_mean_distance': sum(g['mean_distance'] for g in groups) / len(groups) if groups else 0
    }
