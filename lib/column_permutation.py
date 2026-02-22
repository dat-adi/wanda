"""
Column permutation module for Wanda pruning system.

This module performs exhaustive sampling without replacement to group and reorder
matrix columns by Hamming distance similarity. It creates groups of 8 columns where
each group contains features that are spatially close in Hamming space, enabling
efficient kernel processing with banded workload patterns.

Groups are sorted by their mean Hamming distance (lowest to highest), creating a
gradient from tightest clusters to loosest clusters in the permuted matrix.

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
import numpy as np
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


def compute_pairwise_hamming_distances(subset: torch.Tensor) -> torch.Tensor:
    """
    Compute all pairwise Hamming distances for a subset of columns.

    Args:
        subset: Weight matrix subset [D, N] where N is number of columns in group

    Returns:
        Distance matrix [N, N] with pairwise Hamming distances
    """
    N = subset.shape[1]
    D = subset.shape[0]

    # Convert to binary
    binary_subset = (subset != 0).int()
    binary_float = binary_subset.float()

    # Compute dot products (counts matching 1s)
    dot_products = binary_float.T @ binary_float  # [N, N]

    # Count of 1s in each column
    ones_count = binary_subset.sum(dim=0)  # [N]

    # Hamming distance formula: hamming(i, j) = (ones_i + ones_j - 2 * matches) / D
    ones_count_i = ones_count.unsqueeze(1)  # [N, 1]
    ones_count_j = ones_count.unsqueeze(0)  # [1, N]

    distance_matrix = (ones_count_i + ones_count_j - 2 * dot_products).float() / D

    return distance_matrix


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
        indices, _ = find_nearest_neighbors_excluding(
            matrix, seed_idx, group_size - 1, excluded_indices
        )

        # Mark as used
        for idx in indices.tolist():
            excluded_indices.add(idx)

        # Extract the subset of columns for this group
        subset = matrix[:, indices]

        # Compute pairwise Hamming distances for all pairs in the group
        pairwise_dist = compute_pairwise_hamming_distances(subset)
        mean_dist = pairwise_dist.mean().item()

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
) -> Tuple[torch.Tensor, List[int], List[Dict]]:
    """
    Create permutation mapping from sampling groups, sorted by mean Hamming distance.

    Args:
        groups: List of group dictionaries
        n_features: Total number of features

    Returns:
        permutation: Permutation tensor mapping old->new indices [N]
        permuted_indices: List of column indices in new order
        sorted_groups: Groups sorted by mean Hamming distance (low to high)
    """
    # Sort groups by mean Hamming distance (lowest to highest)
    sorted_groups = sorted(groups, key=lambda g: g['mean_distance'])

    # Build permuted indices from sorted groups
    permuted_indices = []
    for group in sorted_groups:
        permuted_indices.extend(group['indices'])

    # Add any remaining columns that weren't grouped
    used_indices = set(permuted_indices)
    remaining = [i for i in range(n_features) if i not in used_indices]
    permuted_indices.extend(remaining)

    # Create inverse mapping (new position -> old index)
    permutation = torch.tensor(permuted_indices, dtype=torch.long)

    return permutation, permuted_indices, sorted_groups


def apply_normal_transform(binary_matrix: np.ndarray) -> np.ndarray:
    """Apply normal transformation - no change to binary matrix."""
    return binary_matrix


def apply_line_transform(binary_matrix: np.ndarray, group_size: int = 8) -> np.ndarray:
    """
    Apply line transformation: for each group of 8 columns, if a row has any 1,
    convert all values in that row for the entire group to 1.

    Args:
        binary_matrix: Binary matrix [D, N]
        group_size: Size of each group (default: 8)

    Returns:
        Transformed matrix [D, N]
    """
    display_matrix = binary_matrix.copy()
    n_groups = binary_matrix.shape[1] // group_size

    for group_idx in range(n_groups):
        start_col = group_idx * group_size
        end_col = start_col + group_size
        group = display_matrix[:, start_col:end_col]

        # For each row, if there's any 1 in the group, set all values to 1
        has_activation = (group.sum(axis=1) > 0).reshape(-1, 1)
        display_matrix[:, start_col:end_col] = has_activation * np.ones((1, group_size), dtype=int)

    return display_matrix


def apply_compress_transform(binary_matrix: np.ndarray, group_size: int = 8) -> np.ndarray:
    """
    Apply compress transformation: apply line transform then compress each group to a single column.

    Args:
        binary_matrix: Binary matrix [D, N]
        group_size: Size of each group (default: 8)

    Returns:
        Compressed matrix [D, N_groups]
    """
    # First apply line transformation
    temp_matrix = apply_line_transform(binary_matrix, group_size)
    n_groups = binary_matrix.shape[1] // group_size

    # Compress each group to a single column
    compressed_cols = []
    for group_idx in range(n_groups):
        start_col = group_idx * group_size
        group = temp_matrix[:, start_col:start_col + group_size]
        # Take any column (they're all the same after line transformation)
        compressed_cols.append(group[:, 0])

    return np.column_stack(compressed_cols)


def apply_line_transform_rows(binary_matrix: np.ndarray, group_size: int = 8) -> np.ndarray:
    """
    Apply line transformation along row groups: for each group of group_size rows,
    if any column has a 1 within the group, set all rows in that group for that column to 1.

    Args:
        binary_matrix: Binary matrix [D, N]
        group_size: Size of each row group (default: 8)

    Returns:
        Transformed matrix [D, N]
    """
    display_matrix = binary_matrix.copy()
    n_groups = binary_matrix.shape[0] // group_size

    for group_idx in range(n_groups):
        start_row = group_idx * group_size
        end_row = start_row + group_size
        group = display_matrix[start_row:end_row, :]
        # For each column, if there's any 1 in the row-group, set all rows in the group to 1
        has_activation = (group.sum(axis=0) > 0).reshape(1, -1)
        display_matrix[start_row:end_row, :] = np.ones((group_size, 1), dtype=int) * has_activation

    return display_matrix


def apply_compress_transform_rows(binary_matrix: np.ndarray, group_size: int = 8) -> np.ndarray:
    """
    Apply compress transformation along row groups: apply row line transform then
    compress each row group to a single representative row.

    Args:
        binary_matrix: Binary matrix [D, N]
        group_size: Size of each row group (default: 8)

    Returns:
        Compressed matrix [D_groups, N]
    """
    temp_matrix = apply_line_transform_rows(binary_matrix, group_size)
    n_groups = binary_matrix.shape[0] // group_size

    compressed_rows = []
    for group_idx in range(n_groups):
        start_row = group_idx * group_size
        group = temp_matrix[start_row:start_row + group_size, :]
        # Take any row (they're all the same after line transformation)
        compressed_rows.append(group[0, :])

    return np.row_stack(compressed_rows)


def visualize_permuted_matrix(
    matrix: torch.Tensor,
    permutation: torch.Tensor,
    groups: List[Dict],
    layer_idx: int,
    layer_name: str,
    output_path: Path,
    mode: str = 'normal',
    axis: str = 'columns',
    group_size: int = 8,
):
    """
    Visualize the permuted binary matrix with group boundaries.

    Args:
        matrix: Original weight matrix [D, N]
        permutation: Permutation indices along the selected axis
        groups: List of group dictionaries
        layer_idx: Layer index
        layer_name: Layer name
        output_path: Path to save visualization
        mode: Visualization mode - 'normal', 'line', or 'compress'
        axis: Permutation axis - 'columns' (horizontal) or 'rows' (vertical)
        group_size: Number of features per group
    """
    # Apply permutation to the correct axis
    if axis == 'rows':
        permuted_matrix = matrix[permutation, :]
    else:
        permuted_matrix = matrix[:, permutation]
    binary_matrix = (permuted_matrix != 0).int().cpu().numpy()

    # Apply transformation based on mode and axis
    if axis == 'rows':
        if mode == 'normal':
            display_matrix = apply_normal_transform(binary_matrix)
            n_dim = binary_matrix.shape[0]
        elif mode == 'line':
            display_matrix = apply_line_transform_rows(binary_matrix, group_size)
            n_dim = binary_matrix.shape[0]
        elif mode == 'compress':
            display_matrix = apply_compress_transform_rows(binary_matrix, group_size)
            n_dim = display_matrix.shape[0]
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'normal', 'line', or 'compress'.")
    else:
        if mode == 'normal':
            display_matrix = apply_normal_transform(binary_matrix)
            n_dim = binary_matrix.shape[1]
        elif mode == 'line':
            display_matrix = apply_line_transform(binary_matrix, group_size)
            n_dim = binary_matrix.shape[1]
        elif mode == 'compress':
            display_matrix = apply_compress_transform(binary_matrix, group_size)
            n_dim = display_matrix.shape[1]
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'normal', 'line', or 'compress'.")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot binary heatmap
    im = ax.imshow(display_matrix, cmap='binary', aspect='auto', interpolation='nearest')

    # Compute statistics for the display matrix
    total_bits = display_matrix.size
    num_zeros = np.sum(display_matrix == 0)
    num_ones = np.sum(display_matrix == 1)
    percent_zeros = (num_zeros / total_bits) * 100 if total_bits > 0 else 0

    # Labels — swap axes based on permutation direction
    if axis == 'rows':
        ax.set_xlabel('Column Index', fontsize=12)
        ax.set_ylabel('Row Index (Permuted - Sorted by Hamming Distance)', fontsize=12)
        axis_unit = 'row groups' if mode == 'compress' else 'rows'
    else:
        ax.set_xlabel('Column Index (Permuted - Sorted by Hamming Distance)', fontsize=12)
        ax.set_ylabel('Row Index', fontsize=12)
        axis_unit = 'groups' if mode == 'compress' else 'columns'

    mode_desc = {
        'normal': 'Standard',
        'line': 'Line-filled (per group)',
        'compress': 'Compressed (1 per group)'
    }

    title_text = (
        f'Layer {layer_idx} - {layer_name}\n'
        f'Permuted Matrix - {mode_desc[mode]} ({n_dim} {axis_unit}, axis={axis})\n'
        f'Groups: {len(groups)}, Group size: {group_size} | Sorted: Low → High Hamming Distance\n'
        f'Zeros: {num_zeros:,} | Total bits: {total_bits:,} | Zero %: {percent_zeros:.2f}%'
    )

    ax.set_title(title_text, fontsize=13, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Weight Active (0=pruned, 1=active)', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_group_metrics(
    groups: List[Dict],
    layer_idx: int,
    layer_name: str,
    output_path: Path
):
    """Create comprehensive visualization matching the original style."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Layer {layer_idx} - {layer_name}\nGroup Metrics Analysis (Sorted by Mean Distance)',
                 fontsize=16, fontweight='bold')

    # Extract data (groups are already sorted by mean distance)
    num_groups = len(groups)
    group_numbers = [i + 1 for i in range(num_groups)]  # 1-indexed
    mean_distances = [g['mean_distance'] for g in groups]
    zero_rows = [g['metrics']['zero_rows'] for g in groups]
    one_rows = [g['metrics']['one_rows'] for g in groups]
    densities = [g['metrics']['density'] for g in groups]
    unique_rows = [g['metrics']['unique_rows'] for g in groups]

    # Smart tick configuration based on number of groups
    if num_groups <= 20:
        tick_step = 1
        rotation = 0
        fontsize = 9
    elif num_groups <= 50:
        tick_step = 5
        rotation = 45
        fontsize = 8
    elif num_groups <= 100:
        tick_step = 10
        rotation = 90
        fontsize = 7
    else:
        tick_step = 20
        rotation = 90
        fontsize = 6

    tick_positions = group_numbers[::tick_step]
    tick_labels = [str(x) for x in tick_positions]

    def configure_xaxis(ax, group_numbers):
        ax.set_xlim(0.5, len(group_numbers) + 0.5)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=rotation, fontsize=fontsize,
                          ha='right' if rotation > 0 else 'center')

    # Plot 1: Mean Hamming Distance (bar chart, not line)
    ax1 = axes[0, 0]
    ax1.bar(group_numbers, mean_distances, color='steelblue', alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Group Rank (by Mean Distance)', fontsize=11)
    ax1.set_ylabel('Mean Hamming Distance', fontsize=11)
    ax1.set_title('Mean Pairwise Hamming Distance by Group', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    configure_xaxis(ax1, group_numbers)

    # Plot 2: All-Zero Rows
    ax2 = axes[0, 1]
    ax2.bar(group_numbers, zero_rows, color='coral', alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Group Rank (by Mean Distance)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('All-Zero Rows by Group', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    configure_xaxis(ax2, group_numbers)

    # Plot 3: All-One Rows
    ax3 = axes[0, 2]
    ax3.bar(group_numbers, one_rows, color='mediumseagreen', alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Group Rank (by Mean Distance)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('All-One Rows by Group', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    configure_xaxis(ax3, group_numbers)

    # Plot 4: Density
    ax4 = axes[1, 0]
    ax4.bar(group_numbers, densities, color='mediumpurple', alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Group Rank (by Mean Distance)', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('Density by Group', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    configure_xaxis(ax4, group_numbers)

    # Plot 5: Unique Rows
    ax5 = axes[1, 1]
    ax5.bar(group_numbers, unique_rows, color='gold', alpha=0.7,
            edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('Group Rank (by Mean Distance)', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Unique Rows by Group', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    configure_xaxis(ax5, group_numbers)

    # Plot 6: Summary Statistics (text panel)
    axes[1, 2].axis('off')

    total_zeros = sum(zero_rows)
    total_ones = sum(one_rows)
    avg_density = sum(densities) / len(densities)
    avg_unique = sum(unique_rows) / len(unique_rows)

    summary_text = f"Summary Statistics:\n\n"
    summary_text += f"Total Groups: {num_groups}\n"
    summary_text += f"Group Size: 8\n\n"
    summary_text += f"Avg Mean Distance: {sum(mean_distances)/len(mean_distances):.4f}\n"
    summary_text += f"Min Mean Distance: {min(mean_distances):.4f}\n"
    summary_text += f"Max Mean Distance: {max(mean_distances):.4f}\n\n"
    summary_text += f"Total Zero Rows: {total_zeros}\n"
    summary_text += f"Total One Rows: {total_ones}\n"
    summary_text += f"Avg Density: {avg_density:.4f}\n"
    summary_text += f"Avg Unique Rows: {avg_unique:.1f}"

    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11,
                   verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    axis: str = 'columns',
    seed: int = 42,
    viz_normal: bool = True,
    viz_line: bool = False,
    viz_compress: bool = True,
    save_permuted: bool = False,
    save_line: bool = False
) -> Dict:
    """
    Main entry point: perform exhaustive Hamming-distance permutation analysis and visualization.

    Groups are formed by exhaustive sampling without replacement, each containing
    group_size features (columns or rows) that are close in Hamming space.
    Groups are then sorted by their mean pairwise Hamming distance (lowest first),
    so the permuted matrix runs from tightest to loosest clusters.

    Args:
        weight_matrix: Pruned weight matrix [D, N]
        layer_idx: Layer index in the model
        layer_name: Name of the layer (e.g., 'self_attn.q_proj')
        output_dirs: Dictionary with 'metrics', 'images', and 'workloads' output directories
        group_size: Number of features per group (default: 8)
        axis: Axis to permute - 'columns' (horizontal) or 'rows' (vertical) (default: 'columns')
        seed: Random seed for reproducibility
        viz_normal: Generate normal/standard visualization (default: True)
        viz_line: Generate line-filled visualization (default: False)
        viz_compress: Generate compressed visualization (default: True)
        save_permuted: Save permuted matrix + permutation order to workloads directory (default: False)
        save_line: Save line-transformed matrix to workloads directory (default: False)

    Returns:
        Dictionary containing:
            - permutation: Permutation tensor along selected axis
            - groups: List of group dictionaries
            - sorted_groups: Groups sorted by mean Hamming distance
            - metrics_file: Path to saved metrics file
            - image_files: Dictionary of generated image file paths by mode
            - permuted_file: Path to saved permuted matrix (or None)
            - perm_order_file: Path to saved permutation order tensor (or None)
            - line_file: Path to saved line-transformed matrix (or None)
    """
    random.seed(seed)
    torch.manual_seed(seed)

    # Create output directories
    metrics_dir = Path(output_dirs['metrics'])
    images_dir = Path(output_dirs['images'])
    workloads_dir = Path(output_dirs['workloads'])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    workloads_dir.mkdir(parents=True, exist_ok=True)

    # For row permutation, transpose so the column grouping logic applies to rows
    working_matrix = weight_matrix.T if axis == 'rows' else weight_matrix

    # Sample groups using exhaustive Hamming-distance-based grouping
    groups = sample_groups_exhaustive(working_matrix, group_size=group_size)

    # Create permutation — sorts groups by mean Hamming distance (low → high)
    permutation, _, sorted_groups = create_permutation_matrix(groups, working_matrix.shape[1])

    # Generate sanitized layer name for filenames
    sanitized_name = layer_name.replace('.', '_')

    # Save permuted matrix and permutation order to workloads directory if enabled
    permuted_file = None
    perm_order_file = None
    if save_permuted:
        if axis == 'rows':
            permuted_matrix = weight_matrix[permutation, :]
        else:
            permuted_matrix = weight_matrix[:, permutation]
        permuted_file = workloads_dir / f"layer_{layer_idx:02d}_{sanitized_name}_{axis}_permuted.pt"
        torch.save(permuted_matrix, permuted_file)
        # Save the permutation order separately so it can be re-applied to non-binary weights
        perm_order_file = workloads_dir / f"layer_{layer_idx:02d}_{sanitized_name}_{axis}_perm_order.pt"
        torch.save(permutation, perm_order_file)

    # Save line-transformed matrix to workloads directory if enabled
    line_file = None
    if save_line:
        if axis == 'rows':
            permuted_matrix = weight_matrix[permutation, :]
            binary_matrix = (permuted_matrix != 0).int().cpu().numpy()
            line_matrix = apply_line_transform_rows(binary_matrix, group_size)
        else:
            permuted_matrix = weight_matrix[:, permutation]
            binary_matrix = (permuted_matrix != 0).int().cpu().numpy()
            line_matrix = apply_line_transform(binary_matrix, group_size)
        line_file = workloads_dir / f"layer_{layer_idx:02d}_{sanitized_name}_{axis}_line.pt"
        torch.save(torch.from_numpy(line_matrix), line_file)

    # Save metrics (using sorted groups for better readability)
    metrics_file = metrics_dir / f"layer_{layer_idx:02d}_{sanitized_name}_metrics.txt"
    save_metrics(sorted_groups, layer_idx, layer_name, weight_matrix.shape, metrics_file)

    # Generate visualizations based on flags
    image_files = {}

    if viz_normal:
        image_file = images_dir / f"layer_{layer_idx:02d}_{sanitized_name}_{axis}_permuted_normal.png"
        visualize_permuted_matrix(
            weight_matrix, permutation, sorted_groups, layer_idx, layer_name,
            image_file, mode='normal', axis=axis, group_size=group_size
        )
        image_files['normal'] = str(image_file)

    if viz_line:
        image_file = images_dir / f"layer_{layer_idx:02d}_{sanitized_name}_{axis}_permuted_line.png"
        visualize_permuted_matrix(
            weight_matrix, permutation, sorted_groups, layer_idx, layer_name,
            image_file, mode='line', axis=axis, group_size=group_size
        )
        image_files['line'] = str(image_file)

    if viz_compress:
        image_file = images_dir / f"layer_{layer_idx:02d}_{sanitized_name}_{axis}_permuted_compress.png"
        visualize_permuted_matrix(
            weight_matrix, permutation, sorted_groups, layer_idx, layer_name,
            image_file, mode='compress', axis=axis, group_size=group_size
        )
        image_files['compress'] = str(image_file)

    # Visualize group metrics (using sorted groups)
    metrics_viz_file = images_dir / f"layer_{layer_idx:02d}_{sanitized_name}_group_metrics.png"
    visualize_group_metrics(sorted_groups, layer_idx, layer_name, metrics_viz_file)

    return {
        'permutation': permutation,
        'groups': groups,
        'sorted_groups': sorted_groups,
        'metrics_file': str(metrics_file),
        'image_files': image_files,
        'metrics_viz_file': str(metrics_viz_file),
        'permuted_file': str(permuted_file) if permuted_file else None,
        'perm_order_file': str(perm_order_file) if perm_order_file else None,
        'line_file': str(line_file) if line_file else None,
        'n_groups': len(groups),
        'avg_mean_distance': sum(g['mean_distance'] for g in groups) / len(groups) if groups else 0
    }
