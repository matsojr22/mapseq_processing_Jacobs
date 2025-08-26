import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, kruskal
import ast
from itertools import combinations
import os
import glob

# Set font to Helvetica
plt.rcParams["font.family"] = "Helvetica"

# Base directory containing per-animal data
BASE_DIR = "/Volumes/euiseokdataUCSC_3/Matt Jacobs/mapseq_analysis_adam/motif_analysis/motif_observed_summary"

# Results directory for saving outputs
RESULTS_DIR = "/Volumes/euiseokdataUCSC_3/Matt Jacobs/mapseq_analysis_adam/motif_analysis/motif_observed_summary/results"

# To use a different directory, uncomment and modify the line below:
# set_base_directory("/path/to/your/motif_observed_summary")


def ensure_results_directory():
    """Create results directory if it doesn't exist"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created results directory: {RESULTS_DIR}")
    else:
        print(f"Using existing results directory: {RESULTS_DIR}")


def perform_kruskal_wallis_tests(datasets, all_motifs, normalization_type="global"):
    """
    Perform Kruskal-Wallis tests for each motif across timepoints.

    Args:
        datasets: dict of {timepoint: dataframe}
        all_motifs: list of all unique motifs
        normalization_type: 'global' or 'domain'

    Returns:
        pandas.DataFrame with test results
    """
    from scipy.stats import kruskal

    results = []
    datasets_list = list(datasets.keys())

    for motif in all_motifs:
        # Collect data for this motif from all timepoints
        groups = []
        group_labels = []

        for timepoint, df in datasets.items():
            motif_data = df[df["motif label_Clean"] == motif]

            if len(motif_data) > 0:
                if normalization_type == "global":
                    values = motif_data["normalized_freq"].values
                else:
                    # For domain-wise, we'll use domain_normalized_freq if available
                    if "domain_normalized_freq" in motif_data.columns:
                        values = motif_data["domain_normalized_freq"].values
                    else:
                        values = motif_data["normalized_freq"].values

                if len(values) > 0:
                    groups.append(values)
                    group_labels.append(timepoint)

        # Perform Kruskal-Wallis test if we have data from at least 2 groups
        if len(groups) >= 2:
            try:
                # Check if all groups have at least some variation
                valid_groups = [
                    group for group in groups if len(group) > 0 and np.var(group) > 0
                ]

                if len(valid_groups) >= 2:
                    h_stat, p_value = kruskal(*valid_groups)

                    # Get sample sizes for each group
                    sample_sizes = {
                        label: len(group) for label, group in zip(group_labels, groups)
                    }

                    results.append(
                        {
                            "Motif": motif,
                            "H_statistic": h_stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05,
                            "n_groups": len(valid_groups),
                            **sample_sizes,
                            "normalization": normalization_type,
                        }
                    )
                else:
                    results.append(
                        {
                            "Motif": motif,
                            "H_statistic": np.nan,
                            "p_value": np.nan,
                            "significant": False,
                            "n_groups": len(groups),
                            "note": "Insufficient variation in groups",
                            "normalization": normalization_type,
                        }
                    )
            except Exception as e:
                results.append(
                    {
                        "Motif": motif,
                        "H_statistic": np.nan,
                        "p_value": np.nan,
                        "significant": False,
                        "n_groups": len(groups),
                        "note": f"Error: {str(e)}",
                        "normalization": normalization_type,
                    }
                )
        else:
            results.append(
                {
                    "Motif": motif,
                    "H_statistic": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                    "n_groups": len(groups),
                    "note": "Insufficient groups for testing",
                    "normalization": normalization_type,
                }
            )

    return pd.DataFrame(results)


def set_base_directory(new_base_dir):
    """
    Set a new base directory for loading per-animal data.

    Args:
        new_base_dir (str): Path to directory containing timepoint subdirectories
    """
    global BASE_DIR
    BASE_DIR = new_base_dir
    print(f"Base directory set to: {BASE_DIR}")


def load_per_animal_data(base_dir):
    """
    Load per-animal motif data from directory structure, preserving individual animal measurements.

    Expected structure:
    base_dir/
    ├── p12/
    │   ├── animal1_motif_observed_summary.csv
    │   ├── animal2_motif_observed_summary.csv
    │   └── ...
    ├── p20/
    │   └── ...
    └── p60/
        └── ...

    Returns:
        dict: {timepoint: dataframe with individual animal data}
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    datasets = {}

    # Look for timepoint directories (p12, p20, p60)
    timepoint_dirs = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.lower().startswith("p")
    ]

    if not timepoint_dirs:
        raise ValueError(
            f"No timepoint directories (p12, p20, p60) found in {base_dir}"
        )

    for timepoint_dir in sorted(timepoint_dirs):
        timepoint_path = os.path.join(base_dir, timepoint_dir)
        timepoint_name = timepoint_dir.upper()  # Convert to P12, P20, P60

        print(f"Loading data for {timepoint_name}...")

        # Find all CSV files in this timepoint directory
        csv_files = glob.glob(os.path.join(timepoint_path, "*.csv"))

        if not csv_files:
            print(f"  Warning: No CSV files found in {timepoint_path}")
            continue

        print(f"  Found {len(csv_files)} files")

        # Load data from all animals for this timepoint (keeping individual animal data)
        all_animal_data = []

        for csv_file in csv_files:
            animal_id = os.path.basename(csv_file).split("_")[0]  # Extract animal ID
            try:
                df = pd.read_csv(csv_file)

                # Validate expected columns
                required_columns = ["motif label", "motif size", "observed"]
                missing_columns = [
                    col for col in required_columns if col not in df.columns
                ]
                if missing_columns:
                    print(
                        f"    Warning: {animal_id} missing columns: {missing_columns}"
                    )
                    continue

                # Calculate normalized frequency for this animal
                total_observations = df["observed"].sum()
                df["normalized_freq"] = df["observed"] / total_observations
                df["Animal_ID"] = animal_id  # Add animal identifier
                df["Timepoint"] = timepoint_name  # Add timepoint identifier

                all_animal_data.append(df)
                print(
                    f"    Loaded {animal_id}: {len(df)} motifs, {df['observed'].sum()} observations"
                )
            except Exception as e:
                print(f"    Error loading {csv_file}: {e}")

        if not all_animal_data:
            print(f"  Warning: No valid data loaded for {timepoint_name}")
            continue

        # Combine all animal data for this timepoint (preserving individual measurements)
        combined_df = pd.concat(all_animal_data, ignore_index=True)

        datasets[timepoint_name] = combined_df
        print(
            f"  Combined data from {len(csv_files)} animals with {len(combined_df)} total motif observations"
        )

    return datasets


def clean_motif_label(label):
    """Clean and standardize motif labels"""
    if isinstance(label, str):
        try:
            # Try to evaluate as a Python literal
            return str(sorted(ast.literal_eval(label)))
        except:
            # If that fails, just return the string
            return label
    return str(label)


def add_bracket_annotation(fig, ax, x_start, x_end, text):
    """Add an upward-opening bracket with text annotation below the x tick labels"""

    # Force a draw to ensure tick labels are rendered
    fig.canvas.draw()

    # Convert x positions to axis coordinates (0-1 range within the axis)
    xlim = ax.get_xlim()
    x_start_ax = (x_start - xlim[0]) / (xlim[1] - xlim[0])
    x_end_ax = (x_end - xlim[0]) / (xlim[1] - xlim[0])

    # Find the bottom of the tick labels by checking their bounding boxes
    tick_bottom = 0  # Start at axis bottom
    for tick in ax.get_xticklabels():
        if tick.get_text():  # Only consider non-empty labels
            bbox = tick.get_window_extent()
            # Convert to axes coordinates
            axes_coords = ax.transAxes.inverted().transform(
                [(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)]
            )
            tick_y_bottom = axes_coords[0][
                1
            ]  # Bottom of tick label in axes coordinates
            tick_bottom = min(tick_bottom, tick_y_bottom)

    # Position brackets below the tick labels
    bracket_spacing = 0.04  # Small gap below tick labels
    bracket_height = 0.015  # Height of bracket
    text_spacing = 0.04  # Gap between bracket and text

    bracket_y = tick_bottom - bracket_spacing
    text_y = bracket_y - bracket_height - text_spacing

    # Draw bracket using plot with axes coordinates and clip_on=False
    # Left vertical line
    ax.plot(
        [x_start_ax, x_start_ax],
        [bracket_y, bracket_y + bracket_height],
        "k-",
        linewidth=1.5,
        transform=ax.transAxes,
        clip_on=False,
    )
    # Right vertical line
    ax.plot(
        [x_end_ax, x_end_ax],
        [bracket_y, bracket_y + bracket_height],
        "k-",
        linewidth=1.5,
        transform=ax.transAxes,
        clip_on=False,
    )
    # Bottom horizontal line
    ax.plot(
        [x_start_ax, x_end_ax],
        [bracket_y, bracket_y],
        "k-",
        linewidth=1.5,
        transform=ax.transAxes,
        clip_on=False,
    )

    # Add text below the bracket
    ax.text(
        (x_start_ax + x_end_ax) / 2,
        text_y,
        text,
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
        clip_on=False,
    )


def calculate_distribution_jsd(freq_arrays):
    """Calculate JSD between three frequency distributions"""
    # Add small epsilon to avoid log(0) and normalize
    epsilon = 1e-10

    freq1 = np.array(freq_arrays[0]) + epsilon
    freq2 = np.array(freq_arrays[1]) + epsilon
    freq3 = np.array(freq_arrays[2]) + epsilon

    freq1 = freq1 / freq1.sum()
    freq2 = freq2 / freq2.sum()
    freq3 = freq3 / freq3.sum()

    # Calculate pairwise JSDs
    jsd_12 = jensenshannon(freq1, freq2)
    jsd_13 = jensenshannon(freq1, freq3)
    jsd_23 = jensenshannon(freq2, freq3)

    return jsd_12, jsd_13, jsd_23


def calculate_summary_stats(datasets, all_motifs, normalization_type="global"):
    """
    Calculate mean and SEM for each motif across animals for each timepoint.

    Args:
        datasets: dict of {timepoint: dataframe}
        all_motifs: list of all unique motifs
        normalization_type: 'global' or 'domain'

    Returns:
        dict: {timepoint: {motif: {'mean': float, 'sem': float, 'n': int, 'values': list}}}
    """
    from scipy import stats

    summary_stats = {}

    for timepoint, df in datasets.items():
        summary_stats[timepoint] = {}

        for motif in all_motifs:
            # Get all animal values for this motif
            motif_data = df[df["motif label_Clean"] == motif]

            if len(motif_data) > 0:
                if normalization_type == "global":
                    values = motif_data["normalized_freq"].values
                else:  # domain-wise normalization will be handled separately
                    values = motif_data["normalized_freq"].values

                if len(values) > 0:
                    mean_val = np.mean(values)
                    sem_val = stats.sem(values) if len(values) > 1 else 0
                    summary_stats[timepoint][motif] = {
                        "mean": mean_val,
                        "sem": sem_val,
                        "n": len(values),
                        "values": values.tolist(),
                    }
                else:
                    summary_stats[timepoint][motif] = {
                        "mean": 0,
                        "sem": 0,
                        "n": 0,
                        "values": [],
                    }
            else:
                summary_stats[timepoint][motif] = {
                    "mean": 0,
                    "sem": 0,
                    "n": 0,
                    "values": [],
                }

    return summary_stats


def plot_bars_with_points_and_errors(
    ax, x_positions, summary_stats, datasets_list, colors, width, alpha=0.8
):
    """
    Plot bars with individual points and error bars.

    Args:
        ax: matplotlib axis
        x_positions: array of x positions
        summary_stats: summary statistics dictionary
        datasets_list: list of dataset names
        colors: list of colors
        width: bar width
        alpha: bar transparency

    Returns:
        list of bar objects
    """
    all_bars = []

    for i, dataset in enumerate(datasets_list):
        means = []
        sems = []

        # Get means and SEMs for all motifs
        for j, x_pos in enumerate(x_positions):
            motif_key = list(summary_stats[dataset].keys())[j]
            stats = summary_stats[dataset][motif_key]
            means.append(stats["mean"])
            sems.append(stats["sem"])

        # Plot bars
        bars = ax.bar(
            x_positions + i * width,
            means,
            width,
            label=dataset,
            color=colors[i],
            alpha=alpha,
            yerr=sems,
            capsize=3,
            error_kw={"elinewidth": 1},
        )
        all_bars.extend(bars)

        # Plot individual points (smaller and fully black)
        for j, x_pos in enumerate(x_positions):
            motif_key = list(summary_stats[dataset].keys())[j]
            values = summary_stats[dataset][motif_key]["values"]

            if values:  # Only plot if there are values
                # Add small random jitter to x position for visibility
                x_jitter = np.random.normal(0, width * 0.1, len(values))
                ax.scatter(
                    x_pos + i * width + x_jitter,
                    values,
                    color="black",
                    alpha=1.0,
                    s=4,
                    zorder=3,
                )

    return all_bars


# Read and process data
print("Loading and aggregating per-animal data...")
datasets = load_per_animal_data(BASE_DIR)

if not datasets:
    raise ValueError(f"No valid datasets found in {BASE_DIR}")

print(f"\nSuccessfully loaded {len(datasets)} timepoints: {list(datasets.keys())}")

# Ensure results directory exists
ensure_results_directory()

# Process datasets - clean motif labels and collect all motifs in original order
# Get motifs in the order they appear in the first dataset
first_dataset = list(datasets.values())[0]
all_motifs_original_order = []
seen_motifs = set()

for _, row in first_dataset.iterrows():
    clean_motif = clean_motif_label(row["motif label"])
    if clean_motif not in seen_motifs:
        all_motifs_original_order.append(clean_motif)
        seen_motifs.add(clean_motif)

# Add any motifs from other datasets that weren't in the first dataset
for name, df in datasets.items():
    df["motif label_Clean"] = df["motif label"].apply(clean_motif_label)
    for motif in df["motif label_Clean"].unique():
        if motif not in seen_motifs:
            all_motifs_original_order.append(motif)
            seen_motifs.add(motif)

all_motifs = all_motifs_original_order  # Use original order instead of sorted

for name, df in datasets.items():
    print(
        f"{name}: {df['Animal_ID'].nunique()} animals, {len(df['motif label_Clean'].unique())} unique motifs"
    )

print(f"\nTotal unique motifs across all timepoints: {len(all_motifs)}")

# Calculate summary statistics for global normalization
print("\nCalculating summary statistics...")
global_summary_stats = calculate_summary_stats(datasets, all_motifs, "global")

# Perform Kruskal-Wallis tests for global normalization
print("Performing Kruskal-Wallis tests for global normalization...")
global_kw_results = perform_kruskal_wallis_tests(datasets, all_motifs, "global")

# =============================================================================
# FIGURE 1: Global Normalization (each dataset sums to 1)
# =============================================================================

# Create Figure 1 with reasonable height
fig1, ax1 = plt.subplots(figsize=(24, 8))

n_motifs = len(all_motifs)
x = np.arange(n_motifs)
width = 0.25

datasets_list = list(datasets.keys())
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# Plot bars with individual points and error bars
all_bars = plot_bars_with_points_and_errors(
    ax1, x, global_summary_stats, datasets_list, colors, width
)

# Set labels and title
ax1.set_ylabel("Normalized Frequency (Global)", fontsize=12)
ax1.set_title(
    "Motif Frequency Distribution - Global Normalization\n(Bars = Mean ± SEM, Points = Individual Animals)",
    fontsize=14,
    fontweight="bold",
)
ax1.set_xticks(x + width)
ax1.set_xticklabels(all_motifs, rotation=90, ha="center")
ax1.legend()

# Calculate global distribution JSD using means
global_freqs = []
for dataset in datasets_list:
    dataset_freqs = []
    for motif in all_motifs:
        mean_freq = global_summary_stats[dataset][motif]["mean"]
        dataset_freqs.append(mean_freq)
    global_freqs.append(dataset_freqs)

jsd_12, jsd_13, jsd_23 = calculate_distribution_jsd(global_freqs)

# Calculate bracket span from the actual x-axis range
bracket_start = -width / 2  # Start from left edge of first group
bracket_end = n_motifs - 1 + 2.5 * width  # End at right edge of last group

jsd_text = f"P12-P20: {jsd_12:.3f}\nP12-P60: {jsd_13:.3f}\nP20-P60: {jsd_23:.3f}"
add_bracket_annotation(fig1, ax1, bracket_start, bracket_end, jsd_text)

# Remove manual bottom adjustment since bracket function handles this automatically
plt.tight_layout()

# Save figure as SVG
global_fig_path = os.path.join(RESULTS_DIR, "motif_analysis_global_normalization.svg")
fig1.savefig(global_fig_path, format="svg", dpi=300, bbox_inches="tight")
print(f"Saved global normalization figure: {global_fig_path}")

plt.show()

# Save global Kruskal-Wallis results
global_kw_path = os.path.join(RESULTS_DIR, "kruskal_wallis_global_normalization.csv")
global_kw_results.to_csv(global_kw_path, index=False)
print(f"Saved global Kruskal-Wallis results: {global_kw_path}")

# Print summary of significant results
n_significant_global = global_kw_results["significant"].sum()
print(
    f"Global normalization: {n_significant_global}/{len(global_kw_results)} motifs show significant differences (p < 0.05)"
)

# =============================================================================
# FIGURE 2: Domain-wise Normalization (each motif length group sums to 1)
# =============================================================================

# Group motifs by count (domain)
motif_domains = {}
for motif in all_motifs:
    if motif.startswith("["):
        count = len(ast.literal_eval(motif))
    else:
        count = 1
    if count not in motif_domains:
        motif_domains[count] = []
    motif_domains[count].append(motif)

# Sort domains and motifs within domains (preserve original order within domains)
sorted_domains = sorted(motif_domains.keys())
domain_sorted_motifs = []
domain_boundaries = []
current_pos = 0

for domain in sorted_domains:
    # Get motifs in original order for this domain
    domain_motifs_original_order = [
        motif for motif in all_motifs if motif in motif_domains[domain]
    ]
    domain_sorted_motifs.extend(domain_motifs_original_order)
    domain_boundaries.append(
        (current_pos, current_pos + len(domain_motifs_original_order) - 1, domain)
    )
    current_pos += len(domain_motifs_original_order)

# Calculate domain-wise normalized frequencies for each animal
print("Calculating domain-wise normalization...")
domain_datasets = {}

for timepoint, df in datasets.items():
    domain_data = []

    for _, row in df.iterrows():
        motif = row["motif label_Clean"]
        animal_id = row["Animal_ID"]

        # Find which domain this motif belongs to
        motif_domain = None
        for domain, domain_motifs in motif_domains.items():
            if motif in domain_motifs:
                motif_domain = domain
                break

        if motif_domain is not None:
            # Calculate domain total for this animal
            animal_domain_data = df[(df["Animal_ID"] == animal_id)]
            domain_motifs_for_animal = animal_domain_data[
                animal_domain_data["motif label_Clean"].isin(
                    motif_domains[motif_domain]
                )
            ]
            domain_total = domain_motifs_for_animal["observed"].sum()

            # Calculate domain-wise normalized frequency
            if domain_total > 0:
                domain_normalized_freq = row["observed"] / domain_total
            else:
                domain_normalized_freq = 0

            domain_data.append(
                {
                    "motif label_Clean": motif,
                    "motif size": row["motif size"],
                    "observed": row["observed"],
                    "normalized_freq": row["normalized_freq"],
                    "domain_normalized_freq": domain_normalized_freq,
                    "Animal_ID": animal_id,
                    "Timepoint": timepoint,
                    "Domain": motif_domain,
                }
            )

    domain_datasets[timepoint] = pd.DataFrame(domain_data)

# Calculate summary statistics for domain-wise normalization
domain_summary_stats = {}
for timepoint, df in domain_datasets.items():
    domain_summary_stats[timepoint] = {}

    for motif in domain_sorted_motifs:
        motif_data = df[df["motif label_Clean"] == motif]

        if len(motif_data) > 0:
            values = motif_data["domain_normalized_freq"].values

            if len(values) > 0:
                from scipy import stats

                mean_val = np.mean(values)
                sem_val = stats.sem(values) if len(values) > 1 else 0
                domain_summary_stats[timepoint][motif] = {
                    "mean": mean_val,
                    "sem": sem_val,
                    "n": len(values),
                    "values": values.tolist(),
                }
            else:
                domain_summary_stats[timepoint][motif] = {
                    "mean": 0,
                    "sem": 0,
                    "n": 0,
                    "values": [],
                }
        else:
            domain_summary_stats[timepoint][motif] = {
                "mean": 0,
                "sem": 0,
                "n": 0,
                "values": [],
            }

# Perform Kruskal-Wallis tests for domain-wise normalization
print("Performing Kruskal-Wallis tests for domain-wise normalization...")
domain_kw_results = perform_kruskal_wallis_tests(
    domain_datasets, domain_sorted_motifs, "domain"
)

# Create Figure 2 with reasonable height
fig2, ax2 = plt.subplots(figsize=(24, 8))

n_motifs_domain = len(domain_sorted_motifs)
x_domain = np.arange(n_motifs_domain)

# Plot bars with individual points and error bars
all_bars_domain = plot_bars_with_points_and_errors(
    ax2, x_domain, domain_summary_stats, datasets_list, colors, width
)

# Set labels and title
ax2.set_ylabel("Normalized Frequency (Domain-wise)", fontsize=12)
ax2.set_title(
    "Motif Frequency Distribution - Domain-wise Normalization\n(Bars = Mean ± SEM, Points = Individual Animals)",
    fontsize=14,
    fontweight="bold",
)
ax2.set_xticks(x_domain + width)
ax2.set_xticklabels(domain_sorted_motifs, rotation=90, ha="center")
ax2.legend()

# Add domain separators
for i, (start, end, domain) in enumerate(domain_boundaries[:-1]):
    separator_x = end + 0.5 + width
    ax2.axvline(x=separator_x, color="k", linestyle="--", alpha=0.7, linewidth=0.5)

# Calculate domain-wise JSDs and add brackets at the same level (without modifying y-limits)
for start, end, domain in domain_boundaries:
    # Get frequencies for this domain across all datasets using means
    domain_freqs = []
    domain_motifs = motif_domains[domain]

    for dataset in datasets_list:
        dataset_domain_freqs = []
        for motif in domain_motifs:
            mean_freq = domain_summary_stats[dataset][motif]["mean"]
            dataset_domain_freqs.append(mean_freq)
        domain_freqs.append(dataset_domain_freqs)

    # Calculate JSDs for this domain
    jsd_12, jsd_13, jsd_23 = calculate_distribution_jsd(domain_freqs)

    # Calculate bracket span using domain boundaries and actual bar layout
    # Get the actual x positions of bars for this domain
    first_motif_x = start  # x position of first motif in domain
    last_motif_x = end  # x position of last motif in domain

    # Calculate bracket span from leftmost bar edge to rightmost bar edge
    bracket_start = first_motif_x - width / 2  # Left edge of first bar group
    bracket_end = last_motif_x + width + width  # Right edge of last bar group

    jsd_text = f"P12-P20: {jsd_12:.3f}\nP12-P60: {jsd_13:.3f}\nP20-P60: {jsd_23:.3f}"
    add_bracket_annotation(fig2, ax2, bracket_start, bracket_end, jsd_text)

# Remove manual bottom adjustment since bracket function handles this automatically
plt.tight_layout()

# Save figure as SVG
domain_fig_path = os.path.join(RESULTS_DIR, "motif_analysis_domain_normalization.svg")
fig2.savefig(domain_fig_path, format="svg", dpi=300, bbox_inches="tight")
print(f"Saved domain normalization figure: {domain_fig_path}")

plt.show()

# Save domain-wise Kruskal-Wallis results
domain_kw_path = os.path.join(RESULTS_DIR, "kruskal_wallis_domain_normalization.csv")
domain_kw_results.to_csv(domain_kw_path, index=False)
print(f"Saved domain Kruskal-Wallis results: {domain_kw_path}")

# Print summary of significant results
n_significant_domain = domain_kw_results["significant"].sum()
print(
    f"Domain normalization: {n_significant_domain}/{len(domain_kw_results)} motifs show significant differences (p < 0.05)"
)

# Create a summary of all statistical results
print("\nCreating statistical summary...")
combined_results = []

# Add global results
for _, row in global_kw_results.iterrows():
    combined_results.append(
        {
            "Motif": row["Motif"],
            "Normalization": "Global",
            "H_statistic": row["H_statistic"],
            "p_value": row["p_value"],
            "significant": row["significant"],
            "n_groups": row["n_groups"],
        }
    )

# Add domain results
for _, row in domain_kw_results.iterrows():
    combined_results.append(
        {
            "Motif": row["Motif"],
            "Normalization": "Domain",
            "H_statistic": row["H_statistic"],
            "p_value": row["p_value"],
            "significant": row["significant"],
            "n_groups": row["n_groups"],
        }
    )

combined_df = pd.DataFrame(combined_results)
summary_path = os.path.join(RESULTS_DIR, "kruskal_wallis_summary.csv")
combined_df.to_csv(summary_path, index=False)
print(f"Saved combined statistical summary: {summary_path}")

# =============================================================================
# STATISTICAL SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("COMPREHENSIVE STATISTICAL ANALYSIS")
print("=" * 100)

print("\n1. GLOBAL NORMALIZATION - Overall Distribution Comparison:")
print("-" * 80)
global_jsd_12, global_jsd_13, global_jsd_23 = calculate_distribution_jsd(global_freqs)
print(f"P12 vs P20: JSD = {global_jsd_12:.4f}")
print(f"P12 vs P60: JSD = {global_jsd_13:.4f}")
print(f"P20 vs P60: JSD = {global_jsd_23:.4f}")

print("\n2. DOMAIN-WISE NORMALIZATION - Domain-specific Comparisons Only:")
print("-" * 80)
print("(No global comparison - only within-domain comparisons are meaningful)")
for start, end, domain in domain_boundaries:
    domain_freqs = []
    domain_motifs = motif_domains[domain]

    for dataset in datasets_list:
        dataset_domain_freqs = []
        for motif in domain_motifs:
            mean_freq = domain_summary_stats[dataset][motif]["mean"]
            dataset_domain_freqs.append(mean_freq)
        domain_freqs.append(dataset_domain_freqs)

    domain_jsd_12, domain_jsd_13, domain_jsd_23 = calculate_distribution_jsd(
        domain_freqs
    )

    print(f"Domain {domain} ({len(domain_motifs)} motifs):")
    print(f"  P12 vs P20: JSD = {domain_jsd_12:.4f}")
    print(f"  P12 vs P60: JSD = {domain_jsd_13:.4f}")
    print(f"  P20 vs P60: JSD = {domain_jsd_23:.4f}")

print("\n3. DOMAIN COMPOSITION:")
print("-" * 80)
for domain in sorted_domains:
    motifs_in_domain = motif_domains[domain]
    print(f"Domain {domain}: {len(motifs_in_domain)} motifs")
    for motif in motifs_in_domain:
        print(f"  {motif}")

print("\n4. SAMPLE SIZES:")
print("-" * 80)
for timepoint in datasets_list:
    n_animals = len(set(datasets[timepoint]["Animal_ID"]))
    print(f"{timepoint}: {n_animals} animals")

    # Show sample sizes for each domain
    for domain in sorted_domains:
        domain_motifs = motif_domains[domain]
        domain_data = domain_datasets[timepoint]
        animals_with_domain_data = set()
        for motif in domain_motifs:
            motif_animals = domain_data[domain_data["motif label_Clean"] == motif][
                "Animal_ID"
            ].unique()
            animals_with_domain_data.update(motif_animals)
        print(f"  Domain {domain}: {len(animals_with_domain_data)} animals with data")

print("\n5. KRUSKAL-WALLIS TEST RESULTS:")
print("-" * 80)
print(
    f"Global normalization: {n_significant_global}/{len(global_kw_results)} motifs show significant differences"
)
print(
    f"Domain normalization: {n_significant_domain}/{len(domain_kw_results)} motifs show significant differences"
)

print("\nMost significant motifs (Global normalization, p < 0.01):")
if len(global_kw_results) > 0:
    significant_global = global_kw_results[global_kw_results["p_value"] < 0.01].copy()
    if len(significant_global) > 0:
        # Sort by p_value using argsort for better type compatibility
        sort_idx = significant_global["p_value"].argsort()
        significant_global_sorted = significant_global.iloc[sort_idx]
        for _, row in significant_global_sorted.head(10).iterrows():
            print(
                f"  {row['Motif']}: H = {row['H_statistic']:.3f}, p = {row['p_value']:.6f}"
            )
    else:
        print("  No motifs with p < 0.01")
else:
    print("  No results available")

print("\nMost significant motifs (Domain normalization, p < 0.01):")
if len(domain_kw_results) > 0:
    significant_domain = domain_kw_results[domain_kw_results["p_value"] < 0.01].copy()
    if len(significant_domain) > 0:
        # Sort by p_value using argsort for better type compatibility
        sort_idx = significant_domain["p_value"].argsort()
        significant_domain_sorted = significant_domain.iloc[sort_idx]
        for _, row in significant_domain_sorted.head(10).iterrows():
            print(
                f"  {row['Motif']}: H = {row['H_statistic']:.3f}, p = {row['p_value']:.6f}"
            )
    else:
        print("  No motifs with p < 0.01")
else:
    print("  No results available")

print("\n" + "=" * 100)
print("INTERPRETATION")
print("=" * 100)
print(
    "• Global Normalization: Compares entire frequency distributions across all motifs"
)
print(
    "• Domain-wise Normalization: Compares frequency distributions only within each motif complexity level"
)
print(
    "• JSD values: 0 = identical distributions, 1 = maximally different distributions"
)
print(
    "• Lower JSD = more similar temporal patterns, Higher JSD = more divergent temporal patterns"
)
print(
    "• Each domain represents motifs of the same complexity (number of brain regions involved)"
)
print("• Domain-wise analysis isolates complexity-specific developmental patterns")
print("• Error bars represent SEM across individual animals")
print("• Individual points show data from each animal")
print("• Kruskal-Wallis test: Non-parametric test for differences between timepoints")
print(
    "• Significant Kruskal-Wallis results (p < 0.05) indicate developmental changes in motif usage"
)

print(f"\n📁 All results saved to: {RESULTS_DIR}")
print("📊 SVG figures can be edited in vector graphics software")
print("📈 CSV files contain detailed statistical results for further analysis")
