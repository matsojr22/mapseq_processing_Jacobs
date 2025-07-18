import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import re
import os
from collections import defaultdict

# Set font preferences
matplotlib.rcParams['font.family'] = ['Helvetica', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['svg.fonttype'] = 'none'

# Input file
input_path = "motif_transition_significance_summary.txt"
real_comparison_file = "motif_real_p12vsP60.tsv"

# Output directory
output_dir = "motif_plots"
os.makedirs(output_dir, exist_ok=True)

# Parse the text file into a usable structure
with open(input_path, 'r') as f:
    lines = f.readlines()

# Collect data by transition
true_data = defaultdict(dict)
false_data = defaultdict(dict)
current_transition = None
for line in lines:
    line = line.strip()
    if not line:
        continue
    if re.match(r"P\d+ vs P\d+", line):
        current_transition = line.replace(" ", "")
        continue
    if 'JS Divergence' in line:
        motif_match = re.match(r"\[(.*?)\]: .*?JS Divergence = ([\d\.]+|nan), Significant = (True|False)", line)
        if motif_match:
            motif_str, divergence, significant = motif_match.groups()
            if divergence != 'nan':
                divergence = float(divergence)
                motif_clean = motif_str.replace("'", "").replace(",", "+")
                if significant == 'True':
                    true_data[current_transition][motif_clean] = divergence
                else:
                    false_data[current_transition][motif_clean] = divergence

# Override P12vsP60 with real divergence values
if os.path.exists(real_comparison_file):
    real_df = pd.read_csv(real_comparison_file, sep='\t')
    real_df = real_df.dropna(subset=['JS_Divergence'])

    true_data['P12vsP60'] = {}
    false_data['P12vsP60'] = {}

    for _, row in real_df.iterrows():
        motif = row['Motif_Label']
        divergence = float(row['JS_Divergence'])
        significant = bool(row['Significant']) if 'Significant' in row else False
        if significant:
            true_data['P12vsP60'][motif] = divergence
        else:
            false_data['P12vsP60'][motif] = divergence

# Plotting function
def plot_true_and_false(transition, true_motifs_dict, false_motifs_dict):
    true_sorted = sorted(true_motifs_dict.items(), key=lambda x: x[1], reverse=True)
    false_sorted = sorted(false_motifs_dict.items(), key=lambda x: x[1], reverse=True)
    combined = true_sorted + false_sorted

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [m for m, _ in combined]
    values = [d for _, d in combined]
    colors = ['red'] * len(true_sorted) + ['blue'] * len(false_sorted)

    ax.bar(labels, values, color=colors)
    ax.set_title(f"JS Divergences: {transition} (True=Red, False=Blue)")
    ax.set_ylabel("JS Divergence")
    ax.set_xticklabels(labels, rotation=90)

    # Set standard Y-axis range (based on global max)
    ax.set_ylim(0, max_global_divergence)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"divergence_{transition}.svg"), format="svg")
    plt.close()

# Get global Y limit
all_divergences = [d for motifs in list(true_data.values()) + list(false_data.values()) for d in motifs.values()]
max_global_divergence = max(all_divergences)

# Generate one plot per transition
for transition in true_data:
    plot_true_and_false(transition, true_data[transition], false_data.get(transition, {}))
