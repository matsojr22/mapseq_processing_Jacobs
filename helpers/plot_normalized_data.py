import os
import glob
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend for headless environments
matplotlib.rcParams['svg.fonttype'] = 'none'  # Keep text editable in SVGs
matplotlib.rcParams['font.family'] = ['Arial']  # List of fonts to try

import matplotlib.pyplot as plt
from pathlib import Path

def main(data_dir, output_dir='plots'):
    csv_files = glob.glob(os.path.join(data_dir, "*_raw_data.csv"))
    grouped_files = {}

    for file in csv_files:
        filename = Path(file).stem
        parts = filename.split('_')
        title = '_'.join(parts[1:-2]) if parts[-2] == "raw" else '_'.join(parts[1:-1])
        if title not in grouped_files:
            grouped_files[title] = []
        grouped_files[title].append(file)

    os.makedirs(output_dir, exist_ok=True)

    for title, files in grouped_files.items():
        plt.figure(figsize=(12, 6))

        # Determine unique sample IDs and assign colors
        sample_ids = [Path(f).stem.split('_')[0] for f in files]
        unique_samples = sorted(set(sample_ids))
        color_map = plt.colormaps.get_cmap('tab10')
        color_lookup = {
            sample: color_map(i / max(1, len(unique_samples) - 1))
            for i, sample in enumerate(unique_samples)
        }

        all_data = []
        plotted_samples = set()

        for file in files:
            df = pd.read_csv(file)
            sample_id = Path(file).stem.split('_')[0]
            regions = df.columns[1:]

            normalized_data = []
            for _, row in df.iterrows():
                values = row[1:].values.astype(float)
                normalized_data.append(values)
                color = color_lookup[sample_id]
                label = sample_id if sample_id not in plotted_samples else None
                plt.plot(regions, values, color=color, alpha=0.9, label=label)

            plotted_samples.add(sample_id)
            norm_array = np.array(normalized_data)
            all_data.append(norm_array)

        combined_data = np.vstack(all_data)
        mean_vals = combined_data.mean(axis=0)
        sem_vals = combined_data.std(axis=0) / np.sqrt(combined_data.shape[0])
        std_vals = combined_data.std(axis=0)

        plt.errorbar(regions, mean_vals, fmt='-o', color='black',  # yerr=sem_vals,
                     linewidth=2, capsize=5, label='Mean')

        plt.title(f"Normalized Regional Data: {title}")
        plt.xlabel("Region")
        plt.ylabel("Normalized Value (0-1)")
        plt.grid(False)

        # Deduplicate and sort legend entries, with 'Mean' last
        handles, labels = plt.gca().get_legend_handles_labels()
        label_handle_pairs = dict(zip(labels, handles))

        mean_handle = label_handle_pairs.pop('Mean', None)
        sorted_pairs = sorted(label_handle_pairs.items(), key=lambda x: x[0])
        if mean_handle:
            sorted_pairs.append(('Mean', mean_handle))

        sorted_labels, sorted_handles = zip(*sorted_pairs)
        plt.legend(sorted_handles, sorted_labels, title="Sample ID")

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{title}_normalized_plot.svg")
        plt.savefig(output_path, format='svg')
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot normalized data with mean and SEM.")
    parser.add_argument('data_dir', help="Directory containing *_raw_data.csv files")
    parser.add_argument('--output_dir', default='plots', help="Directory to save SVG plots")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
