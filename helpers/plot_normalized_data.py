
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
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
        color_map = plt.colormaps.get_cmap('Set1')
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
                norm = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
                normalized_data.append(norm)
                color = color_lookup[sample_id]
                label = sample_id if sample_id not in plotted_samples else None
                plt.plot(regions, norm, color=color, alpha=0.4, label=label)

            plotted_samples.add(sample_id)
            norm_array = np.array(normalized_data)
            all_data.append(norm_array)

        combined_data = np.vstack(all_data)
        mean_vals = combined_data.mean(axis=0)
        sem_vals = combined_data.std(axis=0) / np.sqrt(combined_data.shape[0])

        plt.errorbar(regions, mean_vals, yerr=sem_vals, fmt='-o', color='black',
                     linewidth=2, capsize=5, label='Mean ± SEM')

        plt.title(f"Normalized Regional Data: {title}")
        plt.xlabel("Region")
        plt.ylabel("Normalized Value (0-1)")
        plt.grid(True)

        # Deduplicate legend entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title="Sample ID")

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
