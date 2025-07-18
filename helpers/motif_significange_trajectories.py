#Takes the upsetplot.csv files from the process-nbcm.tsv pipeline output and produces an effect size trajectory for each motif.

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import fisher_exact

def motif_label(motif_str):
    try:
        items = eval(motif_str)
        return "+".join(sorted(items)) if items else "<null>"
    except:
        return "<parse_error>"

def load_and_process_files(input_dir):
    all_data = []
    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith("upsetplot.csv"):
            stage = fname.split("_")[0].upper()
            fpath = os.path.join(input_dir, fname)
            df = pd.read_csv(fpath)
            df["Motif_Label"] = df["Motifs"].apply(motif_label)
            df["Stage"] = stage
            df["Significant"] = df["P-value"].apply(lambda p: float(p) <= 0.05)
            df["Observed"] = df["Observed"].astype(int)
            all_data.append(df[["Motif_Label", "Effect Size", "Stage", "Significant", "Observed"]])
    combined_df = pd.concat(all_data)
    combined_df.to_csv(os.path.join(input_dir, "combined_effect_sizes.csv"), index=False)
    return combined_df

def compute_transition_significance(df, stage_order, output_dir):
    rows = []
    totals = df.groupby("Stage")["Observed"].sum().to_dict()
    for motif in df["Motif_Label"].unique():
        motif_data = df[df["Motif_Label"] == motif].set_index("Stage")
        for i in range(len(stage_order) - 1):
            s1, s2 = stage_order[i], stage_order[i + 1]
            if s1 in motif_data.index and s2 in motif_data.index:
                a = motif_data.loc[s1, "Observed"]
                b = totals[s1] - a
                c = motif_data.loc[s2, "Observed"]
                d = totals[s2] - c
                _, p = fisher_exact([[a, b], [c, d]])
                rows.append({
                    "Motif": motif,
                    "Transition": f"{s1}_to_{s2}",
                    "P-value": p,
                    "Significant": p <= 0.05
                })
            else:
                rows.append({
                    "Motif": motif,
                    "Transition": f"{s1}_to_{s2}",
                    "P-value": None,
                    "Significant": False
                })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(output_dir, "transition_significance.csv"), index=False)
    return df_out

def plot_motif_set(df, stage_order, trans_sig_df, motifs, output_pdf_path, output_dir, prefix=""):
    global_ymin = df["Effect Size"].min() - 0.5
    global_ymax = df["Effect Size"].max() + 0.5
    stage_to_x = {s: i for i, s in enumerate(stage_order)}
    x_numeric = np.arange(len(stage_order))

    with PdfPages(output_pdf_path) as pdf:
        for motif in motifs:
            motif_data = df[df["Motif_Label"] == motif].set_index("Stage").reindex(stage_order)
            y = motif_data["Effect Size"].values
            sig = motif_data["Significant"].fillna(False).values

            if np.all(pd.isna(y)):
                continue

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x_numeric, y, linestyle='-', color='black', marker='o', markersize=6)
            for xi, yi, s in zip(x_numeric, y, sig):
                if s:
                    ax.plot(xi, yi, marker='o', color='red', markersize=8)

            for i in range(len(stage_order) - 1):
                s1, s2 = stage_order[i], stage_order[i + 1]
                trans = trans_sig_df[
                    (trans_sig_df["Motif"] == motif) &
                    (trans_sig_df["Transition"] == f"{s1}_to_{s2}")
                ]
                if not trans.empty:
                    pval = trans["P-value"].values[0]
                    if not pd.isna(pval):
                        if pval < 1e-4:
                            stars = "****"
                        elif pval < 1e-3:
                            stars = "***"
                        elif pval < 0.01:
                            stars = "**"
                        elif pval < 0.05:
                            stars = "*"
                        else:
                            stars = None
                        if stars:
                            xm = (x_numeric[i] + x_numeric[i + 1]) / 2
                            ym = (y[i] + y[i + 1]) / 2 if not (np.isnan(y[i]) or np.isnan(y[i + 1])) else 0
                            ax.annotate(stars, xy=(xm, ym), ha='center', va='bottom', color='blue', fontsize=12)

            ax.set_title(f"Motif: {motif}")
            ax.set_ylabel("Effect Size\nlog2(Observed / Expected)")
            ax.set_xlabel("Developmental Stage")
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.set_ylim(global_ymin, global_ymax)
            ax.set_xticks(x_numeric)
            ax.set_xticklabels(stage_order)
            ax.grid(True, linestyle='--', alpha=0.5)

            pdf.savefig(fig)
            svg_path = os.path.join(output_dir, f"{prefix}{motif.replace('+','_')}_effect_size.svg")
            fig.savefig(svg_path, format='svg')
            plt.close(fig)

def plot_motif_trajectories(df, stage_order, output_dir):
    trans_sig_df = compute_transition_significance(df, stage_order, output_dir)
    motif_stage_counts = df.groupby("Motif_Label")["Stage"].nunique()
    full_motifs = motif_stage_counts[motif_stage_counts == len(stage_order)].index
    partial_motifs = motif_stage_counts[motif_stage_counts < len(stage_order)].index

    # Sort full motifs by subset size
    def subset_size(label):
        return len(label.split('+'))

    full_motif_sizes = pd.Series(full_motifs).apply(lambda x: (subset_size(x), x))
    sorted_full_motifs = [x[1] for x in sorted(full_motif_sizes, key=lambda x: (x[0], x[1]))]

    # Full motif plots
    pdf_full = os.path.join(output_dir, "motif_effect_trajectories.pdf")
    plot_motif_set(df, stage_order, trans_sig_df, sorted_full_motifs, pdf_full, output_dir)

    # Partial motif plots
    pdf_partial = os.path.join(output_dir, "motif_effect_partial_trajectories.pdf")
    plot_motif_set(df, stage_order, trans_sig_df, partial_motifs, pdf_partial, output_dir, prefix="partial_")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot motif effect size trajectories across stages.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing *_upsetplot.csv files")
    args = parser.parse_args()

    stage_order = ["P3", "P12", "P20", "P60"]
    df_all = load_and_process_files(args.input_dir)
    plot_motif_trajectories(df_all, stage_order, args.input_dir)
