"""
Visualization and plotting services for NBCM processing pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from scipy.stats import binomtest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from collections import Counter
import upsetplot as up
from adjustText import adjust_text
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import os
import sys
import re
import itertools

from src.infrastructure.logger import Logger
from src.domain.services.motif_analysis_service import MotifAnalysisService
from src.domain.services.statistical_analyzer import StatisticalAnalyzer
from src.domain.services.clustering_analyzer import ClusteringAnalyzer
from src.domain.services.matrix_processor import MatrixProcessor


class PlotGenerator:
    """Visualization and plotting services"""

    def __init__(
        self,
        logger: Optional[Logger] = None,
        motif_analysis_service: Optional[MotifAnalysisService] = None,
        statistical_analyzer: Optional[StatisticalAnalyzer] = None,
        clustering_analyzer: Optional[ClusteringAnalyzer] = None,
        matrix_processor: Optional[MatrixProcessor] = None,
        data_saver: Optional[Any] = None,
    ):
        self.logger = logger if logger is not None else Logger()
        self.motif_analysis_service = motif_analysis_service
        self.statistical_analyzer = statistical_analyzer
        self.clustering_analyzer = clustering_analyzer
        self.matrix_processor = matrix_processor
        self.data_saver = data_saver

    def create_heatmap(
        self,
        matrix: np.ndarray,
        labels: List[str],
        title: str,
        output_path: str,
        plot_type: str = "green_white",
    ) -> None:
        """
        Create heatmap visualizations.

        Args:
            matrix: Input matrix
            labels: Column labels
            title: Plot title
            output_path: Output file path
            plot_type: Type of heatmap ("green_white", "han_style", "probability")
        """
        if plot_type == "green_white":
            self._create_green_white_heatmap(matrix, labels, title, output_path)
        elif plot_type == "han_style":
            self._create_han_style_heatmap(matrix, labels, title, output_path)
        elif plot_type == "probability":
            self._create_probability_heatmap(matrix, labels, title, output_path)
        else:
            self.logger.log_warning(f"Unknown heatmap type: {plot_type}")

    def _create_green_white_heatmap(
        self, matrix: np.ndarray, labels: List[str], title: str, output_path: str
    ) -> None:
        """Create green-white cluster heatmap using MatrixProcessor"""
        # Workaround for Windows recursion bug
        sys.setrecursionlimit(5000)

        ### === Green-White Cluster Heatmap === ###
        print("🔍 Generating Green-White cluster heatmap...")

        # Convert matrix to DataFrame
        df = pd.DataFrame(matrix)
        df.columns = labels

        # Dynamically build full order list
        order_full = [
            col
            for pattern in ["LM", "AL", "RL", "A", "AM", "PM", "RSP"]
            for col in df.columns
            if re.match(f"{pattern}\\d*", col)
        ]
        order_full = list(dict.fromkeys(order_full))

        if not order_full:
            raise ValueError(
                "❌ No matching columns found for green-white cluster heatmap."
            )

        order_partial = ["LM", "AL", "RL", "AM", "PM"]
        order_partial = [col for col in order_partial if col in df.columns]

        print(f"Adjusted order_full: {order_full}")
        print(f"Adjusted order_partial: {order_partial}")

        full_data = True
        df_ = df[order_full] if full_data else df[order_partial]
        print(f"Adjusted df_ columns: {df_.columns.tolist()}")

        # Use MatrixProcessor for data scaling
        if self.matrix_processor is None:
            raise ValueError("MatrixProcessor not provided")

        df_scaled, df_scaled_np = self.matrix_processor.scale_and_normalize_matrix(
            df_.values, df_.columns.tolist()
        )

        # Colormap
        grn_white_cm = LinearSegmentedColormap.from_list(
            "white_to_green", ["white", "green"], N=100
        )

        # Drop constant or all-zero rows
        df_scaled = df_scaled.loc[df_scaled.var(axis=1) > 0]

        # Final check before clustering
        if df_scaled.shape[0] < 2:
            raise ValueError(
                "❌ Too few rows remaining after variance filtering to perform clustering."
            )

        # Draw clustermap
        clusterfig = sns.clustermap(
            df_scaled,
            col_cluster=False,
            metric="cosine",
            method="average",
            cbar_kws=dict(label="Projection Strength"),
            cmap=grn_white_cm,
            vmin=0.0,
            vmax=1.0,
        )

        clusterfig.ax_heatmap.set_title(title.replace("_", " "))
        clusterfig.ax_heatmap.get_yaxis().set_visible(False)

        for ext in ["pdf", "svg", "png"]:
            clusterfig.savefig(
                os.path.normpath(
                    os.path.join(
                        os.path.dirname(output_path),
                        f"{os.path.basename(output_path).replace('.png', '')}.{ext}",
                    )
                )
            )

        print("✅ Green-White cluster heatmap saved.")
        plt.close()

    def _create_han_style_heatmap(
        self, matrix: np.ndarray, labels: List[str], title: str, output_path: str
    ) -> None:
        """Create Han-style heatmap using EXACT same logic as original script"""
        # Convert matrix to DataFrame
        df = pd.DataFrame(matrix)
        df.columns = labels

        # Han colormap - EXACT same as original
        han_cm = LinearSegmentedColormap.from_list(
            "white_to_green", ["white", "green"], N=100
        )

        # Define Han-style targets - EXACT same as original
        han_targets = ["LM", "AL", "PM", "AM", "RL"]
        han_order_full = [
            col
            for pattern in han_targets
            for col in df.columns
            if re.match(f"{pattern}\\d*", col)
        ]
        han_order_full = list(dict.fromkeys(han_order_full))

        if not han_order_full:
            raise ValueError(
                "❌ No matching columns found for Han-style target area pattern."
            )

        df_han = df[han_order_full]
        print(f"🧬 Han target columns: {df_han.columns.tolist()}")
        print("Han df shape:", df_han.shape)

        # Log-transform and normalize - EXACT same as original
        df_han = np.log1p(df_han + 1e-3)
        df_han = df_han.div(df_han.max(axis=1), axis=0)

        if df_han.isnull().values.any():
            raise ValueError("❌ NaNs found in df_han after normalization.")

        # Filter out zero-variance rows - EXACT same as original
        df_han = df_han.loc[df_han.var(axis=1) > 0].reset_index(drop=True)
        if df_han.shape[0] < 2:
            raise ValueError("❌ Not enough valid rows in df_han after filtering.")

        # Sort rows by max projection column index - EXACT same as original
        df_han["max_proj_col"] = df_han.values.argmax(axis=1)
        df_han = (
            df_han.sort_values("max_proj_col")
            .drop(columns="max_proj_col")
            .reset_index(drop=True)
        )

        # Ensure clean float type in DataFrame - EXACT same as original
        df_han = df_han.astype(float)

        # Draw Han-style heatmap - EXACT same as original
        clusterfig_han = sns.clustermap(
            df_han,
            row_cluster=False,
            col_cluster=False,
            cmap=han_cm,
            vmin=0.0,
            vmax=1.0,
            cbar_kws=dict(label="Projection Strength"),
        )

        clusterfig_han.ax_heatmap.set_title(title.replace("_", " ") + " (Han-style)")
        clusterfig_han.ax_heatmap.axes.get_yaxis().set_visible(False)

        for ext in ["pdf", "svg", "png"]:
            clusterfig_han.savefig(output_path.replace(".png", f".{ext}"))

        plt.close()
        self.logger.log_save(output_path)

    def _create_probability_heatmap(
        self, matrix: np.ndarray, labels: List[str], title: str, output_path: str
    ) -> None:
        """Create probability heatmap"""
        # Convert matrix to DataFrame
        df = pd.DataFrame(matrix)
        df.columns = labels

        # Create probability heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, cmap="viridis", cbar_kws={"label": "Probability"})
        plt.title(title.replace("_", " "))
        plt.tight_layout()

        for ext in ["pdf", "svg", "png"]:
            plt.savefig(output_path.replace(".png", f".{ext}"))

        plt.close()

    def create_tsne_plot(
        self, matrix: np.ndarray, output_path: str, title: str = "t-SNE Visualization"
    ) -> None:
        """Create t-SNE visualization using ClusteringAnalyzer"""
        # Use ClusteringAnalyzer for t-SNE analysis
        if self.clustering_analyzer is None:
            raise ValueError("ClusteringAnalyzer not provided")

        tsne_results = self.clustering_analyzer.perform_tsne_analysis(matrix)

        maxproj = tsne_results["tsne_projection"]
        tlabels = tsne_results["labels"]

        # Create plot - EXACT same as original script
        plt.figure(figsize=(12, 9))
        plt.title(title.replace("_", ""), fontsize=20)
        plt.xlabel("tSNE Component 1", fontsize=20)
        plt.ylabel("tSNE Component 2", fontsize=20)
        sc = plt.scatter(maxproj[:, 0], maxproj[:, 1], c=tlabels)
        cb = plt.colorbar(sc)
        cb.set_label("Maximum Projection Target", fontsize=20)

        # Save in multiple formats (EXACT same as original)
        for ext in ["pdf", "svg", "png"]:
            plt.savefig(output_path.replace(".png", f".{ext}"))

        plt.close()
        self.logger.log_save(output_path)

    def create_upset_plot(
        self,
        upset_data: Dict[str, Any],
        output_path: str,
        title: str = "Upset Plot",
    ) -> None:
        """Create upset plot using pre-generated data"""
        # Get the pre-generated upset data
        dfdata = upset_data.get("dfdata")
        if dfdata is None:
            self.logger.log_warning("Missing upset plot data")
            return

        # Sort data by Group and Observed - EXACT same as original script
        dfdata = dfdata.sort_values(by=["Group", "Observed"], ascending=[True, False])

        # Create upset plot using the kplot function
        fig, _ = self._kplot_original(dfdata)

        # Save in multiple formats - EXACT same as original
        for ext in ["pdf", "svg", "png"]:
            fig.savefig(output_path.replace(".png", f".{ext}"))

        plt.close()
        self.logger.log_save(output_path)

    def _kplot_original(self, df):
        """Original kplot function from the original script"""
        motiflabels = df["Motifs"].to_list()
        data = up.from_memberships(motiflabels, data=df["Observed"].to_numpy())
        xlen = df.shape[0]
        xticks = np.arange(xlen)
        uplot = up.UpSet(data, sort_by="cardinality")  # sort_by='cardinality'
        fig, ax = plt.subplots(
            2, 2, gridspec_kw={"width_ratios": [1, 3], "height_ratios": [3, 1]}
        )
        fig.set_size_inches((30, 12))
        ax[1, 0].set_ylabel("Set Totals")
        uplot.plot_matrix(ax[1, 1])
        uplot.plot_totals(ax[1, 0])
        ax[0, 0].axis("off")
        ax[0, 1].spines["bottom"].set_visible(False)
        ax[0, 1].spines["top"].set_visible(False)
        ax[0, 1].spines["right"].set_visible(False)
        width = 0.35
        dodge = width / 2
        x = np.arange(8)
        ax[1, 0].set_title("Totals")
        ax[0, 1].set_ylabel("Counts")
        ax[0, 1].set_xlim(ax[1, 1].get_xlim())
        ox = xticks - dodge
        ex = xticks + dodge
        # colorlist = ['cyan','darkgray','darkgray','red']
        colorlist = ["red", "darkblue", "black", "black"]
        cs = [colorlist[i - 1] for i in df["Group"]]
        ax[0, 1].bar(
            ox,
            df["Observed"].to_numpy(),
            width=width,
            label="Observed",
            align="center",
            color=cs,
            edgecolor="lightgray",
        )
        ax[0, 1].bar(
            ex,
            df["Expected"].to_numpy(),
            yerr=df["Expected SD"].to_numpy(),
            width=width / 2,
            label="Expected",
            align="center",
            color="gray",
            alpha=0.5,
            ecolor="lightgray",
        )
        grp_ = df["Group"].to_numpy()
        idsig = np.concatenate([np.where(grp_ == 1)[0], np.where(grp_ == 2)[0]])
        [
            ax[0, 1].text(
                ox[idsig][i] - 0.5 * dodge,
                df["Observed"].to_numpy()[idsig][i] + 1,
                s="*",
            )
            for i in range(idsig.shape[0])
        ]
        #
        ax[0, 1].xaxis.grid(False)
        ax[0, 1].xaxis.set_visible(False)
        ax[1, 1].xaxis.set_visible(False)
        ax[1, 1].xaxis.grid(False)
        # ax[0,1].legend()
        fig.tight_layout()
        return fig, ax

    def create_simple_upset_plot(
        self,
        motif_analysis_results: Dict[str, Any],
        output_path: str,
        title: str = "Simple Upset Plot",
    ) -> None:
        """Create simple upset plot using MotifAnalysisService"""
        # Get the motif data from motif analysis results
        detailed_analysis = motif_analysis_results.get("detailed_motif_analysis", [])
        motif_obs_exp_data = motif_analysis_results.get("motif_obs_exp_data", {})

        if not detailed_analysis or not motif_obs_exp_data:
            self.logger.log_warning("Missing motif analysis data for simple upset plot")
            return

        # Get the DataFrame with observed/expected data
        df_obs_exp = motif_obs_exp_data.get("df_obs_exp")
        if df_obs_exp is None:
            self.logger.log_warning("Missing motif observed/expected DataFrame")
            return

        # Use MotifAnalysisService for data preparation
        if self.motif_analysis_service is None:
            raise ValueError("MotifAnalysisService not provided")

        motif_probs = motif_analysis_results.get("motif_probs", [])
        n0 = motif_analysis_results.get("n0", 0)  # Get n0 from results

        upset_data = self.motif_analysis_service.prepare_upset_plot_data(
            df_obs_exp, motif_probs, n0
        )

        dfraw = upset_data["dfraw"]
        dfdata = upset_data["dfdata"]
        dfdata = dfdata.sort_values(by=["Group", "Observed"], ascending=[True, False])

        # Save CSV - EXACT same as original
        dfdata.to_csv(output_path.replace(".png", ".csv"), index=False)

        # Create simple upset plot - EXACT same as original script
        fig, _ = self._kplot_original(dfdata)

        # Save in multiple formats - EXACT same as original
        for ext in ["pdf", "svg", "png"]:
            fig.savefig(output_path.replace(".png", f".{ext}"))

        plt.close()
        self.logger.log_save(output_path)

    def create_effect_significance_plot(
        self,
        motif_analysis_results: Dict[str, Any],
        output_path: str,
        title: str = "Effect Significance",
    ) -> None:
        """Create effect significance plot using EXACT same data as original script"""
        # Get the raw motif data - EXACT same as original script
        if self.motif_analysis_service is None:
            raise ValueError("MotifAnalysisService not provided")

        # Get the matrix and columns from the motif analysis results
        df = motif_analysis_results.get("df")
        if df is None:
            self.logger.log_warning("Missing DataFrame for effect significance plot")
            return

        # Generate motifs and get dcounts - EXACT same as original script
        motifs, motif_labels = self.motif_analysis_service._gen_motifs(
            df.shape[1], df.columns
        )
        dcounts, _ = self.motif_analysis_service._count_motifs(
            df, motifs, return_ids=True
        )
        n0 = motif_analysis_results.get("n0")

        if dcounts is None or motif_labels is None or n0 is None:
            self.logger.log_warning(
                "Missing raw motif data for effect significance plot"
            )
            return

        # Use EXACT same function as original script
        def get_motif_sig_pts(
            dcounts,
            labels,
            prob_edge=0.2,  # Default value from original
            n0=n0,
            exclude_zeros=False,  # EXACT same as original
            p_transform=lambda x: -1 * np.log10(x),  # EXACT same as original
        ):
            from scipy.stats import binomtest

            # Get expected counts using the same method as original
            if self.motif_analysis_service is None:
                raise ValueError("MotifAnalysisService not provided")

            expected, probs = self.motif_analysis_service._get_expected_counts(
                labels, prob_edge, n0
            )

            num_motifs = dcounts.shape[0]
            assert dcounts.shape[0] == expected.shape[0]

            if exclude_zeros:
                nonzid = np.nonzero(dcounts)[0]
            else:
                nonzid = np.arange(dcounts.shape[0])

            num_nonzid_motifs = nonzid.shape[0]
            dcounts_ = dcounts[nonzid]
            expected_ = expected[nonzid]
            probs_ = probs[nonzid]

            # Effect size is log2(observed/expected) - EXACT same as original
            effect_size = np.log2((dcounts_ + 1) / (expected_ + 1))
            matches = np.zeros(num_nonzid_motifs)
            assert dcounts_.shape[0] == expected_.shape[0]
            dcounts_ = dcounts_.astype(int)

            for i in range(num_nonzid_motifs):
                pi = max(probs_[i], 1e-10)  # avoid zero or very small probs
                matches[i] = binomtest(int(dcounts_[i]), n=n0, p=pi).pvalue
                matches[i] = max(matches[i], 1e-10)
            matches = p_transform(matches)
            # matches is the significance level
            res = zip(effect_size, matches)
            mlabels = [labels[h] for h in nonzid]
            return list(res), mlabels

        # SET TO TRUE IF YOU WANT TO EXCLUDE ZERO MOTIFS - EXACT same as original
        sigs, slabels = get_motif_sig_pts(dcounts, motif_labels, exclude_zeros=False)

        # Bonferroni correction: p-threshold / Num comparisons - EXACT same as original
        alpha = 0.05
        pcutoff = -1 * np.log10(alpha / len(slabels))  # adjust alpha with argument

        list_sig = [i for (i, (e, s)) in enumerate(sigs) if s > pcutoff]
        color_labels = ["gray" for i in range(len(sigs))]
        for i in list_sig:
            e, s = sigs[i]
            if e > 0:  # overrepresented
                color_labels[i] = "red"
            else:
                color_labels[i] = "blue"

        hide_singlets = True
        if hide_singlets:
            mask = [i for (i, l) in enumerate(slabels) if len(l) > 1]

        # Extract effect sizes and significances from sigs - EXACT same as original
        effect_sizes = [e for e, s in sigs]
        significances = [s for e, s in sigs]

        # Create plot - EXACT same as original script
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(20, 20)
        plt.rc("text", usetex=False)
        plt.rc("font", family="serif")
        ax.set_title(title.replace("_", ""), fontsize=16)
        ax.set_xlabel("Effect Size \n$log_2($observed/expected$)$", fontsize=16)
        ax.set_ylabel("Significance\n $-log_{10}(P)$", fontsize=16)
        ax.axhline(y=pcutoff, linestyle="--")
        ax.axvline(x=0, linestyle="--")
        ax.text(x=-0.5, y=pcutoff + 0.05, s="P-value cutoff", fontsize=16)

        # Prepare data for plotting - EXACT same as original
        sigs = list(zip(effect_sizes, significances))

        # Helper function - EXACT same as original
        def subset_list(lis, ids):
            return [lis[i] for i in ids]

        def concatenate_list_data(slist, join="+"):
            result = []
            for i in slist:
                sub = ""
                for j in i:
                    if sub:
                        sub = sub + join + str(j)
                    else:
                        sub += str(j)
                result.append(sub)
            return result

        # Scatter plot - EXACT same as original
        ax.scatter(*zip(*subset_list(sigs, mask)), c=subset_list(color_labels, mask))

        # Prepare text labels - EXACT same as original
        from adjustText import adjust_text

        pretty_slabels = concatenate_list_data(subset_list(slabels, mask))
        coordinates = subset_list(sigs, mask)
        texts = []

        for n, (z, y) in enumerate(coordinates):
            txt = pretty_slabels[n]
            texts.append(ax.text(z, y, txt, fontsize=12))

        # Adjust y-axis limits - EXACT same as original
        y_vals = [y for _, y in subset_list(sigs, mask)]
        padding = 0.1 * (max(y_vals) - min(y_vals))  # Add 10% padding
        ax.set_ylim(min(y_vals) - padding, max(y_vals) + padding)

        # Adjust x-axis limits with padding - EXACT same as original
        x_vals = [x for x, _ in subset_list(sigs, mask)]
        x_padding = 0.1 * (max(x_vals) - min(x_vals))  # 10% padding

        x_min = min(x_vals) - x_padding
        x_max = max(x_vals) + x_padding

        # Ensure symmetric padding if range is around 0 - EXACT same as original
        x_abs_max = max(abs(x_min), abs(x_max))
        ax.set_xlim(-x_abs_max, x_abs_max)

        # Adjust text positions to avoid overlap - EXACT same as original
        adjust_text(
            texts,
            expand_points=(1.5, 2.5),  # Add padding around points
            force_text=1,  # Increase separation force for text
            force_points=1,  # Increase separation force for points
        )

        # Save in multiple formats - EXACT same as original
        for ext in ["pdf", "svg", "png"]:
            plt.savefig(output_path.replace(".png", f".{ext}"))

        plt.close()
        self.logger.log_save(output_path)

    def create_blueyellow_probability_heatmap(
        self, matrix: np.ndarray, columns: List[str], output_path: str, sample_name: str
    ) -> None:
        """Create blue-yellow probability heatmap using MatrixProcessor"""
        if self.matrix_processor is None:
            raise ValueError("MatrixProcessor not provided")

        # Generate probability matrix
        probmat = self.matrix_processor.generate_probability_matrix(matrix, columns)

        # Create plot - EXACT same as original script
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(sample_name.replace("_", ""), fontsize=20)
        colors2 = ["darkblue", "#1f9ed1", "#26ffc5", "#ffc526", "yellow"]
        cm2 = LinearSegmentedColormap.from_list("white_to_red", colors2, N=100)
        ax.set_facecolor("#a8a8a8")
        ax = sns.heatmap(
            probmat.T,
            mask=probmat.T == 1,
            ax=ax,
            cbar_kws=dict(label="$P(B | A)$"),
            cmap=cm2,
        )
        ax.set_xlabel("Area A", fontsize=16)
        ax.set_ylabel("Area B", fontsize=16)

        # Save in multiple formats
        for ext in ["pdf", "svg", "png"]:
            plt.savefig(output_path.replace(".png", f".{ext}"))

        plt.close()
        self.logger.log_save(output_path)

    def create_region_probabilities_plot(
        self, region_probabilities: Dict[str, float], output_path: str, sample_name: str
    ) -> None:
        """Create region probabilities plot"""
        # Create plot - EXACT same as original script
        plt.figure(figsize=(8, 5))
        plt.bar(list(region_probabilities.keys()), list(region_probabilities.values()))
        plt.title("Region-specific Probabilities")
        plt.ylabel("Probability")
        plt.xlabel("Region")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        self.logger.log_save(output_path)

    def create_roots_plot(
        self, roots: List[float], output_path: str, sample_name: str
    ) -> None:
        """Create roots scatterplot"""
        # Create plot - EXACT same as original script
        plt.figure(figsize=(8, 5))
        plt.scatter(range(len(roots)), roots)
        plt.title("Roots")
        plt.ylabel("Root Value")
        plt.xlabel("Index")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        self.logger.log_save(output_path)

    def create_calculated_value_plot(
        self, calculated_value: float, output_path: str, sample_name: str
    ) -> None:
        """Create calculated value LaTeX expression plot"""
        # Create LaTeX expression
        latex_output = r"$" + str(calculated_value) + r"$"

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            latex_output,
            fontsize=16,
            va="center",
            ha="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        plt.title("Calculated Value Visualization", fontsize=16)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        self.logger.log_save(output_path)

    def create_simplified_pi_plot(
        self, simplified_pi: str, output_path: str, sample_name: str
    ) -> None:
        """Create simplified Pi LaTeX expression plot using EXACT same method as original"""
        # Use EXACT same method as original script's save_latex_expression
        from sympy import latex

        # Convert string back to sympy expression if needed
        if isinstance(simplified_pi, str):
            # Try to evaluate as sympy expression
            try:
                from sympy import sympify

                simplified_pi = sympify(simplified_pi)
            except:
                # If it's already a string representation, use it directly
                pass

        # Format exactly like original
        latex_output = r"$" + latex(simplified_pi) + r"$"  # Use single-dollar format

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            latex_output,
            fontsize=16,
            va="center",
            ha="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)  # Remove borders

        plt.title("Simplified Pi Visualization", fontsize=16)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        self.logger.log_save(output_path)

    def create_pie_chart(
        self,
        matrix: np.ndarray,
        output_path: str,
        title: str = "Number of Targets per Cell",
    ) -> None:
        """Create pie chart using StatisticalAnalyzer"""
        # Use StatisticalAnalyzer for data preparation
        if self.statistical_analyzer is None:
            raise ValueError("StatisticalAnalyzer not provided")

        pie_data = self.statistical_analyzer.calculate_pie_chart_data(matrix)

        counts = pie_data["counts"]
        row_names = pie_data["row_names"]

        # Create DataFrame for plotting
        c_df = pd.DataFrame(counts, columns=["# Cells"], index=row_names)
        c_np = c_df.to_numpy(copy=True).flatten()
        c_tot = c_np.sum()

        # Create plot - EXACT same as original script
        plt.figure(figsize=(10, 10))
        plt.title(title.replace("_", ""))
        glabels = ["1 target \n {:0.3}%".format(100 * c_np[0] / c_tot)]
        glabels += [
            "{} targets \n {:0.3}%".format(i + 2, 100 * j / c_tot)
            for (i, j) in zip(range(c_np.shape[0] - 1), c_np[1:])
        ]
        patches, texts = plt.pie(c_df.to_numpy().flatten(), labels=glabels)
        [txt.set_fontsize(8) for txt in texts]

        # Save in multiple formats (same as original)
        for ext in ["pdf", "svg", "png"]:
            plt.savefig(output_path.replace(".png", f".{ext}"))

        plt.close()
        self.logger.log_save(output_path)

    def create_kmeans_plot(
        self,
        normalized_matrix: np.ndarray,
        consensus_k: int,
        output_path: str,
        title: str = "K-means Clustering",
    ) -> None:
        """Create K-means clustering plot using EXACT same logic as original script"""
        # Use EXACT same method as original script
        from sklearn.cluster import KMeans
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np
        import matplotlib.pyplot as plt
        import os

        # Convert to DataFrame if it's not already
        if not isinstance(normalized_matrix, pd.DataFrame):
            df = pd.DataFrame(normalized_matrix)
        else:
            df = normalized_matrix

        # Clustering - EXACT same as original
        X = df.to_numpy()
        k = min(consensus_k, X.shape[0])  # Prevents ValueError when too few samples
        km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
        clusters, regions = km.cluster_centers_.shape

        # Save Data - EXACT same as original
        df.to_csv(
            os.path.normpath(
                os.path.join(os.path.dirname(output_path), "motif_obs_exp.csv")
            )
        )

        # Plotting setup - EXACT same as original
        scolors = ["black", "red", "orange", "yellow"]
        scm = LinearSegmentedColormap.from_list("white_to_red", scolors, N=100)
        fig, ax = plt.subplots(figsize=(10, 10))

        # Axes labels and style - EXACT same as original
        ax.set_title("K-means Clustering")
        ax.set_xlabel("Regions")
        ax.set_ylabel("Cluster")
        ax.set_xticks(range(regions))
        ax.set_xticklabels(df.columns.to_list(), rotation=45, ha="right")
        ax.set_yticks(range(clusters))
        ax.set_yticklabels([str(i) for i in range(1, clusters + 1)])
        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_visible(False)

        # Data plot - EXACT same as original
        X_vals = range(regions)
        for i in range(km.n_clusters):
            y_vals = np.full(regions, i)
            size = km.cluster_centers_[i]
            size_norm = (
                (size - size.min()) / (size.max() - size.min())
                if size.max() > size.min()
                else size
            )
            ax_ = ax.scatter(x=X_vals, y=y_vals, s=1000, c=size_norm, cmap=scm)

        # Add colorbar - EXACT same as original
        fig.colorbar(ax_, label="Normalized Projection Strength")

        # Save plot - EXACT same as original
        for ext in ["pdf", "svg", "png"]:
            fig.savefig(output_path.replace(".png", f".{ext}"))

        plt.close()
        self.logger.log_save(output_path)

    def create_cluster_diagnostics(
        self, matrix: np.ndarray, output_path: str, title: str = "Cluster Diagnostics"
    ) -> None:
        """Create cluster diagnostics plot using ClusteringAnalyzer"""
        # Use ClusteringAnalyzer for cluster diagnostics
        if self.clustering_analyzer is None:
            raise ValueError("ClusteringAnalyzer not provided")

        diagnostics = self.clustering_analyzer.compute_cluster_diagnostics(matrix)

        K = diagnostics["K"]
        inertias = diagnostics["inertias"]
        elbow_k = diagnostics["elbow_k"]
        silhouette_k = diagnostics["silhouette_k"]
        gap_k = diagnostics["gap_k"]
        bic_k = diagnostics["bic_k"]
        consensus_k = diagnostics["consensus_k"]

        # Create plot - EXACT same as original script
        plt.figure(figsize=(10, 7))
        plt.plot(K[: len(inertias)], inertias, "x-", color="blue", label="Inertia")
        plt.axvline(
            x=elbow_k, color="gray", linestyle="--", label=f"Elbow: k={elbow_k}"
        )
        plt.axvline(
            x=silhouette_k,
            color="green",
            linestyle="--",
            label=f"Silhouette: k={silhouette_k}",
        )
        plt.axvline(x=gap_k, color="orange", linestyle="--", label=f"Gap: k={gap_k}")
        if bic_k is not None:
            plt.axvline(
                x=bic_k, color="purple", linestyle="--", label=f"BIC: k={bic_k}"
            )
        plt.axvline(
            x=consensus_k,
            color="red",
            linestyle=":",
            linewidth=2.0,
            label=f"Consensus: k={consensus_k}",
        )
        plt.xlabel("k", fontsize=20)
        plt.ylabel("Inertia", fontsize=20)
        plt.title("Cluster Evaluation Methods", fontsize=20)
        plt.legend()
        plt.tight_layout()

        # Save in multiple formats (same as original)
        for ext in ["pdf", "svg", "png"]:
            elbow_plt = plt.gcf()
            elbow_plt.savefig(output_path.replace(".png", f".{ext}"))

        plt.close()
        self.logger.log_save(output_path)

    def create_all_visualizations(
        self,
        matrix: np.ndarray,
        columns: List[str],
        results: Dict[str, Any],
        output_dir: str,
        sample_name: str,
        pe_num: float,
        observed_cells: int,
    ) -> None:
        """Create all visualizations for the NBCM pipeline"""
        # Store n0 for use in other functions
        self._n0 = observed_cells

        # Create analysis directory
        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        # Create heatmaps
        self.create_heatmap(
            matrix,
            columns,
            "Green-White Cluster Heatmap",
            os.path.join(
                analysis_dir, f"{sample_name}_green_white_cluster_heatmap.png"
            ),
            "green_white",
        )

        self.create_heatmap(
            matrix,
            columns,
            "Han-style Cluster Heatmap",
            os.path.join(analysis_dir, f"{sample_name}_Hanstyle_cluster_heatmap.png"),
            "han_style",
        )

        # Create t-SNE plot
        self.create_tsne_plot(
            matrix, os.path.join(analysis_dir, f"{sample_name}_tsne.png")
        )

        # Create upset plots
        if "upset_plot_data" in results:
            self.create_upset_plot(
                results["upset_plot_data"],
                os.path.join(analysis_dir, f"{sample_name}_upsetplot.png"),
            )

        # Create pie chart
        self.create_pie_chart(
            matrix, os.path.join(analysis_dir, f"{sample_name}_num_targets_pie.png")
        )

        # Create clustering plots
        self.create_kmeans_plot(
            matrix,
            results["consensus_k"],
            os.path.join(analysis_dir, f"{sample_name}_kmeans.png"),
        )

        self.create_cluster_diagnostics(
            matrix, os.path.join(analysis_dir, f"{sample_name}_cluster_diagnostics.png")
        )

        # Create effect significance plot
        self.create_effect_significance_plot(
            results["motif_analysis_results"],
            os.path.join(analysis_dir, f"{sample_name}_effect_significance.png"),
        )

        # Create blue-yellow probability heatmap
        self.create_blueyellow_probability_heatmap(
            matrix,
            columns,
            os.path.join(
                analysis_dir, f"{sample_name}_blueyellow_probability_heatmap.png"
            ),
            sample_name,
        )

        # Create region probabilities plot
        if "region_probabilities" in results:
            self.create_region_probabilities_plot(
                results["region_probabilities"],
                os.path.join(output_dir, f"{sample_name}_Region_Probabilities.png"),
                sample_name,
            )

        # Create roots plot
        if "roots" in results:
            self.create_roots_plot(
                results["roots"],
                os.path.join(output_dir, f"{sample_name}_Roots.png"),
                sample_name,
            )

        # Create calculated value plot
        if "calculated_value" in results:
            self.create_calculated_value_plot(
                results["calculated_value"],
                os.path.join(output_dir, f"{sample_name}_Calculated_Value.png"),
                sample_name,
            )

        # Create simplified Pi plot
        if "simplified_pi" in results:
            self.create_simplified_pi_plot(
                results["simplified_pi"],
                os.path.join(output_dir, f"{sample_name}_Simplified_Pi.png"),
                sample_name,
            )

        # Create per-cell projection strength plot
        if "per_cell_data" in results:
            self.create_per_cell_projection_strength(
                results["per_cell_data"],
                sample_name,
                os.path.join(analysis_dir, f"{sample_name}_per_cell_proj_strength.svg"),
            )

        # Create extended data figure
        self.create_extended_data_figure(
            results["motif_analysis_results"],
            os.path.join(
                analysis_dir, f"{sample_name}_ExtendedDataFig10_Recreation.svg"
            ),
            results,
        )

        # Create panel G figure
        self.create_panel_g_figure(
            results["motif_analysis_results"],
            os.path.join(
                analysis_dir, f"{sample_name}_panel_g_broadcasting_from_canonical.svg"
            ),
            sample_name,
        )

    def create_per_cell_projection_strength(
        self, per_cell_data: Dict[str, Any], sample_name: str, output_path: str
    ) -> None:
        """Create per-cell projection strength visualization using pre-generated data"""
        # Get the pre-generated data
        df = per_cell_data.get("df")
        motif_labels = per_cell_data.get("motif_labels", [])
        dcounts = per_cell_data.get("dcounts", [])
        non0cell_ids = per_cell_data.get("non0cell_ids", [])
        has_multi_region_motifs = per_cell_data.get("has_multi_region_motifs", False)

        if not has_multi_region_motifs:
            # If no multi-region motifs, create a simple histogram
            projection_strength = df.max(axis=1)
            plt.figure(figsize=(12, 8))
            plt.hist(projection_strength, bins=50, alpha=0.7, edgecolor="black")
            plt.xlabel("Maximum Projection Strength")
            plt.ylabel("Number of Cells")
            plt.title(f"Per-Cell Projection Strength Distribution - {sample_name}")
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path)
            plt.close()
            self.logger.log_save(output_path)
            return

        # Create plot with subplots for each motif (like original)
        plot_titles = ["_".join(l) for l in motif_labels]
        ncols = 2
        nrows = int(np.ceil(len(non0cell_ids) / ncols))
        fig = plt.figure(figsize=(16, 35))

        for n, (i, cellids) in enumerate(non0cell_ids):
            ax = fig.add_subplot(nrows, ncols, n + 1)
            title = plot_titles[i]
            ax.set_title(title)
            ax.set_xticks(np.arange(df.shape[1]))
            ax.set_xticklabels(df.columns.to_list(), rotation=90)
            ax.set_ylabel("Projection Strength")

            x = df.iloc[cellids, :].to_numpy()

            # Add observed count legend
            obs = dcounts[i]
            textstr = f"Observed: {int(obs)}"
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax.text(
                0.55,
                0.9,
                textstr,
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
                bbox=props,
            )

            # Plot individual cell lines
            for j in range(x.shape[0]):
                ax.plot(
                    np.arange(df.shape[1]),
                    x[j],
                    markerfacecolor="none",
                    alpha=0.2,
                    c="gray",
                )

            # Plot mean with error bars
            yerr = x.std(axis=0) / np.sqrt(x.shape[0])
            ax.errorbar(
                x=np.arange(df.shape[1]),
                y=x.mean(axis=0),
                yerr=yerr,
                marker="o",
                color="blue",
                linewidth=2,
                markersize=6,
            )

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        self.logger.log_save(output_path)

    def create_extended_data_figure(
        self,
        motif_analysis_results: Dict[str, Any],
        output_path: str,
        results: Dict[str, Any] = None,
    ) -> None:
        """Create Extended Data Fig 10 Recreation - K-means cluster centroid heatmap"""
        # Get the normalized matrix from motif analysis results
        normalized_matrix = motif_analysis_results.get("normalized_matrix")
        # Get consensus_k from results if available, otherwise use default
        consensus_k = 8  # Default value
        if results and "consensus_k" in results:
            consensus_k = results["consensus_k"]

        if normalized_matrix is None:
            self.logger.log_warning(
                "Missing normalized matrix for Extended Data Fig 10"
            )
            return

        # Convert to DataFrame if it's not already
        if not isinstance(normalized_matrix, pd.DataFrame):
            normalized_matrix = pd.DataFrame(normalized_matrix)

        # Use EXACT same logic as original script
        from sklearn.cluster import KMeans
        from matplotlib.colors import LinearSegmentedColormap

        scolors = ["white", "red"]
        scm = LinearSegmentedColormap.from_list("white_to_red", scolors, N=256)

        k_clusters = consensus_k
        kmeans = KMeans(n_clusters=k_clusters, random_state=42)
        kmeans.fit(normalized_matrix)
        centroids = kmeans.cluster_centers_

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(centroids, aspect="auto", cmap=scm, vmin=0, vmax=1)
        ax.set_title(
            "Projection Motif Clusters (Extended Data Fig. 10 Style)", fontsize=14
        )
        ax.set_xlabel("Target Regions", fontsize=12)
        ax.set_ylabel("Cluster ID", fontsize=12)
        ax.set_xticks(range(len(normalized_matrix.columns)))
        ax.set_xticklabels(normalized_matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(k_clusters))
        ax.set_yticklabels([f"Cluster {i+1}" for i in range(k_clusters)])

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(top=False, bottom=True, left=True, right=False)

        cbar = fig.colorbar(im, ax=ax, orientation="vertical")
        cbar.set_label("Normalized Projection Strength", rotation=270, labelpad=15)

        fig.tight_layout()
        fig.savefig(output_path, format="svg")
        plt.close()
        self.logger.log_save(output_path)

    def create_panel_g_figure(
        self, motif_analysis_results: Dict[str, Any], output_path: str, sample_name: str
    ) -> None:
        """Create panel G broadcasting from canonical figure using EXACT same method as original"""
        import networkx as nx
        import ast
        from matplotlib.lines import Line2D

        # Get the upset plot CSV data which contains the motif information
        upset_csv_path = output_path.replace(
            "_panel_g_broadcasting_from_canonical.svg", "_upsetplot.csv"
        )
        upset_csv_path = upset_csv_path.replace("/analysis/", "/analysis/")

        try:
            df_motifs = pd.read_csv(upset_csv_path)
        except FileNotFoundError:
            self.logger.log_warning(f"Expected motif CSV not found: {upset_csv_path}")
            return

        # Parse 'Motifs' column into real Python lists - EXACT same as original
        if isinstance(df_motifs["Motifs"].iloc[0], str):
            df_motifs["Motifs"] = df_motifs["Motifs"].apply(ast.literal_eval)

        # Drop malformed rows just in case
        df_motifs = df_motifs[df_motifs["Motifs"].apply(lambda x: isinstance(x, list))]

        # Filter for 2-region motifs only - EXACT same as original
        df_motifs["motif_size"] = df_motifs["Motifs"].apply(len)
        df_2region = df_motifs[df_motifs["motif_size"] == 2].copy()

        # Ensure numeric values - EXACT same as original
        for col in ["Observed", "Expected", "P-value"]:
            df_2region[col] = pd.to_numeric(df_2region[col], errors="coerce")

        # Classify significance label - EXACT same as original
        def get_sig_label_from_group(row):
            if row["Group"] == 1:
                return "over"
            elif row["Group"] == 2:
                return "under"
            else:
                return "ns"

        df_2region["sig_label"] = df_2region.apply(get_sig_label_from_group, axis=1)

        # Build networkx graph - EXACT same as original
        G = nx.Graph()
        max_obs = df_2region["Observed"].max()
        if pd.isna(max_obs) or max_obs <= 0:
            max_obs = 1  # prevent division by zero

        for row in df_2region.itertuples():
            regions = row.Motifs
            if len(regions) != 2:
                continue
            r1, r2 = regions
            color = {"over": "red", "under": "blue", "ns": "black"}.get(
                row.sig_label, "black"
            )
            width = 1 + 9 * (row.Observed / max_obs)
            G.add_edge(r1, r2, weight=width, color=color)

        # Layout and draw - EXACT same as original
        manual_region_order = ["RSP", "PM", "AM", "AL", "LM"]
        sorted_nodes = [r for r in manual_region_order if r in G.nodes]

        if not sorted_nodes:
            self.logger.log_warning("No valid 2-region motifs found. Skipping plot.")
            return

        angle_step = 2 * np.pi / len(sorted_nodes)
        pos = {
            region: np.array([np.cos(i * angle_step), np.sin(i * angle_step)])
            for i, region in enumerate(sorted_nodes)
        }

        edges = G.edges(data=True)
        colors = [e[2]["color"] for e in edges]
        widths = [e[2]["weight"] for e in edges]

        plt.figure(figsize=(8, 8))
        nx.draw_networkx(
            G, pos, with_labels=True, node_size=1000, edge_color=colors, width=widths
        )
        plt.title("Fig 10g: 2-Region Broadcasting Motifs (from canonical CSV)")

        legend_elements = [
            Line2D([0], [0], color="red", lw=2, label="Overrepresented"),
            Line2D([0], [0], color="blue", lw=2, label="Underrepresented"),
            Line2D([0], [0], color="black", lw=2, label="Not Significant"),
        ]
        plt.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=2,
            frameon=False,
        )
        plt.tight_layout()

        plt.savefig(output_path, format="svg")
        plt.close()
        self.logger.log_save(output_path)
