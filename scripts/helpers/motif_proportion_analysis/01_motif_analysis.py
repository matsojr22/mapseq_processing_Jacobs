# motif_analysis_pipeline.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import ttest_ind, ks_2samp, entropy
import numpy as np
from scipy.spatial.distance import jensenshannon

# === Load Data ===
matrix_df = pd.read_csv("motif_percent_matrix_by_age.csv")
matrix_df.set_index("Motif_Label", inplace=True)

# === Melt into Long Format ===
long_df = matrix_df.reset_index().melt(id_vars="Motif_Label", var_name="Age", value_name="Percent")
long_df["Motif_List"] = long_df["Motif_Label"].apply(eval)
long_df["Motif_Count"] = long_df["Motif_List"].apply(len)

ordered_ages = ["P3", "P12", "P20", "P60"]
unique_degrees = sorted(long_df["Motif_Count"].unique())

# === Prepare PDF Outputs ===
pdf_path_sorted = "motif_barplots_sorted.pdf"
pdf_path_by_rank = "motif_barplots_ranked.pdf"

# === Plot PDFs ===
for by_rank, out_path in zip([False, True], [pdf_path_sorted, pdf_path_by_rank]):
    with PdfPages(out_path) as pdf:
        for age in ordered_ages:
            age_data = long_df[long_df['Age'] == age]
            fig, axs = plt.subplots(5, 1, figsize=(12, 20))
            fig.suptitle(f"Motif Observed % per Degree - {age}" + (" (Ranked)" if by_rank else ""), fontsize=16)

            for i, degree in enumerate(unique_degrees):
                subset = age_data[age_data["Motif_Count"] == degree]
                if subset.empty or i >= 5:
                    axs[i].axis("off")
                    continue

                subset = subset.copy()
                if by_rank:
                    subset = subset.sort_values("Percent", ascending=False)
                else:
                    subset = subset.sort_values("Motif_Label")

                sns.barplot(data=subset, x="Motif_Label", y="Percent", ax=axs[i], palette='viridis')
                axs[i].set_title(f"{degree}-Target Motifs")
                axs[i].set_ylabel('% of Observed')
                axs[i].set_xlabel('')
                axs[i].tick_params(axis='x', rotation=90)
                axs[i].set_ylim(0, 100)

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)

# === Histogram Similarity Metrics ===
js_data = []
similarity_lines = ["Histogram Comparison Results:\n"]
with open("histogram_similarity_summary.txt", "w") as f:
    for degree in unique_degrees:
        subsets = {
            age: long_df[(long_df["Age"] == age) & (long_df["Motif_Count"] == degree)]["Percent"].values
            for age in ordered_ages
        }
        age_transitions = [(ordered_ages[i], ordered_ages[i + 1]) for i in range(len(ordered_ages) - 1)] + [("P12", "P60")]
    for age1, age2 in age_transitions:
            vec1, vec2 = subsets[age1], subsets[age2]

            p = np.asarray(vec1) / np.sum(vec1) if np.sum(vec1) > 0 else np.zeros_like(vec1)
            q = np.asarray(vec2) / np.sum(vec2) if np.sum(vec2) > 0 else np.zeros_like(vec2)
            min_len = min(len(p), len(q))
            js_div = jensenshannon(p[:min_len], q[:min_len])**2

            t_stat, t_p = ttest_ind(vec1, vec2, equal_var=False)
            ks_stat, ks_p = ks_2samp(vec1, vec2)

            line = (f"Degree {degree}: {age1} vs {age2} -> "
                    f"Welch's p = {t_p:.4e}, KS p = {ks_p:.4e}, JS Divergence = {js_div:.4f}, "
                    f"Significant = {t_p < 0.05}, {ks_p < 0.05}, {js_div > 0.05}")
            print(line)
            f.write(line + "\n")
            js_data.append({"Degree": degree, "Comparison": f"{age1} vs {age2}", "JS_Divergence": js_div})

# === Plot JS Divergence from Histogram Similarity ===
if js_data:
    js_df = pd.DataFrame(js_data)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=js_df, x="Comparison", y="JS_Divergence", hue="Degree", palette="magma")
    plt.title("JS Divergence Across Age Comparisons by Motif Degree")
    plt.ylabel("JS Divergence")
    plt.xlabel("Age Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("js_divergence_histogram_comparisons.png")
    plt.close()

# === Per-Motif Percent Change Analysis Across Ages ===
with open("motif_transition_significance_summary.txt", "w") as f:
    f.write("Per-Motif Transition Comparison\n")
    # Build list of transitions including P12 vs P60
    age_transitions = [(ordered_ages[i], ordered_ages[i + 1]) for i in range(len(ordered_ages) - 1)] + [("P12", "P60")]
    for age1, age2 in age_transitions:
        f.write(f"\n{age1} vs {age2}\n")
        for motif in matrix_df.index:
            v1 = matrix_df.loc[motif, age1]
            v2 = matrix_df.loc[motif, age2]
            p_vec = np.array([v1, 100 - v1]) / 100 if v1 + v2 > 0 else np.zeros(2)
            q_vec = np.array([v2, 100 - v2]) / 100 if v1 + v2 > 0 else np.zeros(2)
            js_motif = jensenshannon(p_vec, q_vec)**2
            significance_flag = js_motif > 0.05
            line = f"{motif}: {age1} = {v1:.2f}%, {age2} = {v2:.2f}% -> JS Divergence = {js_motif:.4f}, Significant = {significance_flag}"
            f.write(line + "\n")

# === Hierarchical Clustering Heatmap ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(matrix_df[ordered_ages].values)
sns.clustermap(X_scaled, method='ward', cmap='viridis', row_cluster=True, col_cluster=False,
               figsize=(12, 16), yticklabels=matrix_df.index, xticklabels=ordered_ages)
plt.title("Hierarchical Clustering (Standardized % by Age)")
plt.savefig("hclust_heatmap.png")
plt.close()

# === PCA + KMeans ===
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(pca_result)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, palette='tab10')
plt.title("PCA + KMeans Clustering")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig("pca_kmeans_plot.png")
plt.close()

# === Annotate PCA axes with contributing features ===
pca_loadings = pd.DataFrame(pca.components_, columns=ordered_ages, index=["PC1", "PC2"])
pca_loadings.T.to_csv("pca_feature_loadings.csv")

print("\nTop contributing features to PC1 and PC2:")
print(pca_loadings.T.abs().sort_values("PC1", ascending=False).head(5))
print(pca_loadings.T.abs().sort_values("PC2", ascending=False).head(5))

# === Export PCA results with cluster assignments ===
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=matrix_df.index)
pca_df["KMeans_Cluster"] = labels
pca_df.to_csv("motif_pca_clusters.csv")
