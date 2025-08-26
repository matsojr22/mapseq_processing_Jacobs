"""
Clustering analysis and motif detection for NBCM processing pipeline.
"""

import numpy as np
from typing import Dict, List, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from collections import Counter

from src.infrastructure.logger import Logger


class ClusteringAnalyzer:
    """Clustering analysis and motif detection"""

    def __init__(self):
        self.logger = Logger()

    def determine_optimal_clusters(self, matrix: np.ndarray) -> int:
        """
        Determine optimal number of clusters using multiple methods.

        Args:
            matrix: Input matrix for clustering

        Returns:
            int: Optimal number of clusters
        """
        if matrix.shape[0] < 2:
            self.logger.log_warning("Too few samples for clustering analysis")
            return 1

        # Set global random seed for reproducibility (same as original script)
        np.random.seed(42)

        K = list(
            range(2, 15)
        )  # skip k=1 for silhouette and BIC - same as original script

        # 1. Elbow Method (Inertia)
        inertias = []
        for k_val in K:
            k = min(
                k_val, matrix.shape[0]
            )  # Prevents ValueError when too few samples - same as original script
            km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(matrix)
            inertias.append(km.inertia_)

        # Compute elbow using max second derivative
        inertia_deltas = np.diff(inertias)
        inertia_deltas2 = np.diff(inertia_deltas)
        elbow_k = (
            K[np.argmax(inertia_deltas2) + 2] if len(inertia_deltas2) > 0 else K[0]
        )

        # 2. Silhouette Score
        sil_scores = []
        for k_val in K:
            if k_val >= matrix.shape[0]:
                sil_scores.append(-1)
                continue
            km = KMeans(n_clusters=k_val, n_init="auto", random_state=42).fit(matrix)
            sil_scores.append(silhouette_score(matrix, km.labels_))
        silhouette_k = K[np.argmax(sil_scores)]

        # 3. Gap Statistic
        gap_k = self._compute_gap_statistic(matrix, K)

        # 4. BIC using GMM
        bic_k = self._compute_bic(matrix, K)

        # Consensus vote
        votes = [v for v in [elbow_k, silhouette_k, gap_k, bic_k] if v is not None]
        vote_counts = Counter(votes)
        consensus_k = vote_counts.most_common(1)[0][0]

        # Debug logging to match original script output
        print(
            f"Elbow k = {elbow_k}, Silhouette k = {silhouette_k}, Gap k = {gap_k}, BIC k = {bic_k}"
        )
        print(f"Consensus k = {consensus_k}")

        self.logger.log_step(
            "Clustering analysis",
            f"Elbow: {elbow_k}, Silhouette: {silhouette_k}, Gap: {gap_k}, BIC: {bic_k}",
        )
        self.logger.log_step("Consensus clustering", f"Optimal k: {consensus_k}")

        return consensus_k

    def compute_cluster_diagnostics(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive cluster diagnostics using multiple methods.

        Args:
            matrix: Input matrix for clustering analysis

        Returns:
            Dict containing all clustering diagnostics data
        """
        try:
            if matrix.shape[0] < 2:
                self.logger.log_warning("Too few samples for clustering analysis")
                return {"consensus_k": 1, "inertias": [], "K": []}

            # Set global random seed for reproducibility (same as original script)
            np.random.seed(42)

            K = list(
                range(2, 15)
            )  # skip k=1 for silhouette and BIC - same as original script
            inertias = []

            for k_val in K:
                k = min(
                    k_val, matrix.shape[0]
                )  # Prevents ValueError when too few samples
                km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(matrix)
                inertias.append(km.inertia_)

            # Compute elbow using max second derivative (same as original)
            inertia_deltas = np.diff(inertias)
            inertia_deltas2 = np.diff(inertia_deltas)
            elbow_k = (
                K[np.argmax(inertia_deltas2) + 2] if len(inertia_deltas2) > 0 else K[0]
            )

            # Silhouette Score (same as original)
            sil_scores = []
            for k_val in K:
                if k_val >= matrix.shape[0]:
                    sil_scores.append(-1)
                    continue
                km = KMeans(n_clusters=k_val, n_init="auto", random_state=42).fit(
                    matrix
                )
                sil_scores.append(silhouette_score(matrix, km.labels_))
            silhouette_k = K[np.argmax(sil_scores)]

            # Gap Statistic (same as original)
            gap_k = self._compute_gap_statistic(matrix, K)

            # BIC using GMM (same as original)
            bic_k = self._compute_bic(matrix, K)

            # Consensus vote (same as original)
            votes = [v for v in [elbow_k, silhouette_k, gap_k, bic_k] if v is not None]
            vote_counts = Counter(votes)
            consensus_k = vote_counts.most_common(1)[0][0]

            # Debug logging to match original script output
            print(
                f"Elbow k = {elbow_k}, Silhouette k = {silhouette_k}, Gap k = {gap_k}, BIC k = {bic_k}"
            )
            print(f"Consensus k = {consensus_k}")

            return {
                "consensus_k": consensus_k,
                "elbow_k": elbow_k,
                "silhouette_k": silhouette_k,
                "gap_k": gap_k,
                "bic_k": bic_k,
                "inertias": inertias,
                "sil_scores": sil_scores,
                "K": K,
            }

        except Exception as e:
            self.logger.log_error(e, "Computing cluster diagnostics")
            raise

    def perform_kmeans_clustering(self, matrix: np.ndarray, k: int) -> Dict[str, Any]:
        """
        Perform K-means clustering and return results.

        Args:
            matrix: Input matrix
            k: Number of clusters

        Returns:
            Dict containing clustering results
        """
        try:
            # Clustering (same as original)
            X = matrix
            k = min(k, X.shape[0])  # Prevents ValueError when too few samples
            km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X)
            clusters, regions = km.cluster_centers_.shape

            # Calculate normalized cluster centers for visualization
            normalized_centers = []
            for i in range(km.n_clusters):
                size = km.cluster_centers_[i]
                size_norm = (
                    (size - size.min()) / (size.max() - size.min())
                    if size.max() > size.min()
                    else size
                )
                normalized_centers.append(size_norm)

            return {
                "kmeans_model": km,
                "clusters": clusters,
                "regions": regions,
                "cluster_centers": km.cluster_centers_,
                "normalized_centers": normalized_centers,
                "labels": km.labels_,
            }

        except Exception as e:
            self.logger.log_error(e, "Performing K-means clustering")
            raise

    def perform_tsne_analysis(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Perform t-SNE dimensionality reduction analysis.

        Args:
            matrix: Input matrix for t-SNE analysis

        Returns:
            Dict containing t-SNE results
        """
        try:
            import pandas as pd
            from sklearn.manifold import TSNE

            # Convert matrix to DataFrame to match original script's df.to_numpy(copy=True)
            df = pd.DataFrame(matrix)

            # Perform t-SNE (EXACT same as original)
            maxproj = TSNE(n_components=2, metric="cosine").fit_transform(
                df.to_numpy(copy=True)
            )

            # Get labels (EXACT same as original)
            tlabels = df.to_numpy(copy=True).argmax(axis=1)

            return {"tsne_projection": maxproj, "labels": tlabels, "dataframe": df}

        except Exception as e:
            self.logger.log_error(e, "Performing t-SNE analysis")
            raise

    def _compute_gap_statistic(
        self, matrix: np.ndarray, K: List[int], refs: int = 10
    ) -> int:
        """Compute gap statistic for optimal cluster determination"""
        gaps = []
        for k_val in K:
            if k_val >= matrix.shape[0]:
                gaps.append(-np.inf)
                continue
            km = KMeans(n_clusters=k_val, n_init="auto", random_state=42).fit(matrix)
            disp = np.mean(
                np.min(cdist(matrix, km.cluster_centers_, "euclidean"), axis=1)
            )

            ref_disps = []
            for _ in range(refs):
                X_ref = np.random.uniform(
                    matrix.min(axis=0), matrix.max(axis=0), matrix.shape
                )
                km_ref = KMeans(n_clusters=k_val, n_init="auto", random_state=42).fit(
                    X_ref
                )
                ref_disp = np.mean(
                    np.min(cdist(X_ref, km_ref.cluster_centers_, "euclidean"), axis=1)
                )
                ref_disps.append(ref_disp)

            gap = np.log(np.mean(ref_disps)) - np.log(disp)
            gaps.append(gap)

        return K[np.argmax(gaps)] if gaps else None

    def _compute_bic(self, matrix: np.ndarray, K: List[int]) -> int:
        """Compute BIC for optimal cluster determination"""
        bics = []
        bic_valid_k = []
        for k_val in K:
            if k_val >= matrix.shape[0]:
                continue
            try:
                gmm = GaussianMixture(
                    n_components=k_val, n_init=1, random_state=42
                ).fit(matrix)
                bics.append(gmm.bic(matrix))
                bic_valid_k.append(k_val)
            except:
                continue

        return bic_valid_k[np.argmin(bics)] if bics else None

    def analyze_motifs(self, matrix: np.ndarray, columns: List[str]) -> Dict[str, Any]:
        """
        Analyze projection motifs from the matrix.

        Args:
            matrix: Normalized matrix
            columns: Column labels

        Returns:
            Dict[str, Any]: Motif analysis results
        """
        # Convert matrix to binary (presence/absence)
        binary_matrix = (matrix > 0).astype(int)

        # Count motif sizes
        motif_sizes = np.sum(binary_matrix, axis=1)
        unique_sizes, size_counts = np.unique(motif_sizes, return_counts=True)

        # Calculate motif size distribution
        motif_distribution = dict(zip(unique_sizes, size_counts))

        # Find most common motifs
        motif_patterns = []
        for i in range(binary_matrix.shape[0]):
            pattern = tuple(binary_matrix[i, :])
            motif_patterns.append(pattern)

        pattern_counts = Counter(motif_patterns)
        most_common_patterns = pattern_counts.most_common(10)

        # Calculate motif complexity metrics
        avg_motif_size = np.mean(motif_sizes)
        motif_size_std = np.std(motif_sizes)
        max_motif_size = np.max(motif_sizes)
        min_motif_size = np.min(motif_sizes)

        results = {
            "motif_distribution": motif_distribution,
            "most_common_patterns": most_common_patterns,
            "avg_motif_size": avg_motif_size,
            "motif_size_std": motif_size_std,
            "max_motif_size": max_motif_size,
            "min_motif_size": min_motif_size,
            "total_motifs": len(motif_patterns),
            "unique_patterns": len(pattern_counts),
        }

        self.logger.log_step(
            "Motif analysis",
            f"Found {len(motif_patterns)} motifs with {len(pattern_counts)} unique patterns",
        )
        self.logger.log_statistics("Average motif size", avg_motif_size)

        return results

    def compute_cluster_centroids(
        self, matrix: np.ndarray, labels: np.ndarray, k: int
    ) -> np.ndarray:
        """
        Compute cluster centroids.

        Args:
            matrix: Input matrix
            labels: Cluster labels
            k: Number of clusters

        Returns:
            np.ndarray: Cluster centroids
        """
        centroids = np.zeros((k, matrix.shape[1]))
        for i in range(k):
            cluster_mask = labels == i
            if np.any(cluster_mask):
                centroids[i] = np.mean(matrix[cluster_mask], axis=0)

        self.logger.log_step("Centroid computation", f"Computed {k} cluster centroids")
        return centroids

    def analyze_cluster_characteristics(
        self, matrix: np.ndarray, labels: np.ndarray, columns: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze characteristics of each cluster.

        Args:
            matrix: Input matrix
            labels: Cluster labels
            columns: Column labels

        Returns:
            Dict[str, Any]: Cluster characteristics
        """
        unique_labels = np.unique(labels)
        cluster_characteristics = {}

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_data = matrix[cluster_mask]

            # Calculate cluster statistics
            cluster_size = np.sum(cluster_mask)
            cluster_mean = np.mean(cluster_data, axis=0)
            cluster_std = np.std(cluster_data, axis=0)

            # Find dominant regions (highest mean projection)
            region_means = dict(zip(columns, cluster_mean))
            dominant_regions = sorted(
                region_means.items(), key=lambda x: x[1], reverse=True
            )[:3]

            # Calculate motif characteristics
            binary_cluster = (cluster_data > 0).astype(int)
            motif_sizes = np.sum(binary_cluster, axis=1)
            avg_motif_size = np.mean(motif_sizes)

            cluster_characteristics[f"cluster_{label}"] = {
                "size": cluster_size,
                "mean_projections": region_means,
                "dominant_regions": dominant_regions,
                "avg_motif_size": avg_motif_size,
                "std_projections": dict(zip(columns, cluster_std)),
            }

        self.logger.log_step(
            "Cluster analysis", f"Analyzed {len(unique_labels)} clusters"
        )
        return cluster_characteristics

    def calculate_cluster_similarity(self, centroids: np.ndarray) -> np.ndarray:
        """
        Calculate similarity matrix between clusters.

        Args:
            centroids: Cluster centroids

        Returns:
            np.ndarray: Similarity matrix
        """
        similarity_matrix = np.zeros((centroids.shape[0], centroids.shape[0]))

        for i in range(centroids.shape[0]):
            for j in range(centroids.shape[0]):
                if i != j:
                    # Use cosine similarity
                    similarity = np.dot(centroids[i], centroids[j]) / (
                        np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j])
                    )
                    similarity_matrix[i, j] = similarity

        self.logger.log_step(
            "Cluster similarity",
            f"Computed similarity matrix for {centroids.shape[0]} clusters",
        )
        return similarity_matrix
