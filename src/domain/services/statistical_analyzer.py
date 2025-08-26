"""
Statistical analysis and calculations for NBCM processing pipeline.
"""

import numpy as np
import sympy
from typing import Dict, List, Tuple, Any
from sympy import symbols, Product, Array, N, latex
from scipy.stats import binomtest, binom

from src.infrastructure.logger import Logger


class StatisticalAnalyzer:
    """Statistical analysis and calculations"""

    def __init__(self):
        self.logger = Logger()

    def calculate_projections(
        self, matrix: np.ndarray, columns: List[str]
    ) -> Tuple[Dict[str, int], int]:
        """
        Calculate projection metrics per region.

        Args:
            matrix: Normalized matrix
            columns: Column labels

        Returns:
            Tuple[Dict[str, int], int]: Column counts and total projections
        """
        # Binary projection presence count (how many cells project to each region)
        column_counts = {
            region: np.count_nonzero(matrix[:, idx])
            for idx, region in enumerate(columns)
        }

        # Total projection events (presence-based)
        total_projections = sum(column_counts.values())

        self.logger.log_step(
            "Projection calculation", f"Total projections: {total_projections}"
        )
        self.logger.log_statistics("Column counts", len(column_counts))

        return column_counts, total_projections

    def solve_for_roots(
        self, projections: Dict[str, int], observed_cells: int
    ) -> Tuple[List[float], Any]:
        """
        Solve for N0 roots using symbolic computation.

        Args:
            projections: Projection counts per region
            observed_cells: Number of observed cells

        Returns:
            Tuple[List[float], Any]: Roots and pi expression
        """
        N0, k = symbols("N_0 k")
        m = len(projections) - 1
        s = Array(list(projections.values()))
        pi = 1 - Product((1 - (s[k] / N0)), (k, 0, m)).doit()
        soln = sympy.solve(pi * N0 - observed_cells)
        roots = [N(x).as_real_imag()[0] for x in soln]

        self.logger.log_step("Root solving", f"Found {len(roots)} roots")
        return roots, pi

    def calculate_probabilities(
        self, projections: Dict[str, int], total_projections: int
    ) -> Dict[str, float]:
        """
        Calculate region-specific probabilities.

        Args:
            projections: Projection counts per region
            total_projections: Total number of projections

        Returns:
            Dict[str, float]: Region-specific probabilities
        """
        probabilities = {
            region: (count / total_projections) for region, count in projections.items()
        }

        self.logger.log_step(
            "Probability calculation", f"Calculated {len(probabilities)} probabilities"
        )
        return probabilities

    def calculate_effect_significance_data(
        self,
        observed_counts: List[int],
        expected_counts: List[float],
        motif_labels: List[str],
    ) -> Dict[str, Any]:
        """
        Calculate effect sizes and significance values for motif analysis.

        Args:
            observed_counts: List of observed motif counts
            expected_counts: List of expected motif counts
            motif_labels: List of motif labels

        Returns:
            Dict containing effect sizes, significances, color labels, and mask
        """
        try:
            effect_sizes = []
            significances = []

            for obs, exp in zip(observed_counts, expected_counts):
                # Effect size is log2(observed/expected)
                effect_size = np.log2((obs + 1) / (exp + 1))

                # Calculate p-value using binomial test
                if exp > 0:
                    n0 = sum(observed_counts)  # Total observed cells
                    pi = max(float(exp / n0), 1e-10)  # Probability, avoid zero
                    p_value = float(binomtest(int(obs), n=int(n0), p=pi).pvalue)
                    p_value = max(p_value, 1e-10)  # Avoid zero p-values
                else:
                    p_value = 1e-10

                # Significance is -log10(p_value)
                significance = -1 * np.log10(p_value)

                effect_sizes.append(effect_size)
                significances.append(significance)

            # Bonferroni correction
            alpha = 0.05
            pcutoff = -1 * np.log10(alpha / len(significances))

            # Color coding
            color_labels = []
            for e, s in zip(effect_sizes, significances):
                if s > pcutoff:
                    if e > 0:  # overrepresented
                        color_labels.append("red")
                    else:
                        color_labels.append("blue")
                else:
                    color_labels.append("gray")

            # Hide singlets (motifs with only one region)
            mask = [i for i, label in enumerate(motif_labels) if len(label) > 1]

            return {
                "effect_sizes": effect_sizes,
                "significances": significances,
                "color_labels": color_labels,
                "mask": mask,
                "pcutoff": pcutoff,
                "alpha": alpha,
            }

        except Exception as e:
            self.logger.log_error(e, "Calculating effect significance data")
            raise

    def calculate_pie_chart_data(self, matrix: np.ndarray) -> Dict[str, Any]:
        """
        Calculate data for pie chart showing number of targets per cell.

        Args:
            matrix: Input matrix

        Returns:
            Dict containing target counts and labels
        """
        try:
            # For each cell (row), determine how many projections it makes
            data = matrix.copy()
            cells, regions = data.shape
            target_counts = []

            for cell in range(cells):
                num_targets = int(np.nonzero(data[cell])[0].shape[0])
                target_counts.append(num_targets)

            target_counts = np.array(target_counts)

            # Get unique counts and their frequencies
            unique_targets, counts = np.unique(target_counts, return_counts=True)

            # Create row names
            row_names = ["1 target"]
            row_names += [f"{i + 2} targets" for i in range(len(counts) - 1)]

            return {
                "target_counts": target_counts,
                "unique_targets": unique_targets,
                "counts": counts,
                "row_names": row_names,
            }

        except Exception as e:
            self.logger.log_error(e, "Calculating pie chart data")
            raise

    def compute_motif_probabilities(
        self, pe_num: float, total_regions: int
    ) -> Dict[int, float]:
        """
        Compute probabilities for each possible motif type.

        Args:
            pe_num: Probability of an edge (p_e)
            total_regions: Number of brain regions

        Returns:
            Dict[int, float]: Motif type to probability mapping
        """
        # Ensure pe_num is a native float
        pe_num = float(pe_num)

        # Compute motif probabilities using safe probability mass function (PMF)
        motif_probs = {
            n: (pe_num**n) * ((1 - pe_num) ** (total_regions - n))
            for n in range(1, total_regions + 1)
        }

        # Normalize probabilities to ensure they sum to exactly 1
        total_motif_prob = sum(motif_probs.values())
        if total_motif_prob > 0:
            motif_probs = {k: v / total_motif_prob for k, v in motif_probs.items()}

        self.logger.log_step(
            "Motif probability calculation",
            f"Calculated {len(motif_probs)} motif probabilities",
        )
        self.logger.log_statistics("Total motif probability", sum(motif_probs.values()))

        return motif_probs

    def perform_binomial_tests(
        self, matrix: np.ndarray, motif_probs: Dict[int, float], observed_cells: int
    ) -> List[Tuple[int, float, float]]:
        """
        Perform binomial tests for each observed motif size.

        Args:
            matrix: Normalized matrix
            motif_probs: Motif probabilities
            observed_cells: Number of observed cells

        Returns:
            List[Tuple[int, float, float]]: Motif size, expected probability, p-value
        """
        # Identify observed motif sizes and counts
        observed_motif_sizes = np.unique(np.sum(matrix > 0, axis=1))
        motif_counts = [
            np.sum(np.sum(matrix > 0, axis=1) == size) for size in observed_motif_sizes
        ]

        # Perform binomial test for each observed motif size
        binomial_test_results = []
        for n_proj in observed_motif_sizes:
            obs_count = int(motif_counts[observed_motif_sizes.tolist().index(n_proj)])
            prob = float(motif_probs.get(n_proj, 0))
            p_value = binom.sf(obs_count - 1, int(observed_cells), prob)
            binomial_test_results.append((n_proj, prob, p_value))

        self.logger.log_step(
            "Binomial testing", f"Performed {len(binomial_test_results)} tests"
        )
        return binomial_test_results

    def calculate_pe_empirical(self, region_probabilities: Dict[str, float]) -> float:
        """
        Calculate empirical p_e from region probabilities.

        Args:
            region_probabilities: Region-specific probabilities

        Returns:
            float: Empirical p_e value
        """
        pe_empirical = np.mean(list(region_probabilities.values()))
        self.logger.log_statistics("Empirical p_e", pe_empirical)
        return pe_empirical

    def solve_pe_symbolic(
        self, projections: Dict[str, int], observed_cells: int
    ) -> List[float]:
        """
        Solve for symbolic p_e values.

        Args:
            projections: Projection counts per region
            observed_cells: Number of observed cells

        Returns:
            List[float]: Valid symbolic p_e solutions
        """
        pe = symbols("p_e")
        pe_solutions = sympy.solve(
            (1 - (1 - pe) ** len(projections)) * sum(projections.values())
            - observed_cells,
            pe,
            force=True,
        )

        # Extract only real solutions within (0,1)
        valid_symbolic_solutions = []
        for sol in pe_solutions:
            try:
                if sol.is_real:
                    val = float(sol.evalf())
                    if 0 < val < 1:
                        valid_symbolic_solutions.append(val)
            except Exception as e:
                self.logger.log_warning(
                    f"Skipping symbolic solution {sol} due to error: {e}"
                )

        self.logger.log_step(
            "Symbolic p_e solving",
            f"Found {len(valid_symbolic_solutions)} valid solutions",
        )
        return valid_symbolic_solutions

    def select_pe_value(
        self, symbolic_solutions: List[float], empirical_pe: float
    ) -> float:
        """
        Select the best p_e estimate from available solutions.

        Args:
            symbolic_solutions: Valid symbolic p_e solutions
            empirical_pe: Empirical p_e value

        Returns:
            float: Selected p_e value
        """
        if symbolic_solutions:
            pe_num = np.mean(symbolic_solutions)
            self.logger.log_step(
                "p_e selection", "Using mean of valid symbolic solutions"
            )
        else:
            pe_num = empirical_pe
            self.logger.log_step("p_e selection", "Using empirical p_e value")

        # Ensure pe_num is within (0,1)
        if not (0 < pe_num < 1):
            self.logger.log_warning(f"Selected p_e = {pe_num} is outside (0,1)")

        self.logger.log_statistics("Final p_e value", pe_num)
        return pe_num

    def calculate_expected_value(
        self, pe_num: float, num_regions: int, observed_cells: int
    ) -> float:
        """
        Calculate expected observed projections.

        Args:
            pe_num: Probability of edge
            num_regions: Number of regions
            observed_cells: Number of observed cells

        Returns:
            float: Expected observed projections
        """
        calculated_value = (1 - (1 - pe_num) ** num_regions) * observed_cells
        self.logger.log_statistics("Expected observed projections", calculated_value)
        return calculated_value

    def calculate_standard_deviation(
        self, scaled_value: float, total_projections: int
    ) -> float:
        """
        Calculate standard deviation for statistical tests.

        Args:
            scaled_value: Scaled probability value
            total_projections: Total number of projections

        Returns:
            float: Standard deviation
        """
        std_dev = np.sqrt(scaled_value * total_projections * (1 - scaled_value))
        self.logger.log_statistics("Standard deviation", std_dev)
        return std_dev

    def compute_umi_total_counts(
        self, matrix: np.ndarray, region_labels: List[str]
    ) -> Dict[str, float]:
        """
        Compute total UMI counts for each region.

        Args:
            matrix: Filtered matrix
            region_labels: Region labels

        Returns:
            Dict[str, float]: Region to UMI count mapping
        """
        umi_total_counts = {
            region: float(np.sum(matrix[:, idx]))
            for idx, region in enumerate(region_labels)
        }

        self.logger.log_step(
            "UMI count calculation",
            f"Calculated counts for {len(umi_total_counts)} regions",
        )
        return umi_total_counts
