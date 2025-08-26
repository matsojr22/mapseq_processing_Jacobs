"""
Main application service orchestrating the NBCM processing pipeline.
"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Any
import numpy as np

from src.domain.models import ProcessingConfig, ProcessingResult
from src.infrastructure.logger import Logger
from src.infrastructure.data.data_loader import NBCMDataLoader
from src.infrastructure.data.data_saver import NBCMDataSaver
from src.domain.services.matrix_processor import MatrixProcessor
from src.domain.services.statistical_analyzer import StatisticalAnalyzer
from src.domain.services.clustering_analyzer import ClusteringAnalyzer
from src.domain.services.motif_analysis_service import MotifAnalysisService
from src.presentation.visualization.plot_generator import PlotGenerator


class NBCMProcessingService:
    """Main application service orchestrating the entire pipeline"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = Logger()

        # Initialize all services
        self.data_loader = NBCMDataLoader()
        self.data_saver = NBCMDataSaver()
        self.matrix_processor = MatrixProcessor()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.clustering_analyzer = ClusteringAnalyzer()
        self.motif_analysis_service = MotifAnalysisService(self.logger)
        self.plot_generator = PlotGenerator()

    def process(self) -> ProcessingResult:
        """
        Main processing pipeline.

        Returns:
            ProcessingResult: Complete processing results
        """
        self.logger.log_step("Processing pipeline", "Starting NBCM data processing")

        # Step 1: Load and validate data
        self.logger.log_step(
            "Data loading", "Loading NBCM matrix and validating labels"
        )
        matrix, labels = self.data_loader.load_and_validate(self.config)

        # Step 2: Process matrix (clean, filter, normalize)
        self.logger.log_step("Matrix processing", "Cleaning and filtering matrix")
        filtered_matrix, max_neg_value, final_umi_threshold = (
            self.matrix_processor.clean_and_filter(matrix, self.config)
        )

        # Step 3: Remove neg/inj columns
        self.logger.log_step(
            "Column filtering", "Removing negative control and injection columns"
        )
        filtered_matrix, columns = self.matrix_processor.remove_neg_inj_columns(
            filtered_matrix, labels
        )

        # Step 4: Normalize matrix
        self.logger.log_step("Normalization", "Normalizing matrix rows")
        normalized_matrix = self.matrix_processor.normalize_rows(filtered_matrix)

        # Step 5: Remove zero rows after normalization
        normalized_matrix = self.matrix_processor.remove_zero_rows(normalized_matrix)

        # Step 6: Calculate observed cells
        observed_cells = normalized_matrix.shape[0]
        self.logger.log_step("Cell counting", f"Observed cells: {observed_cells}")

        # Step 7: Statistical analysis
        self.logger.log_step(
            "Statistical analysis", "Calculating projections and probabilities"
        )
        projections, total_projections = (
            self.statistical_analyzer.calculate_projections(normalized_matrix, columns)
        )

        # Step 8: Solve for N0 roots
        self.logger.log_step("Root solving", "Solving for N0 roots")
        roots, pi_expression = self.statistical_analyzer.solve_for_roots(
            projections, observed_cells
        )

        # Simplify pi expression (same as original script)
        import sympy

        simplified_pi = sympy.simplify(pi_expression)
        self.logger.log_step("Pi simplification", f"Simplified Pi: {simplified_pi}")

        # Step 9: Calculate region probabilities
        region_probabilities = self.statistical_analyzer.calculate_probabilities(
            projections, total_projections
        )

        # Step 10: Solve for p_e
        self.logger.log_step("p_e calculation", "Solving for p_e values")
        symbolic_solutions = self.statistical_analyzer.solve_pe_symbolic(
            projections, observed_cells
        )
        empirical_pe = self.statistical_analyzer.calculate_pe_empirical(
            region_probabilities
        )
        pe_num = self.statistical_analyzer.select_pe_value(
            symbolic_solutions, empirical_pe
        )

        # Step 11: Calculate motif probabilities
        total_regions = len(columns)
        motif_probs = self.statistical_analyzer.compute_motif_probabilities(
            pe_num, total_regions
        )

        # Step 12: Perform binomial tests
        binomial_test_results = self.statistical_analyzer.perform_binomial_tests(
            normalized_matrix, motif_probs, observed_cells
        )

        # Step 13: Calculate expected value and standard deviation
        calculated_value = self.statistical_analyzer.calculate_expected_value(
            pe_num, total_regions, observed_cells
        )

        # Calculate scaled value for standard deviation
        safe_psdict = {
            label: max(region_probabilities.get(label, 0), 1e-10) for label in columns
        }
        log_scaled_value = sum(np.log(safe_psdict[label]) for label in columns)
        scaled_value = np.exp(log_scaled_value)
        standard_deviation = self.statistical_analyzer.calculate_standard_deviation(
            scaled_value, total_projections
        )

        # Step 14: Calculate UMI total counts
        umi_total_counts = self.statistical_analyzer.compute_umi_total_counts(
            filtered_matrix, columns
        )

        # Step 16: Select valid N0 value
        valid_N0 = [root for root in roots if root > observed_cells]
        if valid_N0:
            N0_value = max(valid_N0)
            self.logger.log_step("N0 selection", f"Selected N0: {N0_value}")
        else:
            raise ValueError(
                f"No valid positive real root found for N0 that is greater than observed_cells ({observed_cells})"
            )

        # Step 17: Create processing result
        result = ProcessingResult(
            filtered_matrix=filtered_matrix,
            normalized_matrix=normalized_matrix,
            columns=columns,
            projections=projections,
            total_projections=total_projections,
            region_probabilities=region_probabilities,
            roots=roots,
            pi_expression=simplified_pi,
            binomial_test_results=binomial_test_results,
            expected_value=calculated_value,
            standard_deviation=standard_deviation,
            umi_total_counts=umi_total_counts,
            consensus_k=0,  # Will be updated after clustering
            calculated_value=calculated_value,
            observed_cells=observed_cells,
            N0_value=N0_value,
            pe_num=pe_num,
            final_umi_threshold=final_umi_threshold,
            motif_over=[],  # Will be populated after upsetplot generation
            motif_under=[],  # Will be populated after upsetplot generation
        )

        # Step 18: Save results
        self.logger.log_step("Result saving", "Saving all processing results")
        self.data_saver.save_results(result, self.config)

        # Step 19: Clustering analysis (after data is saved)
        self.logger.log_step(
            "Clustering analysis", "Determining optimal number of clusters"
        )

        # Load normalized matrix from CSV file like the original script
        normalized_csv_path = os.path.join(
            self.config.out_dir, f"{self.config.sample_name}_Normalized_Matrix.csv"
        )
        df_normalized = pd.read_csv(normalized_csv_path)
        clustering_matrix = df_normalized.to_numpy()

        consensus_k = self.clustering_analyzer.determine_optimal_clusters(
            clustering_matrix
        )

        # Update result with consensus k
        result.consensus_k = consensus_k

        # Step 20: Clustering analysis completed

        # Step 21: Generate motif analysis files
        self.logger.log_step("Motif analysis", "Generating motif analysis files")

        # Perform motif analysis
        motif_analysis_results = self.motif_analysis_service.analyze_motifs(
            result.normalized_matrix,
            result.columns,
            result.pe_num,
            result.observed_cells,
        )

        # Create DataFrames for saving
        motif_analysis_results["motif_counts_df"] = (
            self.motif_analysis_service.get_motif_counts_dataframe(
                motif_analysis_results["df"], motif_analysis_results
            )
        )
        motif_analysis_results["pie_chart_df"] = (
            self.motif_analysis_service.get_pie_chart_dataframe(
                motif_analysis_results["target_pie_data"]
            )
        )

        # Save motif analysis files
        analysis_dir = os.path.join(self.config.out_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        self.data_saver.save_motif_analysis_files(
            motif_analysis_results, analysis_dir, self.config.sample_name
        )

        # Step 22: Generate motif raw data files (BEFORE PlotGenerator)
        self.logger.log_step("Motif analysis", "Generating motif raw data files")
        motif_raw_data_files = (
            self.motif_analysis_service.generate_motif_raw_data_files(
                result.normalized_matrix, result.columns, self.config.sample_name
            )
        )

        # Step 23: Save motif raw data files
        motif_raw_data_dir = os.path.join(
            self.config.out_dir, "analysis", "motif_raw_data"
        )
        self.data_saver.save_motif_raw_data_files(
            motif_raw_data_files, motif_raw_data_dir
        )

        # Step 24: Generate upset plot data (BEFORE PlotGenerator)
        self.logger.log_step("Motif analysis", "Generating upset plot data")

        # Get the data needed for upset plot generation
        df_obs_exp = motif_analysis_results["motif_obs_exp_data"]["df_obs_exp"]
        motif_probs = motif_analysis_results["motif_obs_exp_data"][
            "motif_probs"
        ]  # Use the computed motif_probs
        motif_labels = motif_analysis_results["motif_obs_exp_data"][
            "motif_labels"
        ]  # Use the original motif_labels (lists of strings)
        n0 = motif_analysis_results.get(
            "n0", result.observed_cells
        )  # Use stored n0 or fallback to observed_cells

        upset_plot_data = self.motif_analysis_service.prepare_upset_plot_data(
            df_obs_exp, motif_probs, n0, motif_labels
        )

        # Step 25: Save upset plot data
        upset_plot_path = os.path.join(
            analysis_dir, f"{self.config.sample_name}_upsetplot.png"
        )
        self.data_saver.save_upset_plot_data(upset_plot_data, upset_plot_path)

        # Step 26: Generate per-cell projection strength data (BEFORE PlotGenerator)
        self.logger.log_step(
            "Motif analysis", "Generating per-cell projection strength data"
        )
        per_cell_data = (
            self.motif_analysis_service.generate_per_cell_projection_strength_data(
                result.normalized_matrix, result.columns
            )
        )

        # Step 27: Save motif analysis files using DataSaver
        self.logger.log_step("Data Saving", "Saving motif analysis files")
        analysis_dir = os.path.join(self.config.out_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        # Step 25: Create visualizations
        self.logger.log_step("Visualization", "Creating all plots and visualizations")
        # Inject services into PlotGenerator
        self.plot_generator.statistical_analyzer = self.statistical_analyzer
        self.plot_generator.clustering_analyzer = self.clustering_analyzer
        self.plot_generator.matrix_processor = self.matrix_processor
        self.plot_generator.motif_analysis_service = self.motif_analysis_service
        self.plot_generator.create_all_visualizations(
            result.normalized_matrix,
            result.columns,
            {
                "consensus_k": result.consensus_k,
                "region_probabilities": result.region_probabilities,
                "roots": result.roots,
                "binomial_test_results": result.binomial_test_results,
                "simplified_pi": str(
                    simplified_pi
                ),  # Pass simplified pi expression as string
                "calculated_value": result.calculated_value,
                "motif_analysis_results": motif_analysis_results,  # Add motif analysis results
                "upset_plot_data": upset_plot_data,  # Pass pre-generated upset plot data
                "per_cell_data": per_cell_data,  # Pass pre-generated per-cell data
            },
            self.config.out_dir,
            self.config.sample_name,
            result.pe_num,
            result.observed_cells,
        )

        # Step 23: Read motif data from upsetplot file and update result
        motif_over, motif_under = self._read_motif_data_from_upsetplot()
        result.motif_over = motif_over
        result.motif_under = motif_under

        # Step 24: Save final summary with motif data
        self.data_saver.save_summary(result, self.config)

        self.logger.log_success("Processing pipeline completed successfully")
        return result

    def _read_motif_data_from_upsetplot(self) -> Tuple[List[str], List[str]]:
        """
        Read motif over/under data from the upsetplot CSV file.

        Returns:
            Tuple of (motif_over, motif_under) lists
        """
        import os
        import pandas as pd

        upset_file = os.path.join(
            self.config.out_dir, "analysis", f"{self.config.sample_name}_upsetplot.csv"
        )

        motif_over, motif_under = [], []

        try:
            if os.path.exists(upset_file):
                self.logger.log_step(
                    "Motif analysis", f"Reading motif data from {upset_file}"
                )
                df_upset = pd.read_csv(upset_file)

                # Sanitize column names
                df_upset.columns = [col.strip().lower() for col in df_upset.columns]

                # Rename columns to standardized lowercase names
                df_upset.rename(
                    columns={
                        "motifs": "motif",
                        "p-value": "pval",
                        "observed": "observed",
                        "expected": "expected",
                    },
                    inplace=True,
                )

                # Validate required columns exist
                required_cols = {"observed", "expected", "pval", "motif"}
                if not required_cols.issubset(set(df_upset.columns)):
                    raise ValueError(
                        f"Missing expected columns in upsetplot CSV: {required_cols - set(df_upset.columns)}"
                    )

                # Ensure types are correct
                df_upset["observed"] = pd.to_numeric(
                    df_upset["observed"], errors="coerce"
                )
                df_upset["expected"] = pd.to_numeric(
                    df_upset["expected"], errors="coerce"
                )
                df_upset["pval"] = pd.to_numeric(df_upset["pval"], errors="coerce")

                corrected_threshold = 0.05

                motif_over = (
                    df_upset.loc[
                        (df_upset["observed"] > df_upset["expected"])
                        & (df_upset["pval"] < corrected_threshold),
                        "motif",
                    ]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                motif_under = (
                    df_upset.loc[
                        (df_upset["observed"] < df_upset["expected"])
                        & (df_upset["pval"] < corrected_threshold),
                        "motif",
                    ]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                self.logger.log_step(
                    "Motif analysis",
                    f"Found {len(motif_over)} overrepresented and {len(motif_under)} underrepresented motifs",
                )

            else:
                self.logger.log_warning(
                    f"Motif file not found at expected path: {upset_file}"
                )

        except Exception as e:
            self.logger.log_error(e, f"Failed to load upsetplot file {upset_file}")
            motif_over, motif_under = [], []

        return motif_over, motif_under
