"""
Data saving functionality for NBCM processing pipeline.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from collections import Counter
import csv

from src.domain.models import ProcessingConfig, ProcessingResult
from src.infrastructure.logger import Logger


class NBCMDataSaver:
    """Responsible for saving processed data and results"""

    def __init__(self):
        self.logger = Logger()

    def save_matrix(
        self, matrix: np.ndarray, columns: List[str], file_path: str
    ) -> None:
        """
        Save matrix to CSV file.

        Args:
            matrix: Matrix to save
            columns: Column names
            file_path: Output file path
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Create DataFrame and save
            df = pd.DataFrame(matrix)
            df.columns = columns
            df.to_csv(file_path, index=False, float_format="%.8f")

            self.logger.log_save(file_path)

        except Exception as e:
            self.logger.log_error(e, f"Saving matrix to {file_path}")
            raise

    def save_motif_analysis_files(
        self,
        matrix: np.ndarray,
        columns: List[str],
        analysis_dir: str,
        sample_name: str,
        pe_num: float,
        observed_cells: int,
        motif_analysis_service,
    ) -> None:
        """
        Save motif analysis files.

        Args:
            matrix: Normalized matrix
            columns: Column names
            analysis_dir: Analysis directory
            sample_name: Sample name
            pe_num: p_e value
            observed_cells: Number of observed cells
            motif_analysis_service: Motif analysis service
        """
        try:
            # This method is called but the actual motif analysis files are already saved
            # by the motif analysis service during the analysis step
            # So we just log that this step is complete
            self.logger.log_step("Data Saving", "Saving motif analysis files")

        except Exception as e:
            self.logger.log_error(e, "Saving motif analysis files")
            raise

    def save_results(self, results: ProcessingResult, config: ProcessingConfig) -> None:
        """
        Save all processing results to files.

        Args:
            results: Processing results
            config: Processing configuration
        """
        try:
            # Save filtered matrix
            filtered_matrix_file = os.path.join(
                config.out_dir, f"{config.sample_name}_Filtered_Matrix.csv"
            )
            self.save_matrix(
                results.filtered_matrix, results.columns, filtered_matrix_file
            )

            # Save normalized matrix
            normalized_matrix_file = os.path.join(
                config.out_dir, f"{config.sample_name}_Normalized_Matrix.csv"
            )
            self.save_matrix(
                results.normalized_matrix, results.columns, normalized_matrix_file
            )

            # Save UMI total counts
            umi_counts_file = os.path.join(
                config.out_dir, f"{config.sample_name}_UMI_Total_Counts.csv"
            )
            self._save_umi_counts(
                results.umi_total_counts, results.columns, umi_counts_file
            )

            # Save statistical results
            self._save_statistical_results(results, config)

            self.logger.log_success("All results saved successfully")

        except Exception as e:
            self.logger.log_error(e, "Saving results")
            raise

    def _save_umi_counts(
        self, umi_counts: Dict[str, float], columns: List[str], file_path: str
    ) -> None:
        """Save UMI total counts to CSV"""
        try:
            df = pd.DataFrame(
                {
                    "Region": columns,
                    "UMI_Sum": [umi_counts.get(col, 0.0) for col in columns],
                }
            )
            df.to_csv(file_path, index=False)
            self.logger.log_save(file_path)
        except Exception as e:
            self.logger.log_error(e, f"Saving UMI counts to {file_path}")
            raise

    def _save_projections(
        self, projections: Dict[str, int], columns: List[str], file_path: str
    ) -> None:
        """Save projection counts to CSV"""
        try:
            df = pd.DataFrame(
                {
                    "Region": columns,
                    "Cell_Counts": [projections.get(col, 0) for col in columns],
                }
            )
            df.to_csv(file_path, index=False)
            self.logger.log_save(file_path)
        except Exception as e:
            self.logger.log_error(e, f"Saving projections to {file_path}")
            raise

    def _save_statistical_results(
        self, results: ProcessingResult, config: ProcessingConfig
    ) -> None:
        """Save statistical analysis results"""
        try:
            # Save roots
            roots_file = os.path.join(config.out_dir, f"{config.sample_name}_Roots.csv")
            pd.DataFrame({"Roots": results.roots}).to_csv(roots_file, index=False)
            self.logger.log_save(roots_file)

            # Save region probabilities
            probs_file = os.path.join(
                config.out_dir,
                f"{config.sample_name}_Region-specific_Probabilities.csv",
            )
            pd.DataFrame(
                {
                    "Region-specific Probabilities": list(
                        results.region_probabilities.values()
                    )
                }
            ).to_csv(probs_file, index=False)
            self.logger.log_save(probs_file)

            # Save binomial test results
            binomial_file = os.path.join(
                config.out_dir, f"{config.sample_name}_Motif_Binomial_Results.csv"
            )
            binomial_df = pd.DataFrame(
                [
                    {
                        "Motif Size": result[0],
                        "Expected Probability": result[1],
                        "P-Value": result[2],
                    }
                    for result in results.binomial_test_results
                ]
            )
            binomial_df.to_csv(binomial_file, index=False)
            self.logger.log_save(binomial_file)

            # Save binomial test results (alternative format)
            binomial_alt_file = os.path.join(
                config.out_dir, f"{config.sample_name}_Binomial_Test_Results.csv"
            )
            binomial_alt_df = pd.DataFrame(
                [
                    {
                        "Motif Size": result[0],
                        "Expected Probability": result[1],
                        "P-Value": result[2],
                    }
                    for result in results.binomial_test_results
                ]
            )
            binomial_alt_df.to_csv(binomial_alt_file, index=False)
            self.logger.log_save(binomial_alt_file)

            # Save simplified pi expression
            pi_file = os.path.join(
                config.out_dir, f"{config.sample_name}_Simplified_Pi.csv"
            )
            try:
                # Use plain text representation like the original script
                pi_text = str(results.pi_expression)
                pd.DataFrame({"Simplified Pi": [pi_text]}).to_csv(pi_file, index=False)
                self.logger.log_save(pi_file)
            except Exception as e:
                self.logger.log_warning(f"Could not save pi expression: {e}")

            # Save calculated value
            calculated_file = os.path.join(
                config.out_dir, f"{config.sample_name}_Calculated_Value.csv"
            )
            pd.DataFrame({"Calculated Value": [results.calculated_value]}).to_csv(
                calculated_file, index=False
            )
            self.logger.log_save(calculated_file)

            # Save standard deviation
            std_file = os.path.join(
                config.out_dir, f"{config.sample_name}_Standard_Deviation.csv"
            )
            pd.DataFrame({"Standard Deviation": [results.standard_deviation]}).to_csv(
                std_file, index=False
            )
            self.logger.log_save(std_file)

        except Exception as e:
            self.logger.log_error(e, "Saving statistical results")
            raise

    def save_visualizations(
        self, results: ProcessingResult, config: ProcessingConfig
    ) -> None:
        """
        Save visualization data (not the plots themselves, but data for plotting).

        Args:
            results: Processing results
            config: Processing configuration
        """
        try:
            # Create analysis directory
            analysis_dir = os.path.join(config.out_dir, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)

            # Save motif analysis data
            motif_file = os.path.join(
                analysis_dir, f"{config.sample_name}_motif_obs_exp.csv"
            )
            self._save_motif_data(results, motif_file)

            # Save upset plot data
            upset_file = os.path.join(
                analysis_dir, f"{config.sample_name}_upsetplot.csv"
            )
            self._save_upset_data(results, upset_file)

            # Save additional analysis files
            self._save_additional_analysis_files(results, config)

            self.logger.log_success("Visualization data saved")

        except Exception as e:
            self.logger.log_error(e, "Saving visualization data")
            raise

    def _save_motif_counts(
        self, matrix: np.ndarray, columns: List[str], file_path: str
    ) -> None:
        """Save motif counts data"""
        try:
            # Calculate motif sizes
            motif_sizes = [np.sum(row > 0) for row in matrix]
            size_counts = Counter(motif_sizes)

            df = pd.DataFrame(
                [
                    {"Motif_Size": size, "Count": count}
                    for size, count in sorted(size_counts.items())
                ]
            )
            df.to_csv(file_path, index=False)
            self.logger.log_save(file_path)
        except Exception as e:
            self.logger.log_error(e, f"Saving motif counts to {file_path}")
            raise

    def _save_pie_chart_data(
        self, matrix: np.ndarray, columns: List[str], file_path: str
    ) -> None:
        """Save pie chart data"""
        try:
            # Calculate motif sizes
            motif_sizes = [np.sum(row > 0) for row in matrix]
            size_counts = Counter(motif_sizes)

            # Create labels like "1 target", "2 targets", etc.
            data = []
            for size, count in sorted(size_counts.items()):
                if size == 1:
                    label = "1 target"
                else:
                    label = f"{size} targets"
                data.append({"": label, "# Cells": count})

            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            self.logger.log_save(file_path)
        except Exception as e:
            self.logger.log_error(e, f"Saving pie chart data to {file_path}")
            raise

    def _save_counts_txt(
        self, matrix: np.ndarray, columns: List[str], file_path: str
    ) -> None:
        """Save counts as text file"""
        try:
            with open(file_path, "w") as f:
                f.write(f"Total cells: {matrix.shape[0]}\n")
                f.write(f"Total regions: {matrix.shape[1]}\n")
                f.write(f"Regions: {', '.join(columns)}\n")

                # Calculate motif sizes
                motif_sizes = [np.sum(row > 0) for row in matrix]
                size_counts = Counter(motif_sizes)

                f.write("\nMotif size distribution:\n")
                for size, count in sorted(size_counts.items()):
                    f.write(f"{size} targets: {count} cells\n")

            self.logger.log_save(file_path)
        except Exception as e:
            self.logger.log_error(e, f"Saving counts text to {file_path}")
            raise

    def _save_motif_obs_exp_filtered(
        self, matrix: np.ndarray, motif_probs: Dict[int, float], file_path: str
    ) -> None:
        """Save motif observed vs expected (filtered)"""
        try:
            # Calculate observed motif sizes
            motif_sizes = [np.sum(row > 0) for row in matrix]
            size_counts = Counter(motif_sizes)

            df = pd.DataFrame(
                [
                    {
                        "Motif_Size": size,
                        "Observed_Count": count,
                        "Expected_Probability": motif_probs.get(size, 0.0),
                        "Expected_Count": count * motif_probs.get(size, 0.0),
                    }
                    for size, count in sorted(size_counts.items())
                ]
            )
            df.to_csv(file_path, index=False)
            self.logger.log_save(file_path)
        except Exception as e:
            self.logger.log_error(e, f"Saving motif obs/exp filtered to {file_path}")
            raise

    def _save_additional_analysis_files(
        self, results: ProcessingResult, config: ProcessingConfig
    ) -> None:
        """Save additional analysis files"""
        try:
            analysis_dir = os.path.join(config.out_dir, "analysis")

            # Save motif counts
            motif_counts_file = os.path.join(
                analysis_dir, f"{config.sample_name}_motif_counts.csv"
            )
            self._save_motif_counts(
                results.normalized_matrix, results.columns, motif_counts_file
            )

            # Save pie chart data
            pie_data_file = os.path.join(
                analysis_dir, f"{config.sample_name}_pie_chart_data.csv"
            )
            self._save_pie_chart_data(
                results.normalized_matrix, results.columns, pie_data_file
            )

            # Save counts text file
            counts_txt_file = os.path.join(
                analysis_dir, f"{config.sample_name}_counts.txt"
            )
            self._save_counts_txt(
                results.normalized_matrix, results.columns, counts_txt_file
            )

            # Save motif obs exp filtered (commented out - needs motif_probs which is no longer in ProcessingResult)
            # motif_filtered_file = os.path.join(
            #     analysis_dir, f"{config.sample_name}_motif_obs_exp_filtered.csv"
            # )
            # self._save_motif_obs_exp_filtered(
            #     results.normalized_matrix, results.motif_probs, motif_filtered_file
            # )

        except Exception as e:
            self.logger.log_error(e, "Saving additional analysis files")
            raise

    def save_motif_analysis_files(
        self,
        matrix: np.ndarray,
        columns: List[str],
        analysis_dir: str,
        sample_name: str,
        pe_num: float,
        n0: float,
        motif_analysis_service,
    ) -> None:
        """Save motif analysis files using MotifAnalysisService"""
        try:
            # Save motif obs exp data
            motif_obs_exp_file = os.path.join(
                analysis_dir, f"{sample_name}_motif_obs_exp.csv"
            )
            self._save_motif_obs_exp(
                matrix, columns, motif_obs_exp_file, pe_num, n0, motif_analysis_service
            )

            # Save filtered motif obs exp data
            motif_filtered_file = os.path.join(
                analysis_dir, f"{sample_name}_motif_obs_exp_filtered.csv"
            )
            self._save_motif_obs_exp_filtered(
                matrix, columns, motif_filtered_file, pe_num, n0, motif_analysis_service
            )

            # Save upset plot data
            upset_csv_file = os.path.join(analysis_dir, f"{sample_name}_upsetplot.csv")
            self._save_upset_csv(
                matrix, columns, upset_csv_file, pe_num, n0, motif_analysis_service
            )

        except Exception as e:
            self.logger.log_error(e, "Saving motif analysis files")
            raise

    def _save_motif_obs_exp(
        self,
        matrix: np.ndarray,
        columns: List[str],
        file_path: str,
        pe_num: float,
        n0: float,
        motif_analysis_service,
    ) -> None:
        """Save motif observed vs expected data"""
        try:
            df = pd.DataFrame(matrix, columns=columns)
            motif_data = motif_analysis_service._get_motif_obs_exp_data(df, pe_num, n0)
            motif_data["df_obs_exp"].to_csv(file_path, index=False)
            self.logger.log_save(file_path)
        except Exception as e:
            self.logger.log_error(e, f"Saving motif obs exp to {file_path}")
            raise

    def _save_motif_obs_exp_filtered(
        self,
        matrix: np.ndarray,
        columns: List[str],
        file_path: str,
        pe_num: float,
        n0: float,
        motif_analysis_service,
    ) -> None:
        """Save filtered motif observed vs expected data"""
        try:
            df = pd.DataFrame(matrix, columns=columns)
            motif_data = motif_analysis_service._get_motif_obs_exp_data(df, pe_num, n0)
            df_obs_exp = motif_data["df_obs_exp"]
            df_obs_exp = df_obs_exp[df_obs_exp["Motif"] != ""]
            df_obs_exp.to_csv(file_path, index=False)
            self.logger.log_save(file_path)
        except Exception as e:
            self.logger.log_error(e, f"Saving filtered motif obs exp to {file_path}")
            raise

    def _save_upset_csv(
        self,
        matrix: np.ndarray,
        columns: List[str],
        file_path: str,
        pe_num: float,
        n0: float,
        motif_analysis_service,
    ) -> None:
        """Save upset plot CSV data"""
        try:
            df = pd.DataFrame(matrix, columns=columns)
            # Use the same function that generates the visualization data
            upset_data = motif_analysis_service.prepare_upset_plot_data(df, pe_num, n0)
            upset_data["dfdata"].to_csv(file_path, index=False)
            self.logger.log_save(file_path)
        except Exception as e:
            self.logger.log_error(e, f"Saving upset CSV to {file_path}")
            raise

    def _save_motif_data(self, results: ProcessingResult, file_path: str) -> None:
        """Save motif analysis data"""
        try:
            # This would be implemented based on the motif analysis logic
            # For now, create a placeholder
            df = pd.DataFrame(
                {"Motif": ["placeholder"], "Observed": [0], "Expected": [0]}
            )
            df.to_csv(file_path, index=False)
            self.logger.log_save(file_path)
        except Exception as e:
            self.logger.log_error(e, f"Saving motif data to {file_path}")
            raise

    def _save_upset_data(self, results: ProcessingResult, file_path: str) -> None:
        """Save upset plot data"""
        try:
            # This would be implemented based on the upset plot logic
            # For now, create a placeholder
            df = pd.DataFrame(
                {
                    "Motifs": ["placeholder"],
                    "Observed": [0],
                    "Expected": [0],
                    "P_Value": [1.0],
                    "Degree": [1],
                    "Group": [1],
                }
            )
            df.to_csv(file_path, index=False)
            self.logger.log_save(file_path)
        except Exception as e:
            self.logger.log_error(e, f"Saving upset data to {file_path}")
            raise

    def save_summary(self, results: ProcessingResult, config: ProcessingConfig) -> None:
        """
        Save a comprehensive summary of all results.

        Args:
            results: Processing results
            config: Processing configuration
        """
        try:
            summary_file = os.path.join(config.out_dir, "projection_summary.csv")

            # Create summary data with exact column names from original script
            summary_data = {
                "Sample": [config.sample_name],
                "injection min": [config.injection_umi_min],
                "target:inj ratio": [config.min_body_to_target_ratio],
                "at least 1 target minimum": [config.min_target_count],
                "user umi min": [config.target_umi_min],
                "force_user_threshold": [config.force_user_threshold],
                "threshold used": [results.final_umi_threshold],
                "Labels": [",".join(results.columns)],
                "TotalProjections": [results.total_projections],
                "ObservedCells": [results.observed_cells],
                "N0": [results.N0_value],
                "p_e": [results.pe_num],
                "k_consensus": [results.consensus_k],
                "Entropy": [self._calculate_entropy(results.region_probabilities)],
                "MotifOverrepresented": [
                    " ".join(results.motif_over) if results.motif_over else ""
                ],
                "MotifUnderrepresented": [
                    " ".join(results.motif_under) if results.motif_under else ""
                ],
            }

            # Reordered block: group by metric type (ProjCount, UMISum, MeanUMI) - same as original script
            for region in results.columns:
                summary_data[f"ProjCount_{region}"] = [
                    results.projections.get(region, 0)
                ]

            for region in results.columns:
                summary_data[f"UMISum_{region}"] = [
                    results.umi_total_counts.get(region, 0.0)
                ]

            for region in results.columns:
                summary_data[f"MeanUMI_{region}"] = [
                    np.mean(results.normalized_matrix[:, results.columns.index(region)])
                ]

            # Save to CSV with append mode (same as original script)
            write_header = not os.path.exists(summary_file)
            df = pd.DataFrame(summary_data)
            df.to_csv(
                summary_file,
                mode="a",
                header=write_header,
                index=False,
                quoting=csv.QUOTE_ALL,
            )
            self.logger.log_save(summary_file)

        except Exception as e:
            self.logger.log_error(e, "Saving summary")
            raise

    def _calculate_entropy(self, probabilities: Dict[str, float]) -> float:
        """Calculate normalized entropy of probability distribution"""
        try:
            from scipy.stats import entropy

            values = list(probabilities.values())
            if sum(values) == 0:
                return 0.0
            # Normalize
            values = np.array(values) / sum(values)
            return entropy(values) / np.log(len(values)) if len(values) > 1 else 0.0
        except ImportError:
            # Fallback calculation without scipy
            values = list(probabilities.values())
            if sum(values) == 0:
                return 0.0
            values = np.array(values) / sum(values)
            # Manual entropy calculation
            entropy_val = -sum(p * np.log(p) for p in values if p > 0)
            return entropy_val / np.log(len(values)) if len(values) > 1 else 0.0

    def save_motif_analysis_files(
        self,
        motif_analysis_results: Dict[str, Any],
        analysis_dir: str,
        sample_name: str,
    ) -> None:
        """
        Save motif analysis files.

        Args:
            motif_analysis_results: Results from motif analysis
            analysis_dir: Directory to save analysis files
            sample_name: Sample name for file naming
        """
        try:
            # Save motif counts
            motif_counts_file = os.path.join(
                analysis_dir, f"{sample_name}_motif_counts.csv"
            )
            motif_counts_df = motif_analysis_results["motif_counts_df"]
            motif_counts_df.to_csv(motif_counts_file, index=True)
            self.logger.log_save(motif_counts_file)

            # Save pie chart data
            pie_chart_file = os.path.join(
                analysis_dir, f"{sample_name}_pie_chart_data.csv"
            )
            pie_chart_df = motif_analysis_results["pie_chart_df"]
            pie_chart_df.to_csv(pie_chart_file)
            self.logger.log_save(pie_chart_file)

            # Save counts text file
            counts_text_file = os.path.join(analysis_dir, f"{sample_name}_counts.txt")
            detailed_analysis = motif_analysis_results["detailed_motif_analysis"]
            self._save_counts_text(detailed_analysis, counts_text_file)
            self.logger.log_save(counts_text_file)

            # Save motif obs exp files
            motif_obs_exp_data = motif_analysis_results["motif_obs_exp_data"]

            # Save motif_obs_exp.csv
            motif_obs_exp_file = os.path.join(
                analysis_dir, f"{sample_name}_motif_obs_exp.csv"
            )
            motif_obs_exp_data["df_obs_exp"].to_csv(motif_obs_exp_file, index=True)
            self.logger.log_save(motif_obs_exp_file)

            # Save motif_obs_exp_filtered.csv (filtered to exclude empty motifs)
            motif_obs_exp_filtered_file = os.path.join(
                analysis_dir, f"{sample_name}_motif_obs_exp_filtered.csv"
            )
            # Filter out empty motifs (same as original script)
            filtered_df = motif_obs_exp_data["df_obs_exp"][
                motif_obs_exp_data["df_obs_exp"]["Motif"] != ""
            ]
            filtered_df.to_csv(motif_obs_exp_filtered_file, index=False)
            self.logger.log_save(motif_obs_exp_filtered_file)

        except Exception as e:
            self.logger.log_error(e, "Saving motif analysis files")
            raise

    def _save_counts_text(self, detailed_analysis: List[List], file_path: str) -> None:
        """Save detailed motif analysis to text file"""
        try:
            with open(file_path, "w") as f:
                for item in detailed_analysis:
                    f.write("%s\n" % item)
        except Exception as e:
            self.logger.log_error(e, f"Saving counts text to {file_path}")
            raise

    def save_motif_raw_data_files(
        self, motif_data_files: Dict[str, pd.DataFrame], output_dir: str
    ) -> None:
        """
        Save motif raw data files.

        Args:
            motif_data_files: Dictionary mapping filename to DataFrame
            output_dir: Output directory
        """
        try:
            os.makedirs(output_dir, exist_ok=True)

            for filename, df in motif_data_files.items():
                file_path = os.path.join(output_dir, filename)
                df.to_csv(file_path, float_format="%.8f")
                self.logger.log_save(file_path)

        except Exception as e:
            self.logger.log_error(e, "Saving motif raw data files")
            raise

    def save_upset_plot_data(
        self, upset_data: Dict[str, Any], output_path: str
    ) -> None:
        """
        Save upset plot data.

        Args:
            upset_data: Dictionary containing upset plot data
            output_path: Output file path (CSV will be saved with .csv extension)
        """
        try:
            # Save the unfiltered dfraw DataFrame (like original script) which includes "Degree" column
            dfraw = upset_data.get("dfraw")
            if dfraw is not None:
                csv_path = output_path.replace(".png", ".csv")
                dfraw.to_csv(csv_path, index=False)
                self.logger.log_save(csv_path)

        except Exception as e:
            self.logger.log_error(e, "Saving upset plot data")
            raise
