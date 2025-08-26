"""
Core business logic for matrix processing in NBCM pipeline.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

from src.domain.models import ProcessingConfig
from src.infrastructure.logger import Logger


class MatrixProcessor:
    """Core business logic for matrix processing"""

    def __init__(self):
        self.logger = Logger()

    def clean_and_filter(
        self, matrix: np.ndarray, config: ProcessingConfig
    ) -> Tuple[np.ndarray, float, float]:
        """
        Clean and filter the matrix according to NBCM processing rules.

        Args:
            matrix: Raw NBCM matrix
            config: Processing configuration

        Returns:
            Tuple[np.ndarray, float, float]: Filtered matrix, max_neg_value, final_umi_threshold
        """
        self.logger.log_step("Matrix cleaning", "Starting filtering process")

        # Step 1: Remove headers & barcode column
        matrix = matrix[1:, 1:]
        self.logger.log_matrix_shape("After removing headers", matrix.shape)

        # Step 2: Remove rows with all zeros
        matrix = matrix[np.sum(matrix > 0, axis=1) > 0]
        self.logger.log_matrix_shape("After removing zero rows", matrix.shape)

        # Step 2b: Remove rows where no target regions are > min_target_count
        non_neg_inj_cols = [
            i for i, label in enumerate(config.labels) if label not in ["neg", "inj"]
        ]
        if non_neg_inj_cols:
            target_max = np.nanmax(matrix[:, non_neg_inj_cols], axis=1)
            matrix = matrix[target_max >= config.min_target_count]
            self.logger.log_matrix_shape(
                f"After min target count filter ({config.min_target_count})",
                matrix.shape,
            )
        else:
            self.logger.log_warning(
                "No valid target columns found (excluding 'inj' and 'neg')"
            )

        # Step 3: Remove rows where any value >= the corresponding 'inj' column value
        if "inj" in config.labels:
            inj_col_idx = config.labels.index("inj")
            inj_values = matrix[:, inj_col_idx]
            mask = np.all(
                matrix[:, :inj_col_idx] < inj_values[:, None], axis=1
            ) & np.all(matrix[:, inj_col_idx + 1 :] < inj_values[:, None], axis=1)
            matrix = matrix[mask]
            self.logger.log_matrix_shape("After injection site filtering", matrix.shape)
        else:
            self.logger.log_warning("'inj' column not found in sample labels")

        # Step 3b: Remove rows where 'inj' value is below injection_umi_min
        if "inj" in config.labels:
            inj_col_idx = config.labels.index("inj")
            matrix = matrix[matrix[:, inj_col_idx] >= config.injection_umi_min]
            self.logger.log_matrix_shape(
                f"After injection UMI min filter ({config.injection_umi_min})",
                matrix.shape,
            )
        else:
            self.logger.log_warning("'inj' column not found")

        # Step 3c: Remove rows where inj < (max target value * ratio threshold)
        if "inj" in config.labels:
            inj_col_idx = config.labels.index("inj")
            non_neg_inj_cols = [
                i
                for i, label in enumerate(config.labels)
                if label not in ["neg", "inj"]
            ]
            inj_values = matrix[:, inj_col_idx]
            if non_neg_inj_cols:
                max_target_values = np.nanmax(matrix[:, non_neg_inj_cols], axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    valid_mask = inj_values >= (
                        max_target_values * config.min_body_to_target_ratio
                    )
                    valid_mask = np.nan_to_num(valid_mask, nan=False)
                matrix = matrix[valid_mask.astype(bool)]
                self.logger.log_matrix_shape(
                    f"After body-to-target ratio filter ({config.min_body_to_target_ratio})",
                    matrix.shape,
                )
            else:
                self.logger.log_warning("No valid target columns found")
        else:
            self.logger.log_warning("'inj' column not found")

        # Step 4: Extract max value from 'neg' column before removing rows with neg > 0
        neg_columns = [
            i for i, label in enumerate(config.labels) if "neg" in label.lower()
        ]
        if neg_columns:
            neg_values = matrix[:, neg_columns]
            neg_values = neg_values[~np.isnan(neg_values)]
            if neg_values.size > 0:
                max_neg_value = np.nanmax(neg_values)
            else:
                max_neg_value = config.target_umi_min
                self.logger.log_warning("'neg' column contains only NaN values")
            self.logger.log_threshold("Max negative control value", max_neg_value)
        else:
            max_neg_value = config.target_umi_min
            self.logger.log_warning("'neg' column not found")

        # Step 5: Remove rows where any 'neg' column has a nonzero value
        if neg_columns:
            matrix = matrix[np.all(matrix[:, neg_columns] == 0, axis=1)]
        self.logger.log_matrix_shape("After negative control filtering", matrix.shape)

        # Step 6a: Dynamically calculate noise threshold using histogram elbow
        dynamic_threshold = self._calculate_dynamic_threshold(matrix, config)

        # Step 6b: Choose UMI threshold
        if config.force_user_threshold:
            final_umi_threshold = config.target_umi_min
            self.logger.log_threshold("User-forced threshold", final_umi_threshold)
        else:
            final_umi_threshold = max(
                config.target_umi_min, max_neg_value, dynamic_threshold
            )
            self.logger.log_threshold("Final UMI threshold", final_umi_threshold)

        # Step 6c: Apply threshold and remove zero rows
        matrix[matrix < final_umi_threshold] = 0
        num_zero_after_threshold = np.sum(np.sum(matrix > 0, axis=1) == 0)
        self.logger.log_step(
            "Threshold applied", f"New zero rows: {num_zero_after_threshold}"
        )

        matrix = matrix[np.sum(matrix > 0, axis=1) > 0]
        self.logger.log_matrix_shape("After thresholding", matrix.shape)

        # Step 7: Apply optional high-UMI outlier filtering
        if config.apply_outlier_filtering:
            matrix = self._apply_outlier_filtering(matrix, config)

        self.logger.log_success("Matrix cleaning and filtering completed")
        return matrix, max_neg_value, final_umi_threshold

    def _calculate_dynamic_threshold(
        self, matrix: np.ndarray, config: ProcessingConfig
    ) -> float:
        """Calculate dynamic noise threshold using histogram elbow method"""
        non_neg_inj_cols = [
            i for i, label in enumerate(config.labels) if label not in ["neg", "inj"]
        ]
        if non_neg_inj_cols:
            all_target_vals = matrix[:, non_neg_inj_cols].flatten()
            non_zero_target_vals = all_target_vals[all_target_vals > 0]
            if non_zero_target_vals.size > 0:
                log_vals = np.log10(non_zero_target_vals + 1e-5)
                density = gaussian_kde(log_vals)
                xs = np.linspace(log_vals.min(), log_vals.max(), 1000)
                ys = density(xs)
                d2 = np.gradient(np.gradient(ys))
                elbow_idx = np.argmin(d2)
                elbow_log_value = xs[elbow_idx]
                dynamic_threshold = 10**elbow_log_value
                self.logger.log_threshold("Dynamic noise threshold", dynamic_threshold)
                return dynamic_threshold
            else:
                self.logger.log_warning(
                    "No nonzero target values found for threshold estimation"
                )
        else:
            self.logger.log_warning("No target columns found for dynamic thresholding")

        return config.target_umi_min

    def _apply_outlier_filtering(
        self, matrix: np.ndarray, config: ProcessingConfig
    ) -> np.ndarray:
        """Apply high-UMI outlier filtering"""
        non_neg_inj_cols = [
            i for i, label in enumerate(config.labels) if label not in ["neg", "inj"]
        ]
        if non_neg_inj_cols:
            mean_values = np.mean(matrix[:, non_neg_inj_cols], axis=0)
            std_values = np.std(matrix[:, non_neg_inj_cols], axis=0)
            upper_threshold = mean_values + 2 * std_values
            filtered_matrix = []
            for row in matrix:
                if all(
                    row[i] <= upper_threshold[idx]
                    for idx, i in enumerate(non_neg_inj_cols)
                ):
                    filtered_matrix.append(row)
            matrix = np.array(filtered_matrix)
            self.logger.log_matrix_shape("After outlier filtering", matrix.shape)

        return matrix

    def normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalize each row by its maximum value.

        Args:
            matrix: Input matrix

        Returns:
            np.ndarray: Normalized matrix
        """
        if matrix.shape[0] == 0:
            self.logger.log_warning("Normalized matrix is empty")
            return matrix

        normalized = np.apply_along_axis(
            lambda x: x / np.amax(x) if np.amax(x) > 0 else x, axis=1, arr=matrix
        )

        self.logger.log_matrix_shape("Normalized matrix", normalized.shape)
        return normalized

    def remove_neg_inj_columns(
        self, matrix: np.ndarray, labels: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Remove negative control and injection site columns.

        Args:
            matrix: Input matrix
            labels: Column labels

        Returns:
            Tuple[np.ndarray, List[str]]: Filtered matrix and updated labels
        """
        neg_inj_columns = [
            i
            for i, label in enumerate(labels)
            if "neg" in label.lower() or label == "inj"
        ]
        if neg_inj_columns:
            matrix = np.delete(matrix, neg_inj_columns, axis=1)
            self.logger.log_step(
                "Column removal", f"Dropped {len(neg_inj_columns)} neg/inj columns"
            )

        # Update labels
        updated_labels = [
            label for i, label in enumerate(labels) if i not in neg_inj_columns
        ]
        self.logger.log_step("Updated labels", f"Remaining columns: {updated_labels}")

        return matrix, updated_labels

    def remove_zero_rows(self, matrix: np.ndarray) -> np.ndarray:
        """Remove rows that are all zeros after processing"""
        original_shape = matrix.shape
        matrix = matrix[np.sum(matrix > 0, axis=1) > 0]
        removed_rows = original_shape[0] - matrix.shape[0]

        if removed_rows > 0:
            self.logger.log_step(
                "Zero row removal", f"Removed {removed_rows} zero rows"
            )

        return matrix

    def scale_and_normalize_matrix(
        self, matrix: np.ndarray, columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale and normalize matrix for visualization.

        Args:
            matrix: Input matrix
            columns: Column labels

        Returns:
            Tuple[np.ndarray, np.ndarray]: Scaled matrix and scaled numpy array
        """
        # Convert to DataFrame
        df_ = pd.DataFrame(matrix, columns=columns)

        # Scale the data
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_.astype(float)), columns=df_.columns
        )

        # Convert to numpy array
        df_scaled_np = df_scaled.to_numpy(copy=True).astype(float)

        return df_scaled, df_scaled_np

    def prepare_han_style_data(
        self, matrix: np.ndarray, columns: List[str]
    ) -> np.ndarray:
        """
        Prepare data for Han-style clustering visualization.

        Args:
            matrix: Input matrix
            columns: Column labels

        Returns:
            np.ndarray: Prepared data for Han-style clustering
        """
        # Convert to DataFrame
        df_han = pd.DataFrame(matrix, columns=columns)

        # Handle NaN and infinite values
        df_han = df_han.replace([np.inf, -np.inf], np.nan)
        df_han = df_han.fillna(0)

        # Log transform and normalize (EXACT same as original)
        df_han = np.log1p(df_han + 1e-3)
        df_han = df_han / df_han.max(axis=1).values.reshape(-1, 1)

        # Add max projection column
        df_han["max_proj_col"] = df_han.values.argmax(axis=1)

        # Sort by max projection column and drop it
        df_han = (
            df_han.sort_values("max_proj_col")
            .drop(columns="max_proj_col")
            .reset_index(drop=True)
        )

        # Convert to float
        df_han = df_han.astype(float)

        return df_han.values

    def generate_probability_matrix(
        self, matrix: np.ndarray, columns: List[str]
    ) -> np.ndarray:
        """
        Generate probability matrix for blue-yellow heatmap.

        Args:
            matrix: Input matrix
            columns: Column labels

        Returns:
            np.ndarray: Probability matrix
        """
        # Convert to DataFrame
        df = pd.DataFrame(matrix, columns=columns)

        # Calculate probability matrix (EXACT same as original)
        mat = np.zeros((len(columns), len(columns)))
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    mat[i, j] = 1
                else:
                    # Calculate P(B|A) = P(A∩B) / P(A)
                    p_a = (df[col1] > 0).sum()
                    p_ab = ((df[col1] > 0) & (df[col2] > 0)).sum()
                    mat[i, j] = p_ab / p_a if p_a > 0 else 0

        # Set index to match original
        mat = pd.DataFrame(mat, index=columns, columns=columns)
        return mat
