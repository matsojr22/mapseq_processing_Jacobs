"""
Data loading and initial validation for NBCM processing pipeline.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from pathlib import Path

from src.domain.models import ProcessingConfig
from src.infrastructure.logger import Logger


class NBCMDataLoader:
    """Responsible for loading and initial validation of NBCM data"""
    
    def __init__(self):
        self.logger = Logger()
    
    def load_matrix(self, file_path: str) -> np.ndarray:
        """
        Load NBCM data matrix from file.
        
        Args:
            file_path: Path to the NBCM data file (TSV or CSV)
            
        Returns:
            numpy.ndarray: Loaded matrix with float64 dtype
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        try:
            # Determine file format and load accordingly
            if file_path.endswith('.tsv'):
                matrix = np.genfromtxt(file_path, delimiter='\t')
            elif file_path.endswith('.csv'):
                matrix = np.genfromtxt(file_path, delimiter=',')
            else:
                # Try TSV first, then CSV
                try:
                    matrix = np.genfromtxt(file_path, delimiter='\t')
                except:
                    matrix = np.genfromtxt(file_path, delimiter=',')
            
            # Convert to float64 and ensure it's a 2D array
            matrix = np.array(matrix, dtype=np.float64)
            
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            
            self.logger.log_matrix_shape("Loaded matrix", matrix.shape)
            self.logger.log_success(f"Successfully loaded matrix from {file_path}")
            
            return matrix
            
        except FileNotFoundError:
            self.logger.log_error(FileNotFoundError(f"File not found: {file_path}"), "Data loading")
            raise
        except Exception as e:
            self.logger.log_error(e, f"Error loading matrix from {file_path}")
            raise ValueError(f"Invalid file format or corrupted data: {file_path}")
    
    def validate_labels(self, labels: List[str], matrix_shape: Tuple[int, int]) -> bool:
        """
        Validate that labels match matrix dimensions.
        
        Args:
            labels: List of column labels
            matrix_shape: Shape of the matrix (rows, columns)
            
        Returns:
            bool: True if labels are valid
        """
        if not labels:
            self.logger.log_warning("No labels provided")
            return True
        
        # The original script expects labels to match the matrix columns after removing headers
        # The matrix has headers, so we need to account for that
        expected_columns = matrix_shape[1] - 1  # Subtract 1 for the header row
        actual_columns = len(labels)
        
        if actual_columns != expected_columns:
            self.logger.log_error(
                ValueError(f"Label count ({actual_columns}) doesn't match matrix columns ({expected_columns})"),
                "Label validation"
            )
            return False
        
        # Check for required labels
        required_labels = ['neg', 'inj']
        missing_labels = []
        
        for required in required_labels:
            if not any(required in label.lower() for label in labels):
                missing_labels.append(required)
        
        if missing_labels:
            self.logger.log_warning(f"Missing recommended labels: {missing_labels}")
        
        self.logger.log_success(f"Validated {len(labels)} labels")
        return True
    
    def parse_labels(self, labels_input: Union[str, List[str], None]) -> List[str]:
        """
        Parse labels from various input formats.
        
        Args:
            labels_input: Labels as string, list, or None
            
        Returns:
            List[str]: Parsed labels
        """
        if labels_input is None:
            return []
        
        if isinstance(labels_input, str):
            # Strip quotes and split by comma
            labels_input = labels_input.strip('"').strip("'")
            labels = [label.strip().strip('"').strip("'") for label in labels_input.split(",")]
        elif isinstance(labels_input, list):
            labels = [label.strip().strip('"').strip("'") for label in labels_input]
        else:
            self.logger.log_warning(f"Unexpected labels input type: {type(labels_input)}")
            labels = []
        
        self.logger.log_step("Parsed labels", f"Found {len(labels)} labels")
        return labels
    
    def get_matrix_info(self, matrix: np.ndarray) -> dict:
        """
        Get basic information about the loaded matrix.
        
        Args:
            matrix: Loaded NBCM matrix
            
        Returns:
            dict: Matrix information
        """
        info = {
            'shape': matrix.shape,
            'dtype': matrix.dtype,
            'total_elements': matrix.size,
            'non_zero_elements': np.count_nonzero(matrix),
            'zero_rows': np.sum(np.sum(matrix > 0, axis=1) == 0),
            'min_value': np.min(matrix),
            'max_value': np.max(matrix),
            'mean_value': np.mean(matrix),
            'std_value': np.std(matrix)
        }
        
        self.logger.log_step("Matrix info", f"Shape: {info['shape']}, Non-zero: {info['non_zero_elements']}")
        return info
    
    def load_and_validate(self, config: ProcessingConfig) -> Tuple[np.ndarray, List[str]]:
        """
        Load matrix and validate with configuration.
        
        Args:
            config: Processing configuration
            
        Returns:
            Tuple[np.ndarray, List[str]]: Loaded matrix and validated labels
        """
        # Load matrix
        matrix = self.load_matrix(config.data_file)
        
        # Parse and validate labels
        labels = self.parse_labels(config.labels)
        
        # If no labels provided, generate default labels
        if not labels:
            labels = [f"region_{i}" for i in range(matrix.shape[1])]
            self.logger.log_warning("No labels provided, using default region names")
        
        # Validate labels match matrix
        if not self.validate_labels(labels, matrix.shape):
            raise ValueError("Label validation failed")
        
        # Log matrix information
        matrix_info = self.get_matrix_info(matrix)
        self.logger.log_step("Matrix loaded", f"Shape: {matrix_info['shape']}, Zero rows: {matrix_info['zero_rows']}")
        
        return matrix, labels
