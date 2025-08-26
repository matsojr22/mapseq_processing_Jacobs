"""
Core domain models for NBCM processing pipeline.
Contains data structures for configuration and results.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class ProcessingConfig:
    """Configuration for NBCM processing pipeline"""

    out_dir: str
    sample_name: str
    data_file: str
    alpha: float
    injection_umi_min: float
    min_target_count: float
    min_body_to_target_ratio: float
    target_umi_min: float
    labels: List[str]
    special_area_1: Optional[str]
    special_area_2: Optional[str]
    apply_outlier_filtering: bool
    force_user_threshold: bool


@dataclass
class ProcessingResult:
    """Result of NBCM processing"""

    # Core data
    filtered_matrix: np.ndarray
    normalized_matrix: np.ndarray
    columns: List[str]

    # Statistical results
    projections: Dict[str, int]
    umi_total_counts: Dict[str, float]
    total_projections: int
    observed_cells: int
    N0_value: float
    pe_num: float
    pi_expression: Any
    region_probabilities: Dict[str, float]
    roots: List[float]
    binomial_test_results: List[Tuple[int, float, float]]
    expected_value: float
    standard_deviation: float
    calculated_value: float
    final_umi_threshold: float

    # Clustering results
    consensus_k: int

    # Motif analysis
    motif_over: List[str]
    motif_under: List[str]
