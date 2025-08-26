"""
Command line argument parsing and validation for NBCM processing pipeline.
"""

import argparse
import os
from typing import List, Union
from pathlib import Path

from src.domain.models import ProcessingConfig
from src.infrastructure.logger import Logger


class ArgumentParser:
    """Command line argument parsing and validation"""
    
    def __init__(self):
        self.logger = Logger()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser"""
        parser = argparse.ArgumentParser(description="Process NBCM data")
        
        # Required arguments
        parser.add_argument(
            "-o", "--out_dir", 
            type=str, 
            required=True, 
            help="Output directory for saving results"
        )
        parser.add_argument(
            "-s", "--sample_name", 
            type=str, 
            required=True, 
            help="Sample name"
        )
        parser.add_argument(
            "-d", "--data_file", 
            type=str, 
            required=True, 
            help="Path to the input nbcm.csv file"
        )
        
        # Optional arguments with defaults
        parser.add_argument(
            "-a", "--alpha", 
            type=float, 
            default=0.05, 
            help="Significance threshold for Bonferroni correction (default: 0.05)"
        )
        parser.add_argument(
            "-i", "--injection_umi_min", 
            type=float, 
            default=1, 
            help="Sets a threshold for minimum 'inj' UMI values. Rows where 'inj' is below this value will be removed. Default: 1."
        )
        parser.add_argument(
            "-t", "--min_target_count", 
            type=float, 
            default=10,
            help="Minimum UMI count required in at least one target area. Rows not meeting this are excluded."
        )
        parser.add_argument(
            "-r", "--min_body_to_target_ratio", 
            type=float, 
            default=10,
            help="Minimum fold-difference between 'inj' value and the highest target count. Rows not meeting this are excluded."
        )
        parser.add_argument(
            "-u", "--target_umi_min", 
            type=float, 
            default=2, 
            help="Sets a threshold filter for target area UMI counts where smaller values will be set to zero. Typically for noise reduction of single UMI values in targets. (default: 2)"
        )
        parser.add_argument(
            "-l", "--labels",
            type=str,
            help="Comma-separated column labels (e.g., 'target1,target2,target3,target-neg-bio'). These need to match your NBCM columns, and you MUST use the exact label 'neg' in any negative control column and 'inj' in any injection column"
        )
        parser.add_argument(
            "-A", "--special_area_1", 
            type=str, 
            required=False, 
            help="One of your favorite target areas"
        )
        parser.add_argument(
            "-B", "--special_area_2", 
            type=str, 
            required=False, 
            help="Another of your favorite target areas to compare to the first"
        )
        parser.add_argument(
            "-f", "--apply_outlier_filtering", 
            action="store_true", 
            help="Enable outlier filtering (Step 7) using mean + 2*std deviation."
        )
        parser.add_argument(
            "--force_user_threshold",
            action="store_true",
            help="If set, override all automatic thresholding and use the user-defined target_umi_min."
        )
        
        return parser
    
    def parse_arguments(self) -> ProcessingConfig:
        """Parse command line arguments and return ProcessingConfig"""
        args = self.parser.parse_args()
        
        # Parse labels
        labels = self._parse_labels(args.labels)
        
        # Create ProcessingConfig
        config = ProcessingConfig(
            out_dir=args.out_dir,
            sample_name=args.sample_name,
            data_file=args.data_file,
            alpha=args.alpha,
            injection_umi_min=args.injection_umi_min,
            min_target_count=args.min_target_count,
            min_body_to_target_ratio=args.min_body_to_target_ratio,
            target_umi_min=args.target_umi_min,
            labels=labels,
            special_area_1=args.special_area_1,
            special_area_2=args.special_area_2,
            apply_outlier_filtering=args.apply_outlier_filtering,
            force_user_threshold=args.force_user_threshold
        )
        
        # Validate configuration
        if not self.validate_config(config):
            raise ValueError("Invalid configuration")
        
        return config
    
    def _parse_labels(self, labels_input: Union[str, List[str], None]) -> List[str]:
        """Parse labels from string or list input"""
        if labels_input is None:
            return []
        
        if isinstance(labels_input, str):
            # Strip quotes and split by comma
            labels_input = labels_input.strip('"').strip("'")
            labels = [label.strip().strip('"').strip("'") for label in labels_input.split(",")]
        elif isinstance(labels_input, list):
            labels = [label.strip().strip('"').strip("'") for label in labels_input]
        else:
            labels = []
        
        return labels
    
    def validate_config(self, config: ProcessingConfig) -> bool:
        """Validate the processing configuration"""
        try:
            # Check if output directory can be created
            os.makedirs(config.out_dir, exist_ok=True)
            
            # Check if data file exists
            if not os.path.exists(config.data_file):
                self.logger.log_error(
                    FileNotFoundError(f"Data file not found: {config.data_file}"), 
                    "Configuration validation"
                )
                return False
            
            # Validate numeric parameters
            if config.alpha <= 0 or config.alpha > 1:
                self.logger.log_warning(f"Alpha value {config.alpha} is outside expected range (0, 1]")
            
            if config.injection_umi_min < 0:
                self.logger.log_warning(f"Injection UMI minimum {config.injection_umi_min} is negative")
            
            if config.min_target_count < 0:
                self.logger.log_warning(f"Minimum target count {config.min_target_count} is negative")
            
            if config.min_body_to_target_ratio < 0:
                self.logger.log_warning(f"Body to target ratio {config.min_body_to_target_ratio} is negative")
            
            if config.target_umi_min < 0:
                self.logger.log_warning(f"Target UMI minimum {config.target_umi_min} is negative")
            
            # Validate labels if provided
            if config.labels:
                required_labels = ['neg', 'inj']
                missing_labels = [label for label in required_labels if not any(label in l.lower() for l in config.labels)]
                if missing_labels:
                    self.logger.log_warning(f"Missing recommended labels: {missing_labels}")
            
            self.logger.log_success("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.log_error(e, "Configuration validation")
            return False
