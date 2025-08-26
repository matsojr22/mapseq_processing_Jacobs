"""
Centralized logging for NBCM processing pipeline.
"""

import logging
from typing import Optional
from pathlib import Path


class Logger:
    """Centralized logging for the NBCM processing pipeline"""
    
    def __init__(self, log_file: Optional[str] = None, level: int = logging.INFO):
        """Initialize logger with optional file output"""
        self.logger = logging.getLogger('nbcm_processing')
        self.logger.setLevel(level)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def log_step(self, step: str, details: str) -> None:
        """Log a processing step with details"""
        self.logger.info(f"🔍 {step}: {details}")
    
    def log_error(self, error: Exception, context: str) -> None:
        """Log an error with context"""
        self.logger.error(f"❌ Error in {context}: {str(error)}", exc_info=True)
    
    def log_warning(self, warning: str) -> None:
        """Log a warning message"""
        self.logger.warning(f"⚠️ {warning}")
    
    def log_success(self, message: str) -> None:
        """Log a success message"""
        self.logger.info(f"✅ {message}")
    
    def log_save(self, file_path: str) -> None:
        """Log a file save operation"""
        self.logger.info(f"💾 Saved to: {file_path}")
    
    def log_matrix_shape(self, matrix_name: str, shape: tuple) -> None:
        """Log matrix shape information"""
        self.logger.info(f"📊 {matrix_name} shape: {shape}")
    
    def log_threshold(self, threshold_name: str, value: float) -> None:
        """Log threshold information"""
        self.logger.info(f"🎯 {threshold_name}: {value:.4f}")
    
    def log_statistics(self, stat_name: str, value: float) -> None:
        """Log statistical values"""
        self.logger.info(f"📈 {stat_name}: {value:.6f}")
