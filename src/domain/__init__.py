"""
This package contains the domain layer for the NBCM processing pipeline.

The domain layer is responsible for the business logic of the pipeline.
"""

from .models import ProcessingConfig, ProcessingResult

__all__ = ["ProcessingConfig", "ProcessingResult"]
