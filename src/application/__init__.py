"""
This package contains the application layer for the NBCM processing pipeline.

The application layer is responsible for orchestrating the processing of NBCM data.
"""

from .nbcm_processing_service import NBCMProcessingService

__all__ = ["NBCMProcessingService"]
