#!/usr/bin/env python3
"""
NBCM Processing Pipeline - Main Entry Point
Clean, modular architecture with separation of concerns.

This script serves as the main entry point with zero business logic.
All processing is delegated to specialized services.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infrastructure.argument_parser import ArgumentParser
from src.application.nbcm_processing_service import NBCMProcessingService
from src.infrastructure.logger import Logger


def main():
    """Main entry point - no business logic"""
    logger = Logger()

    try:
        logger.log_step("Starting", "NBCM Processing Pipeline")

        # Parse and validate arguments
        logger.log_step("Parsing", "Command line arguments")
        parser = ArgumentParser()
        config = parser.parse_arguments()

        # Initialize and run processing service
        logger.log_step("Initializing", "Processing service")
        service = NBCMProcessingService(config)

        logger.log_step("Processing", "NBCM data")
        result = service.process()

        logger.log_success("Processing completed successfully")
        print("✅ Processing completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.log_warning("Processing interrupted by user")
        print("⚠️ Processing interrupted by user")
        return 130

    except Exception as e:
        logger.log_error(e, "Main execution")
        print(f"❌ Processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
