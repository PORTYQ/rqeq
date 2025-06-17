#!/usr/bin/env python
"""
Main entry point for Crypto Forecast Application
"""
import sys
import subprocess
import logging
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import streamlit
        import pandas
        import numpy
        import tensorflow
        import yfinance
        import ccxt
        import ta
        import shap
        import plotly

        return True
    except ImportError as e:
        logging.error(f"Missing dependency: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    directories = ["cache", "logs", "models"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logging.info(f"Created directory: {directory}")


def main():
    """Main function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)

    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        logger.error("Please install all dependencies: pip install -r requirements.txt")
        sys.exit(1)

    # Create directories
    logger.info("Creating necessary directories...")
    create_directories()

    # Run Streamlit app
    logger.info("Starting Crypto Forecast Application...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "ui.py",
                "--server.headless",
                "true",
                "--server.port",
                "8501",
                "--browser.serverAddress",
                "localhost",
            ]
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Error running application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
