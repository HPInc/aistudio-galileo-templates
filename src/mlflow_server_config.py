"""
MLflow server configuration module for AI Studio Galileo Templates.

This module provides functions to configure the MLflow server parameters,
particularly for extending the default gunicorn worker timeout.
"""

import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

def configure_mlflow_server(timeout_seconds=600):
    """
    Configure MLflow server parameters.
    
    This function sets environment variables that control the behavior of
    the MLflow model server when it's deployed, particularly the timeout for
    gunicorn workers.
    
    Args:
        timeout_seconds: Timeout in seconds for the gunicorn workers. Default is 600 (10 minutes).
    
    Returns:
        None
    """
    # Set the GUNICORN_CMD_ARGS environment variable to extend the timeout
    os.environ["GUNICORN_CMD_ARGS"] = f"--timeout={timeout_seconds}"
    logger.info(f"MLflow server timeout set to {timeout_seconds} seconds")
