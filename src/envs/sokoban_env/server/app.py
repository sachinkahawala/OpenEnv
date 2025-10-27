# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Sokoban Environment.

This module creates an HTTP server that exposes the SokobanEnvironment
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.sokoban_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.sokoban_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.sokoban_env.server.app
"""

import logging
from pathlib import Path
import os

# Setup logging to file
log_dir = Path(__file__).resolve().parents[4] / "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = log_dir / "sokoban_server.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Keep logging to console as well
    ]
)
logger = logging.getLogger(__name__)

from core.env_server.http_server import create_app
from .sokoban_environment import SokobanEnvironment

# Create the environment instance
env = SokobanEnvironment()

# Create the app with web interface and README integration
app = create_app(env, SokobanAction, SokobanObservation, env_name="sokoban_env")

@app.on_event("startup")
async def startup_event():
    logger.info("Sokoban server starting up.")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Sokoban server shutting down.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
