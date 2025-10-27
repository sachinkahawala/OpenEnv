# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sokoban Environment HTTP Client.

This module provides the client for connecting to a Sokoban Environment server
over HTTP.
"""

from typing import Dict

from core.client_types import StepResult
from core.env_server.types import State
from core.http_env_client import HTTPEnvClient

from .models import SokobanAction, SokobanObservation


class SokobanEnv(HTTPEnvClient[SokobanAction, SokobanObservation]):
    """
    HTTP client for the Sokoban Environment.

    This client connects to a Sokoban Environment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = SokobanEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(f"Board shape: {result.observation.board_shape}")
        >>> print(f"Number of boxes: {result.observation.num_boxes}")
        >>>
        >>> # Make a move
        >>> result = client.step(SokobanAction(direction="up"))
        >>> print(f"Boxes on goals: {result.observation.boxes_on_goals}")
        >>> print(f"Is solved: {result.observation.is_solved}")
        >>> print(f"Reward: {result.reward}")

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SokobanEnv.from_docker_image("sokoban-env:latest")
        >>> result = client.reset()
        >>> result = client.step(SokobanAction(direction="right"))
    """

    def _step_payload(self, action: SokobanAction) -> Dict:
        """
        Convert SokobanAction to JSON payload for step request.

        Args:
            action: SokobanAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "direction": action.direction,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SokobanObservation]:
        """
        Parse server response into StepResult[SokobanObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with SokobanObservation
        """
        obs_data = payload.get("observation", {})
        observation = SokobanObservation(
            board=obs_data.get("board", []),
            board_shape=obs_data.get("board_shape", []),
            num_boxes=obs_data.get("num_boxes", 0),
            boxes_on_goals=obs_data.get("boxes_on_goals", 0),
            player_position=obs_data.get("player_position", [0, 0]),
            moves_count=obs_data.get("moves_count", 0),
            pushes_count=obs_data.get("pushes_count", 0),
            is_solved=obs_data.get("is_solved", False),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
