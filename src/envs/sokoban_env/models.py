# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Sokoban Environment.

Sokoban is a classic puzzle game where the player pushes boxes to goal locations.
The player can move in four directions and push boxes (but not pull them).
"""

from dataclasses import dataclass
from typing import List, Literal

from core.env_server.types import Action, Observation


@dataclass(kw_only=True)
class SokobanAction(Action):
    """
    Action for the Sokoban environment.
    
    Attributes:
        direction: The direction to move ("up", "down", "left", "right")
    """

    direction: Literal["up", "down", "left", "right"]


@dataclass(kw_only=True)
class SokobanObservation(Observation):
    """
    Observation from the Sokoban environment.
    
    Attributes:
        board: Flattened representation of the game board.
                Each cell is encoded as: 
                0 = empty floor
                1 = wall
                2 = box
                3 = goal
                4 = player
                5 = box on goal
                6 = player on goal
        board_shape: Shape of the board (height, width)
        num_boxes: Total number of boxes in the puzzle
        boxes_on_goals: Number of boxes currently on goal positions
        player_position: (row, col) position of the player
        moves_count: Number of moves taken so far
        pushes_count: Number of box pushes performed
        is_solved: Whether all boxes are on goals
    """

    board: List[int]
    board_shape: List[int]
    num_boxes: int
    boxes_on_goals: int
    player_position: List[int]
    moves_count: int = 0
    pushes_count: int = 0
    is_solved: bool = False
