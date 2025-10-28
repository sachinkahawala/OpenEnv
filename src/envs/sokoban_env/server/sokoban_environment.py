# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sokoban Environment Implementation.

A classic puzzle game where the player pushes boxes onto goal positions.
The player can move in four directions and push boxes (but not pull them).
"""

import logging
import random
import numpy as np
from typing import List, Tuple
from uuid import uuid4

from core.env_server.interfaces import Environment
from core.env_server.types import State

from ..models import SokobanAction, SokobanObservation
from . import utils as level_gen


# Cell type constants
EMPTY = 0
WALL = 1
BOX = 2
GOAL = 3
PLAYER = 4
BOX_ON_GOAL = 5
PLAYER_ON_GOAL = 6

logger = logging.getLogger(__name__)


class SokobanEnvironment(Environment):
    """
    Sokoban puzzle game environment.

    The goal is to push all boxes onto goal positions. The player can move
    in four directions. If there's a box in the direction of movement and
    an empty space behind it, the box will be pushed.

    Rewards:
        - +10 for placing a box on a goal
        - -10 for removing a box from a goal
        - +100 for solving the puzzle (all boxes on goals)
        - -0.1 for each move (to encourage efficiency)

    Example:
        >>> env = SokobanEnvironment()
        >>> obs = env.reset()
        >>> print(f"Board size: {obs.board_shape}")
        >>> print(f"Number of boxes: {obs.num_boxes}")
        >>>
        >>> obs = env.step(SokobanAction(direction="up"))
        >>> print(f"Boxes on goals: {obs.boxes_on_goals}/{obs.num_boxes}")
        >>> print(f"Reward: {obs.reward}")
    """

    def __init__(self, board_size: int = 8, num_boxes: int = 3, max_steps: int = 200):
        """
        Initialize the Sokoban environment.

        Args:
            board_size: Size of the square board (default: 8)
            num_boxes: Number of boxes to place (default: 3)
            max_steps: Maximum steps before episode ends (default: 200)
        """
        self.board_size = board_size
        self.num_boxes = num_boxes
        self.max_steps = max_steps
        
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._board: List[List[int]] = []
        self._player_pos: Tuple[int, int] = (0, 0)
        self._goal_positions: List[Tuple[int, int]] = []
        self._moves_count = 0
        self._pushes_count = 0
        self._previous_boxes_on_goals = 0

        logger.info(f"SokobanEnvironment initialized with board_size={board_size}, num_boxes={num_boxes}, max_steps={max_steps}")

    def reset(self, seed: int | None = None) -> SokobanObservation:
        """
        Reset the environment and generate a new puzzle.

        Args:
            seed: Optional random seed for reproducible level generation

        Returns:
            SokobanObservation with the initial board state
        """
        if seed is not None:
            random.seed(seed)
        
        self._state = State(episode_id=str(uuid4()), step_count=0)
        logger.info(f"Environment reset. New episode ID: {self._state.episode_id}")
        self._moves_count = 0
        self._pushes_count = 0
        
        # Generate a new random level
        self._generate_level()
        
        self._previous_boxes_on_goals = self._count_boxes_on_goals()
        
        return self._get_observation()

    def step(self, action: SokobanAction) -> SokobanObservation:  # type: ignore[override]
        """
        Execute a step in the environment by moving the player.

        Args:
            action: SokobanAction containing the direction to move

        Returns:
            SokobanObservation with the updated board state
        """
        self._state.step_count += 1
        self._moves_count += 1
        
        # Get direction delta
        direction_map = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        
        dr, dc = direction_map[action.direction]
        new_r, new_c = self._player_pos[0] + dr, self._player_pos[1] + dc
        
        reward = -0.1  # Small penalty for each move
        
        # Check if new position is valid
        if self._is_valid_position(new_r, new_c):
            cell = self._board[new_r][new_c]
            
            if cell in [EMPTY, GOAL]:
                # Simple move
                self._move_player(new_r, new_c)
            elif cell in [BOX, BOX_ON_GOAL]:
                # Try to push box
                box_new_r, box_new_c = new_r + dr, new_c + dc
                if self._is_valid_position(box_new_r, box_new_c):
                    box_dest_cell = self._board[box_new_r][box_new_c]
                    if box_dest_cell in [EMPTY, GOAL]:
                        # Push successful
                        self._push_box(new_r, new_c, box_new_r, box_new_c)
                        self._move_player(new_r, new_c)
                        self._pushes_count += 1
                        
                        # Calculate reward based on boxes on goals change
                        current_boxes_on_goals = self._count_boxes_on_goals()
                        if current_boxes_on_goals > self._previous_boxes_on_goals:
                            reward += 10  # Placed a box on goal
                        elif current_boxes_on_goals < self._previous_boxes_on_goals:
                            reward -= 10  # Removed a box from goal
                        self._previous_boxes_on_goals = current_boxes_on_goals
        
        observation = self._get_observation()
        
        # Check if puzzle is solved
        if observation.is_solved:
            reward += 100  # Big reward for solving
            observation.done = True
            logger.info(f"Episode {self._state.episode_id} solved! Final reward: {reward}")
        elif self._state.step_count >= self.max_steps:
            observation.done = True
            logger.warning(f"Episode {self._state.episode_id} ended due to max steps reached.")
        
        observation.reward = reward
        logger.debug(f"Step {self._state.step_count}: Action={action.direction}, Reward={reward}, Done={observation.done}")
        return observation

    def _generate_level(self) -> None:
        """
        Generate a solvable Sokoban level using reverse-playing algorithm.
        
        Uses the level generator from utils.py which:
        1. Creates a room topology with random walk
        2. Places boxes on goals and player
        3. Uses reverse-playing (pulling boxes) to create the initial state
        4. Guarantees solvability by construction
        """
        try:
            # Generate level using the proven reverse-playing algorithm
            logger.info("Generating level with advanced generator.")
            room_structure, room_state, box_mapping = level_gen.generate_sokoban_level(
                dim=(self.board_size, self.board_size),
                p_change_directions=0.35,
                num_steps=max(15, self.board_size * 2),
                num_boxes=self.num_boxes,
                tries=5
            )
            
            # Convert from numpy format to our internal format
            self._convert_from_numpy_format(room_structure, room_state)
            
        except Exception as e:
            # Fallback to simple generation if the advanced generator fails
            logger.error(f"Advanced generator failed: {e}, using fallback.")
            self._generate_level_simple()
    
    def _convert_from_numpy_format(self, room_structure: np.ndarray, room_state: np.ndarray) -> None:
        """
        Convert from the numpy-based generator format to our internal format.
        
        Generator format:
            0 = wall, 1 = empty, 2 = goal, 3 = box, 4 = box_on_goal, 5 = player
        
        Our format:
            0 = empty, 1 = wall, 2 = box, 3 = goal, 4 = player, 5 = box_on_goal, 6 = player_on_goal
        """
        # Mapping from generator format to our format
        gen_to_internal = {
            0: WALL,          # wall -> wall
            1: EMPTY,         # empty -> empty
            2: GOAL,          # goal -> goal
            3: BOX,           # box -> box
            4: BOX_ON_GOAL,   # box_on_goal -> box_on_goal
            5: PLAYER,        # player -> player
        }
        
        # Initialize board
        self._board = [[EMPTY for _ in range(self.board_size)] for _ in range(self.board_size)]
        self._goal_positions = []
        
        # First pass: identify goal positions from structure
        for r in range(self.board_size):
            for c in range(self.board_size):
                struct_cell = int(room_structure[r, c])
                if struct_cell == 2:  # goal in structure
                    self._goal_positions.append((r, c))
        
        # Second pass: convert state and properly mark goals
        for r in range(self.board_size):
            for c in range(self.board_size):
                gen_cell = int(room_state[r, c])
                
                # Map the cell value
                if gen_cell in gen_to_internal:
                    self._board[r][c] = gen_to_internal[gen_cell]
                else:
                    self._board[r][c] = EMPTY
                
                # Track player position
                if gen_cell == 5:  # player
                    self._player_pos = (r, c)
                
                # If this is a goal position and cell is empty, mark it as GOAL
                if (r, c) in self._goal_positions:
                    if self._board[r][c] == EMPTY:
                        self._board[r][c] = GOAL
                    elif self._board[r][c] == PLAYER:
                        # Player is on goal, but keep as PLAYER for now
                        # (game logic will handle this)
                        pass

    def _generate_level_simple(self) -> None:
        """Fallback simple level generator."""
        # Initialize empty board
        self._board = [[EMPTY for _ in range(self.board_size)] 
                       for _ in range(self.board_size)]
        
        # Add walls around the border
        for i in range(self.board_size):
            self._board[0][i] = WALL
            self._board[self.board_size - 1][i] = WALL
            self._board[i][0] = WALL
            self._board[i][self.board_size - 1] = WALL
        
        # Add some random internal walls
        num_walls = self.board_size // 2
        for _ in range(num_walls):
            r = random.randint(2, self.board_size - 3)
            c = random.randint(2, self.board_size - 3)
            self._board[r][c] = WALL
        
        # Place player in a random empty position
        while True:
            r = random.randint(1, self.board_size - 2)
            c = random.randint(1, self.board_size - 2)
            if self._board[r][c] == EMPTY:
                self._player_pos = (r, c)
                self._board[r][c] = PLAYER
                break
        
        # Place goals
        self._goal_positions = []
        for _ in range(self.num_boxes):
            while True:
                r = random.randint(1, self.board_size - 2)
                c = random.randint(1, self.board_size - 2)
                if self._board[r][c] == EMPTY:
                    self._goal_positions.append((r, c))
                    self._board[r][c] = GOAL
                    break
        
        # Place boxes (not on goals initially)
        boxes_placed = 0
        while boxes_placed < self.num_boxes:
            r = random.randint(1, self.board_size - 2)
            c = random.randint(1, self.board_size - 2)
            if self._board[r][c] == EMPTY:
                self._board[r][c] = BOX
                boxes_placed += 1

    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is within bounds and not a wall."""
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return self._board[row][col] != WALL

    def _move_player(self, new_r: int, new_c: int) -> None:
        """Move the player to a new position."""
        old_r, old_c = self._player_pos
        
        # Update old position
        if (old_r, old_c) in self._goal_positions:
            self._board[old_r][old_c] = GOAL
        else:
            self._board[old_r][old_c] = EMPTY
        
        # Update new position
        if (new_r, new_c) in self._goal_positions:
            self._board[new_r][new_c] = PLAYER_ON_GOAL
        else:
            self._board[new_r][new_c] = PLAYER
        
        self._player_pos = (new_r, new_c)

    def _push_box(self, box_r: int, box_c: int, new_r: int, new_c: int) -> None:
        """Push a box from one position to another."""
        # Update old box position
        if (box_r, box_c) in self._goal_positions:
            self._board[box_r][box_c] = GOAL
        else:
            self._board[box_r][box_c] = EMPTY
        
        # Update new box position
        if (new_r, new_c) in self._goal_positions:
            self._board[new_r][new_c] = BOX_ON_GOAL
        else:
            self._board[new_r][new_c] = BOX

    def _count_boxes_on_goals(self) -> int:
        """Count how many boxes are currently on goal positions."""
        count = 0
        for row in self._board:
            for cell in row:
                if cell == BOX_ON_GOAL:
                    count += 1
        return count

    def _get_observation(self) -> SokobanObservation:
        """Create an observation from the current board state."""
        # Flatten the board
        board_flat = [cell for row in self._board for cell in row]
        
        boxes_on_goals = self._count_boxes_on_goals()
        is_solved = boxes_on_goals == self.num_boxes
        
        return SokobanObservation(
            board=board_flat,
            board_shape=[self.board_size, self.board_size],
            num_boxes=self.num_boxes,
            boxes_on_goals=boxes_on_goals,
            player_position=list(self._player_pos),
            moves_count=self._moves_count,
            pushes_count=self._pushes_count,
            is_solved=is_solved,
            done=False,
            reward=0.0,
            metadata={
                "step": self._state.step_count,
                "max_steps": self.max_steps,
            },
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
