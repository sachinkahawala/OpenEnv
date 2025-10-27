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

import random
from typing import List, Tuple
from uuid import uuid4

from core.env_server.interfaces import Environment
from core.env_server.types import State

from ..models import SokobanAction, SokobanObservation


# Cell type constants
EMPTY = 0
WALL = 1
BOX = 2
GOAL = 3
PLAYER = 4
BOX_ON_GOAL = 5
PLAYER_ON_GOAL = 6


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

    def __init__(self, board_size: int = 8, num_boxes: int = 1, max_steps: int = 200):
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
        self._moves_count = 0
        self._pushes_count = 0
        self._previous_boxes_on_goals = 0
        
        # Generate a new random level
        self._generate_level()
        
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
        elif self._state.step_count >= self.max_steps:
            observation.done = True
        
        observation.reward = reward
        return observation

    def _generate_level(self) -> None:
        """
        Generate a solvable Sokoban level using reverse-playing algorithm.
        
        The algorithm works by:
        1. Starting with all boxes on goals (solved state)
        2. Performing reverse moves (pulling boxes) to create the initial state
        3. This guarantees the level is solvable since we know the solution path
        """
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                self._generate_level_reverse_playing()
                # Verify we have a valid level
                if self._count_boxes() == self.num_boxes:
                    return
            except Exception:
                # If generation fails, try again
                continue
        
        # Fallback to simple generation if reverse-playing fails
        self._generate_level_simple()
    
    def _generate_level_reverse_playing(self) -> None:
        """
        Generate a level using reverse-playing (backward search).
        
        Start from a solved state and work backwards by pulling boxes away from goals.
        This ensures the generated level is solvable.
        """
        # Initialize board with walls
        self._board = [[EMPTY for _ in range(self.board_size)] 
                       for _ in range(self.board_size)]
        
        # Add walls around the border
        for i in range(self.board_size):
            self._board[0][i] = WALL
            self._board[self.board_size - 1][i] = WALL
            self._board[i][0] = WALL
            self._board[i][self.board_size - 1] = WALL
        
        # Add some random internal walls to create interesting layouts
        num_walls = random.randint(self.board_size // 3, self.board_size // 2)
        wall_attempts = 0
        max_wall_attempts = num_walls * 5
        
        while wall_attempts < max_wall_attempts and num_walls > 0:
            r = random.randint(2, self.board_size - 3)
            c = random.randint(2, self.board_size - 3)
            
            # Don't create isolated areas or corners
            if self._board[r][c] == EMPTY:
                self._board[r][c] = WALL
                # Check if this wall doesn't block too much space
                if self._count_reachable_cells(r + 1, c + 1) > self.num_boxes * 3:
                    num_walls -= 1
                else:
                    self._board[r][c] = EMPTY  # Undo
            wall_attempts += 1
        
        # Initialize solved state: place goals and boxes on them
        self._initialize_solved_state()
        
        # Perform reverse moves (pull boxes away from goals)
        num_reverse_moves = random.randint(
            self.num_boxes * 3,  # Minimum complexity
            min(self.num_boxes * 10, self.max_steps // 2)  # Maximum complexity
        )
        
        visited_states = set()
        visited_states.add(self._board_hash())
        
        for _ in range(num_reverse_moves):
            # Get all possible box pulls
            pullable_boxes = self._get_pullable_boxes()
            
            if not pullable_boxes:
                break
            
            # Choose a random pullable box and direction
            box_pos, pull_directions = random.choice(pullable_boxes)
            direction = random.choice(pull_directions)
            
            # Perform the pull
            self._pull_box_reverse(box_pos, direction)
            
            # Track visited states to avoid trivial back-and-forth
            board_hash = self._board_hash()
            if board_hash not in visited_states:
                visited_states.add(board_hash)
        
        # Normalize the board (ensure goals are marked correctly)
        self._normalize_board()
    
    def _generate_level_simple(self) -> None:
        """Fallback simple level generator (original algorithm)."""
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
    
    def _initialize_solved_state(self) -> None:
        """
        Initialize the board in a solved state (all boxes on goals).
        Place player adjacent to the boxes.
        """
        self._goal_positions = []
        
        # Find suitable positions for goals in a cluster
        center_r = self.board_size // 2
        center_c = self.board_size // 2
        
        # Try to place goals in a connected area
        placed = 0
        attempts = 0
        max_attempts = 100
        
        while placed < self.num_boxes and attempts < max_attempts:
            if placed == 0:
                # First goal near center
                r = random.randint(center_r - 1, center_r + 1)
                c = random.randint(center_c - 1, center_c + 1)
            else:
                # Subsequent goals near existing goals
                existing_goal = random.choice(self._goal_positions)
                dr = random.randint(-1, 1)
                dc = random.randint(-1, 1)
                r = existing_goal[0] + dr
                c = existing_goal[1] + dc
            
            # Check if position is valid
            if (1 <= r < self.board_size - 1 and 
                1 <= c < self.board_size - 1 and
                self._board[r][c] == EMPTY and
                (r, c) not in self._goal_positions):
                
                self._goal_positions.append((r, c))
                self._board[r][c] = BOX_ON_GOAL  # Start with boxes on goals
                placed += 1
            
            attempts += 1
        
        # If we couldn't place all goals, fall back to random positions
        while placed < self.num_boxes:
            r = random.randint(1, self.board_size - 2)
            c = random.randint(1, self.board_size - 2)
            if self._board[r][c] == EMPTY and (r, c) not in self._goal_positions:
                self._goal_positions.append((r, c))
                self._board[r][c] = BOX_ON_GOAL
                placed += 1
        
        # Place player adjacent to a goal
        for goal_r, goal_c in self._goal_positions:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                pr = goal_r + dr
                pc = goal_c + dc
                if (1 <= pr < self.board_size - 1 and 
                    1 <= pc < self.board_size - 1 and
                    self._board[pr][pc] == EMPTY):
                    self._player_pos = (pr, pc)
                    self._board[pr][pc] = PLAYER
                    return
        
        # Fallback: place player in any empty cell
        for r in range(1, self.board_size - 1):
            for c in range(1, self.board_size - 1):
                if self._board[r][c] == EMPTY:
                    self._player_pos = (r, c)
                    self._board[r][c] = PLAYER
                    return
    
    def _get_pullable_boxes(self) -> List[Tuple[Tuple[int, int], List[str]]]:
        """
        Get all boxes that can be pulled by the player.
        
        Returns:
            List of tuples: (box_position, list_of_valid_pull_directions)
        """
        pr, pc = self._player_pos
        pullable = []
        
        # Check all four directions
        directions = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        
        for direction, (dr, dc) in directions.items():
            # Box position (in front of player)
            box_r = pr + dr
            box_c = pc + dc
            
            # Position behind player (where box will be pulled to)
            pull_to_r = pr - dr
            pull_to_c = pc - dc
            
            # Check if we can pull this box
            if (self._is_valid_position(box_r, box_c) and
                self._is_valid_position(pull_to_r, pull_to_c)):
                
                box_cell = self._board[box_r][box_c]
                pull_to_cell = self._board[pull_to_r][pull_to_c]
                
                # Check if there's a box in front and empty space behind
                if (box_cell in [BOX, BOX_ON_GOAL] and 
                    pull_to_cell in [EMPTY, GOAL]):
                    
                    # Find or create entry for this box
                    box_pos = (box_r, box_c)
                    found = False
                    for i, (pos, dirs) in enumerate(pullable):
                        if pos == box_pos:
                            pullable[i] = (pos, dirs + [direction])
                            found = True
                            break
                    
                    if not found:
                        pullable.append((box_pos, [direction]))
        
        return pullable
    
    def _pull_box_reverse(self, box_pos: Tuple[int, int], direction: str) -> None:
        """
        Pull a box in the reverse-playing algorithm.
        
        This simulates the player pulling a box towards themselves,
        which is the reverse of pushing.
        
        Args:
            box_pos: Current position of the box
            direction: Direction to pull the box (relative to player)
        """
        pr, pc = self._player_pos
        box_r, box_c = box_pos
        
        # Direction deltas
        direction_map = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        
        dr, dc = direction_map[direction]
        
        # New player position (where the box currently is)
        new_pr = box_r
        new_pc = box_c
        
        # New box position (behind the current player)
        new_box_r = pr - dr
        new_box_c = pc - dc
        
        # Move the box
        is_box_on_goal = self._board[box_r][box_c] == BOX_ON_GOAL
        if is_box_on_goal:
            self._board[box_r][box_c] = GOAL
        else:
            self._board[box_r][box_c] = EMPTY
        
        # Place box in new position
        if (new_box_r, new_box_c) in self._goal_positions:
            self._board[new_box_r][new_box_c] = BOX_ON_GOAL
        else:
            self._board[new_box_r][new_box_c] = BOX
        
        # Move the player
        self._board[pr][pc] = EMPTY
        
        if (new_pr, new_pc) in self._goal_positions:
            self._board[new_pr][new_pc] = PLAYER_ON_GOAL
        else:
            self._board[new_pr][new_pc] = PLAYER
        
        self._player_pos = (new_pr, new_pc)
    
    def _normalize_board(self) -> None:
        """
        Normalize the board to ensure goals are properly marked.
        After reverse-playing, we need to ensure GOAL cells are visible.
        """
        # First pass: mark all goal positions
        for goal_r, goal_c in self._goal_positions:
            cell = self._board[goal_r][goal_c]
            if cell == EMPTY:
                self._board[goal_r][goal_c] = GOAL
            elif cell == PLAYER:
                self._board[goal_r][goal_c] = PLAYER_ON_GOAL
            elif cell == BOX:
                self._board[goal_r][goal_c] = BOX_ON_GOAL
            # BOX_ON_GOAL and PLAYER_ON_GOAL already correct
    
    def _board_hash(self) -> int:
        """
        Create a hash of the current board state for deduplication.
        
        Returns:
            Hash value representing the current board state
        """
        # Create a tuple of (player_pos, frozenset of box positions)
        box_positions = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self._board[r][c] in [BOX, BOX_ON_GOAL]:
                    box_positions.append((r, c))
        
        return hash((self._player_pos, frozenset(box_positions)))
    
    def _count_reachable_cells(self, start_r: int, start_c: int) -> int:
        """
        Count the number of cells reachable from a starting position.
        Used to avoid creating isolated areas.
        
        Args:
            start_r: Starting row
            start_c: Starting column
            
        Returns:
            Number of reachable empty cells
        """
        if not (0 <= start_r < self.board_size and 0 <= start_c < self.board_size):
            return 0
        
        if self._board[start_r][start_c] == WALL:
            return 0
        
        visited = set()
        queue = [(start_r, start_c)]
        visited.add((start_r, start_c))
        count = 0
        
        while queue:
            r, c = queue.pop(0)
            count += 1
            
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if ((nr, nc) not in visited and
                    0 <= nr < self.board_size and
                    0 <= nc < self.board_size and
                    self._board[nr][nc] != WALL):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return count
    
    def _count_boxes(self) -> int:
        """Count the total number of boxes on the board."""
        count = 0
        for row in self._board:
            for cell in row:
                if cell in [BOX, BOX_ON_GOAL]:
                    count += 1
        return count

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
