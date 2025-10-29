"""
Sokoban Level Generator using Reverse-Playing Algorithm.

This implementation generates solvable Sokoban levels by:
1. Creating a random room topology with a random walk
2. Placing boxes on goals and a player
3. Reverse-playing (pulling boxes away from goals) to create the initial state
4. Using depth-first search to explore states and find good configurations

Based on the algorithm described in the paper:
"Procedural Generation of Sokoban Levels" by Joshua Taylor and Ian Parberry
"""

import random
import numpy as np
from typing import Dict, Tuple, Set, Optional


# Cell type encoding for generation
WALL = 0
EMPTY = 1
GOAL = 2
BOX = 3
BOX_ON_GOAL = 4
PLAYER = 5

# Action definitions
ACTION_PULL_UP = 0
ACTION_PULL_DOWN = 1
ACTION_PULL_LEFT = 2
ACTION_PULL_RIGHT = 3
ACTION_MOVE_UP = 4
ACTION_MOVE_DOWN = 5
ACTION_MOVE_LEFT = 6
ACTION_MOVE_RIGHT = 7

# Direction mappings
CHANGE_COORDINATES = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1)    # right
}

# Global variables for reverse playing search
explored_states: Set[bytes] = set()
num_boxes: int = 0
best_room_score: float = -1
best_room: Optional[np.ndarray] = None
best_box_mapping: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None


def generate_sokoban_level(
    dim: Tuple[int, int] = (8, 8),
    p_change_directions: float = 0.35,
    num_steps: int = 25,
    num_boxes: int = 3,
    tries: int = 4
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate a solvable Sokoban level using reverse-playing algorithm.
    
    Args:
        dim: Dimensions of the room (height, width)
        p_change_directions: Probability of changing direction during room generation
        num_steps: Number of steps in the random walk for room topology
        num_boxes: Number of boxes to place
        tries: Number of attempts to generate a good level
        
    Returns:
        room_structure: Static structure (walls, empty spaces, goals)
        room_state: Current state (includes movable objects)
        box_mapping: Mapping from goal positions to box positions
    """
    room_state = np.zeros(shape=dim, dtype=int)
    room_structure = np.zeros(shape=dim, dtype=int)
    
    level_valid = False

    # Try multiple times to generate a level with a good score
    for attempt in range(tries):
        # Generate room topology (walls and floors)
        room = room_topology_generation(dim, p_change_directions, num_steps)
        
        # Place boxes (on goals) and player
        room = place_boxes_and_player(room, num_boxes=num_boxes)
        
        # Room structure represents all immovable parts
        room_structure = np.copy(room)
        room_structure[room_structure == PLAYER] = EMPTY
        
        # Room state represents current state with movable objects
        room_state = room.copy()
        room_state[room_state == GOAL] = BOX_ON_GOAL  # Start with all boxes on goals
        
        # Use reverse-playing to generate solvable level
        room_state, score, box_mapping = reverse_playing(room_state, room_structure)

        # Validate that the resulting layout has the expected number of boxes/goals
        boxes_on_goals = int(np.count_nonzero(room_state == BOX_ON_GOAL))
        total_boxes = int(np.count_nonzero(room_state == BOX)) + boxes_on_goals
        total_goals = int(np.count_nonzero(room_structure == GOAL))

        if total_goals != num_boxes:
            # Regenerate if we somehow lost goal markers (should be rare)
            continue

        if total_boxes != num_boxes:
            # Regenerate if any box vanished during reverse playing
            continue

        if boxes_on_goals == 0:
            level_valid = True
            break

        # If the configuration is still solved, try another attempt so we do
        # not start from a trivially solved state.
        continue
        
        # Note: Don't convert boxes back to BOX_ON_GOAL - the reverse playing
        # already placed them correctly (away from goals)
        
        if score > 0:
            break
    
    if score == 0:
        # If we couldn't generate a good level, return at least a valid one
        print(f"Warning: Generated level with score 0 after {tries} attempts")

    if not level_valid:
        raise RuntimeError("Failed to generate Sokoban level with boxes off goals")
    
    return room_structure, room_state, box_mapping


def room_topology_generation(
    dim: Tuple[int, int] = (10, 10),
    p_change_directions: float = 0.35,
    num_steps: int = 15
) -> np.ndarray:
    """
    Generate room topology using a random walk with masks.
    
    Creates a connected area of floor tiles by performing a random walk
    and applying various shaped masks at each step.
    
    Args:
        dim: Room dimensions (height, width)
        p_change_directions: Probability of changing direction
        num_steps: Number of steps in the random walk
        
    Returns:
        2D array with walls (0) and empty spaces (1)
    """
    dim_x, dim_y = dim
    
    # Masks define the shape of floor tiles placed at each step
    # The center position corresponds to the current walk position
    masks = [
        [[0, 0, 0], [1, 1, 1], [0, 0, 0]],  # horizontal line
        [[0, 1, 0], [0, 1, 0], [0, 1, 0]],  # vertical line
        [[0, 0, 0], [1, 1, 0], [0, 1, 0]],  # L-shape
        [[0, 0, 0], [1, 1, 0], [1, 1, 0]],  # square
        [[0, 0, 0], [0, 1, 1], [0, 1, 0]],  # reverse L-shape
    ]
    
    # Possible movement directions
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    direction = random.choice(directions)
    
    # Starting position (avoid edges)
    position = np.array([
        random.randint(1, dim_x - 2),
        random.randint(1, dim_y - 2)
    ])
    
    level = np.zeros(dim, dtype=int)
    
    # Perform random walk
    for step in range(num_steps):
        # Randomly change direction
        if random.random() < p_change_directions:
            direction = random.choice(directions)
        
        # Update position
        position = position + np.array(direction)
        position[0] = max(min(position[0], dim_x - 2), 1)
        position[1] = max(min(position[1], dim_y - 2), 1)
        
        # Apply random mask at current position
        mask = random.choice(masks)
        mask_start = position - 1
        
        # Add mask to level (use addition to handle overlaps)
        level[mask_start[0]:mask_start[0] + 3, mask_start[1]:mask_start[1] + 3] += mask
    
    # Convert to binary (any positive value becomes floor)
    level[level > 0] = EMPTY
    
    # Ensure borders are walls
    level[:, [0, dim_y - 1]] = WALL
    level[[0, dim_x - 1], :] = WALL
    
    return level


def place_boxes_and_player(room: np.ndarray, num_boxes: int) -> np.ndarray:
    """
    Place boxes (on goals) and player in random floor positions.
    
    Args:
        room: Room topology
        num_boxes: Number of boxes to place
        
    Returns:
        Room with player and boxes placed
    """
    # Find all empty floor positions
    possible_positions = np.where(room == EMPTY)
    num_possible_positions = possible_positions[0].shape[0]
    
    if num_possible_positions <= num_boxes + 1:  # Need space for boxes + player
        raise RuntimeError(
            f'Not enough free spots ({num_possible_positions}) to place '
            f'1 player and {num_boxes} boxes.'
        )
    
    # Place player
    ind = np.random.randint(num_possible_positions)
    player_position = (possible_positions[0][ind], possible_positions[1][ind])
    room[player_position] = PLAYER
    
    # Place boxes (initially on goals - these are goal positions)
    for _ in range(num_boxes):
        possible_positions = np.where(room == EMPTY)
        num_possible_positions = possible_positions[0].shape[0]
        
        ind = np.random.randint(num_possible_positions)
        box_position = (possible_positions[0][ind], possible_positions[1][ind])
        room[box_position] = GOAL  # Mark as goal position
    
    return room


def reverse_playing(
    room_state: np.ndarray,
    room_structure: np.ndarray,
    search_depth: int = 300
) -> Tuple[np.ndarray, float, Dict]:
    """
    Play Sokoban in reverse by pulling boxes away from goals.
    
    This ensures the generated level is solvable because we're working
    backwards from a solved state.
    
    Args:
        room_state: Current room state with boxes on goals
        room_structure: Static room structure
        search_depth: Maximum search depth (TTL)
        
    Returns:
        best_room: Best room state found
        best_score: Score of the best room
        best_box_mapping: Mapping of goal positions to box positions
    """
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping
    
    # Initialize box mapping (maps goal positions to current box positions)
    box_mapping = {}
    box_locations = np.where(room_structure == GOAL)
    num_boxes = len(box_locations[0])
    
    for i in range(num_boxes):
        box = (box_locations[0][i], box_locations[1][i])
        box_mapping[box] = box  # Initially, boxes are at their goals
    
    # Reset global state
    explored_states = set()
    best_room_score = -1
    best_room = None  # Don't initialize with the solved state
    best_box_mapping = box_mapping.copy()
    
    # Perform depth-first search with reverse moves
    depth_first_search(
        room_state=room_state,
        room_structure=room_structure,
        box_mapping=box_mapping,
        box_swaps=0,
        last_pull=(-1, -1),
        ttl=search_depth
    )
    
    # If no valid room found, return the initial state (but this should rarely happen)
    if best_room is None:
        best_room = room_state.copy()
        best_room_score = 0
    
    return best_room, best_room_score, best_box_mapping


def depth_first_search(
    room_state: np.ndarray,
    room_structure: np.ndarray,
    box_mapping: Dict[Tuple[int, int], Tuple[int, int]],
    box_swaps: int = 0,
    last_pull: Tuple[int, int] = (-1, -1),
    ttl: int = 300
) -> None:
    """
    Recursive depth-first search through possible reverse moves.
    
    Explores different ways to pull boxes away from their goals,
    tracking the best configuration found.
    
    Args:
        room_state: Current room state
        room_structure: Static room structure
        box_mapping: Current box mapping
        box_swaps: Number of box moves made
        last_pull: Last box that was pulled
        ttl: Time to live (remaining search depth)
    """
    global explored_states, num_boxes, best_room_score, best_room, best_box_mapping
    
    ttl -= 1
    if ttl <= 0 or len(explored_states) >= 30000:  # Limit state exploration
        return
    
    # Create state hash for deduplication
    state_hash = room_state.tobytes()
    
    # Only explore new states
    if state_hash in explored_states:
        return
    
    explored_states.add(state_hash)
    
    # Calculate score for this state
    # Score is based on number of moves and displacement of boxes from goals
    current_num_boxes = np.where(room_state == BOX)[0].shape[0]
    current_num_boxes += np.where(room_state == BOX_ON_GOAL)[0].shape[0]
    
    room_score = 0
    if current_num_boxes == num_boxes:
        displacement = box_displacement_score(box_mapping)
        # Only score positively if boxes have been moved away from goals
        # Require minimum displacement to avoid accepting the initial solved state
        if displacement > 0:
            room_score = box_swaps * displacement
    
    # Update best room if this is better
    if room_score > best_room_score:
        best_room = room_state.copy()
        best_room_score = room_score
        best_box_mapping = box_mapping.copy()
    
    # Try all possible actions (pulls and moves)
    for action in range(8):  # 4 pulls + 4 moves
        # Copy state for this branch
        room_state_next = room_state.copy()
        box_mapping_next = box_mapping.copy()
        
        # Perform reverse move
        room_state_next, box_mapping_next, last_pull_next = reverse_move(
            room_state_next,
            room_structure,
            box_mapping_next,
            last_pull,
            action
        )
        
        # Count box swaps
        box_swaps_next = box_swaps
        if last_pull_next != last_pull and last_pull_next != (-1, -1):
            box_swaps_next += 1
        
        # Recurse
        depth_first_search(
            room_state_next,
            room_structure,
            box_mapping_next,
            box_swaps_next,
            last_pull_next,
            ttl
        )


def reverse_move(
    room_state: np.ndarray,
    room_structure: np.ndarray,
    box_mapping: Dict[Tuple[int, int], Tuple[int, int]],
    last_pull: Tuple[int, int],
    action: int
) -> Tuple[np.ndarray, Dict, Tuple[int, int]]:
    """
    Perform a reverse move (player pulls box or just moves).
    
    Args:
        room_state: Current room state
        room_structure: Static room structure
        box_mapping: Current box mapping
        last_pull: Last box that was pulled
        action: Action to perform (0-7)
        
    Returns:
        Updated room_state, box_mapping, and last_pull
    """
    # Find player position
    player_positions = np.where(room_state == PLAYER)
    if len(player_positions[0]) == 0:
        return room_state, box_mapping, last_pull
    
    player_position = np.array([player_positions[0][0], player_positions[1][0]])
    
    # Get direction change
    change = np.array(CHANGE_COORDINATES[action % 4])
    next_position = player_position + change
    
    # Check bounds
    if (next_position[0] < 0 or next_position[0] >= room_state.shape[0] or
        next_position[1] < 0 or next_position[1] >= room_state.shape[1]):
        return room_state, box_mapping, last_pull
    
    # Check if next position is walkable (empty floor or goal)
    if room_state[next_position[0], next_position[1]] in [EMPTY, GOAL]:
        # Move player
        room_state[player_position[0], player_position[1]] = \
            room_structure[player_position[0], player_position[1]]
        room_state[next_position[0], next_position[1]] = PLAYER
        
        # If this is a pull action (0-3), try to pull a box
        if action < 4:
            # Box would be behind the player (opposite direction)
            box_offset = -change
            possible_box_location = player_position + box_offset
            
            # Check bounds
            if (possible_box_location[0] >= 0 and 
                possible_box_location[0] < room_state.shape[0] and
                possible_box_location[1] >= 0 and 
                possible_box_location[1] < room_state.shape[1]):
                
                # Check if there's a box to pull
                if room_state[possible_box_location[0], possible_box_location[1]] in [BOX, BOX_ON_GOAL]:
                    # Pull the box to player's old position
                    room_state[player_position[0], player_position[1]] = BOX
                    room_state[possible_box_location[0], possible_box_location[1]] = \
                        room_structure[possible_box_location[0], possible_box_location[1]]
                    
                    # Update box mapping
                    box_loc_tuple = (possible_box_location[0], possible_box_location[1])
                    new_box_loc = (player_position[0], player_position[1])
                    
                    for goal_pos in box_mapping.keys():
                        if box_mapping[goal_pos] == box_loc_tuple:
                            box_mapping[goal_pos] = new_box_loc
                            last_pull = goal_pos
                            break
    
    return room_state, box_mapping, last_pull


def box_displacement_score(box_mapping: Dict[Tuple[int, int], Tuple[int, int]]) -> float:
    """
    Calculate total Manhattan distance between boxes and their goal positions.
    
    Higher scores indicate boxes are further from goals, which generally
    means more interesting puzzles.
    
    Args:
        box_mapping: Maps goal positions to current box positions
        
    Returns:
        Total displacement score
    """
    score = 0.0
    
    for goal_pos, box_pos in box_mapping.items():
        box_array = np.array(box_pos)
        goal_array = np.array(goal_pos)
        manhattan_dist = np.sum(np.abs(box_array - goal_array))
        score += manhattan_dist
    
    return score
