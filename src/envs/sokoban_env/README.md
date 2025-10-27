# Sokoban Environment

A classic puzzle game environment where the player pushes boxes onto goal positions. Sokoban is a challenging puzzle that requires planning and strategy to solve efficiently.

## Quick Start

The simplest way to use the Sokoban environment is through the `SokobanEnv` class:

```python
from envs.sokoban_env import SokobanAction, SokobanEnv

try:
    # Create environment from Docker image
    sokoban_env = SokobanEnv.from_docker_image("sokoban-env:latest")

    # Reset to get a new puzzle
    result = sokoban_env.reset()
    print(f"Board size: {result.observation.board_shape}")
    print(f"Number of boxes: {result.observation.num_boxes}")
    print(f"Boxes on goals: {result.observation.boxes_on_goals}")

    # Play the game
    moves = ["up", "right", "down", "left"]
    
    for direction in moves:
        result = sokoban_env.step(SokobanAction(direction=direction))
        print(f"\nMove: {direction}")
        print(f"  → Player position: {result.observation.player_position}")
        print(f"  → Boxes on goals: {result.observation.boxes_on_goals}/{result.observation.num_boxes}")
        print(f"  → Reward: {result.reward}")
        print(f"  → Solved: {result.observation.is_solved}")
        
        if result.done:
            if result.observation.is_solved:
                print("\n🎉 Puzzle solved!")
            else:
                print("\n⏱️ Max steps reached")
            break

finally:
    # Always clean up
    sokoban_env.close()
```

That's it! The `SokobanEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t sokoban-env:latest -f src/envs/sokoban_env/server/Dockerfile .
```

## Game Rules

### Objective
Push all boxes onto goal positions to solve the puzzle.

### Movement
- The player can move in four directions: up, down, left, right
- The player can push boxes but cannot pull them
- Only one box can be pushed at a time
- A box can only be pushed if there's empty space behind it

### Board Elements
The board is represented as a grid with the following elements:
- **0** = Empty floor
- **1** = Wall
- **2** = Box
- **3** = Goal position
- **4** = Player
- **5** = Box on goal
- **6** = Player on goal

## Environment Details

### Action
**SokobanAction**: Contains the direction to move
- `direction` (str) - One of: "up", "down", "left", "right"

### Observation
**SokobanObservation**: Contains the game state and metadata
- `board` (List[int]) - Flattened board representation (use `board_shape` to reshape)
- `board_shape` (List[int]) - Shape of the board [height, width]
- `num_boxes` (int) - Total number of boxes in the puzzle
- `boxes_on_goals` (int) - Number of boxes currently on goal positions
- `player_position` (List[int]) - [row, col] position of the player
- `moves_count` (int) - Total number of moves taken
- `pushes_count` (int) - Number of box pushes performed
- `is_solved` (bool) - Whether all boxes are on goals
- `reward` (float) - Reward for the last action
- `done` (bool) - Whether the episode is finished
- `metadata` (dict) - Additional information

### Reward Structure
- **+10** for placing a box on a goal
- **-10** for removing a box from a goal
- **+100** for solving the puzzle (all boxes on goals)
- **-0.1** for each move (encourages efficiency)

### Episode Termination
An episode ends when:
1. All boxes are on goals (puzzle solved) ✅
2. Maximum steps reached (default: 200) ⏱️

## Advanced Usage

### Connecting to an Existing Server

If you already have a Sokoban environment server running, you can connect directly:

```python
from envs.sokoban_env import SokobanEnv, SokobanAction

# Connect to existing server
sokoban_env = SokobanEnv(base_url="http://localhost:8000")

# Use as normal
result = sokoban_env.reset()
result = sokoban_env.step(SokobanAction(direction="up"))
```

Note: When connecting to an existing server, `sokoban_env.close()` will NOT stop the server.

### Visualizing the Board

You can reshape and visualize the board from the observation:

```python
import numpy as np

result = sokoban_env.reset()
board = np.array(result.observation.board).reshape(result.observation.board_shape)

# Print the board with symbols
symbols = {
    0: '·',  # Empty
    1: '█',  # Wall
    2: '□',  # Box
    3: '.',  # Goal
    4: '@',  # Player
    5: '▣',  # Box on goal
    6: '+',  # Player on goal
}

for row in board:
    print(''.join(symbols[cell] for cell in row))
```

### Custom Environment Parameters

You can customize the environment when running the server:

```python
# In server/sokoban_environment.py, modify the initialization:
env = SokobanEnvironment(
    board_size=10,    # Larger board
    num_boxes=5,      # More boxes
    max_steps=500     # More time to solve
)
```

## Development & Testing

### Running the Server Locally

Start the server without Docker for development:

```bash
# From project root
cd src
python -m envs.sokoban_env.server.app
```

The server will start at `http://localhost:8000`

### Testing the Environment

Test the environment logic directly:

```python
from envs.sokoban_env.server.sokoban_environment import SokobanEnvironment
from envs.sokoban_env import SokobanAction

env = SokobanEnvironment()
obs = env.reset()

print(f"Initial state:")
print(f"  Board size: {obs.board_shape}")
print(f"  Boxes: {obs.num_boxes}")
print(f"  Player at: {obs.player_position}")

# Try some moves
obs = env.step(SokobanAction(direction="up"))
print(f"\nAfter moving up:")
print(f"  Boxes on goals: {obs.boxes_on_goals}")
print(f"  Reward: {obs.reward}")
```

### API Endpoints

When the server is running, you can access:
- `GET /` - Web interface with game visualization
- `POST /reset` - Reset the environment
- `POST /step` - Take an action
- `GET /state` - Get current state
- `GET /health` - Health check

## Strategies for Solving

1. **Plan Ahead**: Think several moves ahead before pushing a box
2. **Avoid Corners**: Never push a box into a corner unless it's a goal
3. **Avoid Walls**: Don't push boxes against walls unless necessary
4. **Work Backwards**: Sometimes it helps to think from the goal positions backwards
5. **One at a Time**: Focus on getting one box to a goal before moving others

## Example: Simple Solver Loop

```python
from envs.sokoban_env import SokobanEnv, SokobanAction
import random

sokoban_env = SokobanEnv.from_docker_image("sokoban-env:latest")

try:
    result = sokoban_env.reset()
    directions = ["up", "down", "left", "right"]
    
    while not result.done:
        # Simple random strategy (not very effective!)
        direction = random.choice(directions)
        result = sokoban_env.step(SokobanAction(direction=direction))
        
        print(f"Boxes on goals: {result.observation.boxes_on_goals}/{result.observation.num_boxes}")
        
    if result.observation.is_solved:
        print(f"Solved in {result.observation.moves_count} moves!")
    else:
        print("Failed to solve")
        
finally:
    sokoban_env.close()
```

## Technical Details

- **State Management**: Each episode has a unique episode ID
- **Board Generation**: Random level generation with configurable parameters
- **Concurrent Access**: Each client gets its own environment instance
- **Performance**: Lightweight implementation suitable for RL training


## Future Enhancements

Potential improvements for this environment:
- [ ] Pre-defined puzzle levels of varying difficulty
- [ ] Undo functionality
- [ ] Solution validation and optimal path checking
- [ ] Multiple puzzle sizes and configurations
- [ ] Save/load game states
- [ ] Graphical visualization in web interface
- [ ] Performance metrics (solution efficiency)

## References

- Original Sokoban game: https://en.wikipedia.org/wiki/Sokoban
- Classic puzzle collection: http://www.sokobano.de/
