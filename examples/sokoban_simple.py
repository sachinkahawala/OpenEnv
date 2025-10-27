"""
Sokoban Environment Simple Example

This script demonstrates basic usage of the Sokoban environment.
It shows how to connect to the environment, reset it, and take actions.

Usage:
    python examples/sokoban_simple.py
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from envs.sokoban_env import SokobanAction, SokobanEnv


def print_board(observation):
    """Print a visual representation of the Sokoban board."""
    # Reshape the flat board into 2D
    height, width = observation.board_shape
    board = []
    for i in range(height):
        row = observation.board[i * width:(i + 1) * width]
        board.append(row)
    
    # Symbol mapping for visualization
    symbols = {
        0: '·',  # Empty floor
        1: '█',  # Wall
        2: '□',  # Box
        3: '.',  # Goal
        4: '@',  # Player
        5: '▣',  # Box on goal
        6: '+',  # Player on goal
    }
    
    print("\nCurrent Board:")
    print("─" * (width * 2))
    for row in board:
        print(' '.join(symbols[cell] for cell in row))
    print("─" * (width * 2))


def main():
    print("Sokoban Environment Example")
    print("=" * 50)
    
    # Option 1: Connect to existing server
    # sokoban_env = SokobanEnv(base_url="http://localhost:8000")
    
    # Option 2: Use Docker (recommended)
    print("\nStarting Sokoban environment from Docker...")
    sokoban_env = SokobanEnv.from_docker_image("sokoban-env:latest")
    
    try:
        # Reset the environment
        print("\nResetting environment...")
        result = sokoban_env.reset()
        
        print(f"\nInitial State:")
        print(f"  Board size: {result.observation.board_shape}")
        print(f"  Number of boxes: {result.observation.num_boxes}")
        print(f"  Player position: {result.observation.player_position}")
        print(f"  Boxes on goals: {result.observation.boxes_on_goals}/{result.observation.num_boxes}")
        
        print_board(result.observation)
        
        # Try some moves
        print("\nPlaying the game...")
        print("\nTry moving around to push boxes onto goals!")
        print("Available directions: up, down, left, right")
        
        # Example sequence of moves (you can customize this)
        example_moves = ["right", "right", "down", "left", "up", "up", "right", "down"]
        
        for i, direction in enumerate(example_moves, 1):
            print(f"\n--- Move {i}: {direction.upper()} ---")
            result = sokoban_env.step(SokobanAction(direction=direction))
            
            print(f"Player position: {result.observation.player_position}")
            print(f"Boxes on goals: {result.observation.boxes_on_goals}/{result.observation.num_boxes}")
            print(f"Total moves: {result.observation.moves_count}")
            print(f"Total pushes: {result.observation.pushes_count}")
            print(f"Reward: {result.reward:.2f}")
            
            print_board(result.observation)
            
            if result.observation.is_solved:
                print("\n" + "=" * 50)
                print("CONGRATULATIONS! Puzzle solved!")
                print(f"Completed in {result.observation.moves_count} moves")
                print(f"Pushes: {result.observation.pushes_count}")
                print("=" * 50)
                break
            
            if result.done:
                print("\nMaximum steps reached!")
                print(f"Final score: Boxes on goals: {result.observation.boxes_on_goals}/{result.observation.num_boxes}")
                break
        else:
            print("\n\nExample moves completed!")
            print(f"Current progress: {result.observation.boxes_on_goals}/{result.observation.num_boxes} boxes on goals")
            
            if result.observation.boxes_on_goals < result.observation.num_boxes:
                print("\nTip: Keep experimenting with different move sequences to solve the puzzle!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nCleaning up...")
        sokoban_env.close()
        print("✅ Done!")


if __name__ == "__main__":
    main()
