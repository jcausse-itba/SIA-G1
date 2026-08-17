import os
from pathlib import Path
from typing import List, Optional
from PIL import Image
from .board import SokobanBoard, Tile
from .utils import Coordinate, Direction

# --- Constants & Sprite Configuration ---
SPRITE_DIR = Path(__file__).parent / "assets"

SPRITE_PATHS = {
    'ground':        SPRITE_DIR / 'ground.png',
    'wall':          SPRITE_DIR / 'wall.png',
    'crate':         SPRITE_DIR / 'crate.png',
    'crate_on_goal': SPRITE_DIR / 'crate_on_goal.png',
    'player':        SPRITE_DIR / 'player.png',
    'goal':          SPRITE_DIR / 'goal.png',
}

# Configure playback speeds (in milliseconds)
FRAME_DURATION_MS = 150       # Speed for regular moves
FINAL_FRAME_PAUSE_MS = 1500   # Pause duration for the final state (1.5s)


class SokobanVisualizer:
    def __init__(self):
        self.sprites = {}
        try:
            for key, path in SPRITE_PATHS.items():
                img = Image.open(path).convert('RGBA')
                self.sprites[key] = img

            sizes = set(img.size for img in self.sprites.values())
            if len(sizes) > 1:
                raise ValueError(f"All sprites must have identical dimensions. Found: {sizes}")
            self.tile_size = sizes.pop()[0]

        except FileNotFoundError as e:
            print(f"Error loading sprites: {e}")
            raise
        except Exception as e:
            print(f"Failed to initialize sprites: {e}")
            raise

    def render_board(self, board: SokobanBoard) -> Image.Image:
        """Creates a single Image frame of the current board state using layered sprites."""
        grid = board.grid
        width_px = grid.cols * self.tile_size
        height_px = grid.rows * self.tile_size
        
        frame = Image.new('RGBA', (width_px, height_px))

        for r in range(grid.rows):
            for c in range(grid.cols):
                coord = Coordinate(r, c)
                tile_type = grid[coord]
                is_goal = (tile_type == Tile.GOAL)
                pos_px = (c * self.tile_size, r * self.tile_size)

                # 1. Base ground layer
                frame.paste(self.sprites['ground'], pos_px, self.sprites['ground'])

                # 2. Static environment objects
                if tile_type == Tile.WALL:
                    frame.paste(self.sprites['wall'], pos_px, self.sprites['wall'])
                elif is_goal:
                    frame.paste(self.sprites['goal'], pos_px, self.sprites['goal'])

                # 3. Dynamic entities
                if coord == board.player_position:
                    frame.paste(self.sprites['player'], pos_px, self.sprites['player'])
                elif coord in board.box_positions:
                    if is_goal:
                        frame.paste(self.sprites['crate_on_goal'], pos_px, self.sprites['crate_on_goal'])
                    else:
                        frame.paste(self.sprites['crate'], pos_px, self.sprites['crate'])

        return frame

    def create_solution_gif(self, initial_board: SokobanBoard, solution_moves: List[Direction], output_filename: str):
        """Generates a GIF visualizing the solution path with a end-pause on the final state."""
        frames = []
        current_board = initial_board

        print("Generating frames...")
        frames.append(self.render_board(current_board))

        for i, move_dir in enumerate(solution_moves):
            next_board = current_board.move(move_dir)
            if next_board is None:
                print(f"Warning: Move {i+1} ({move_dir}) is invalid in the given sequence. Stopping.")
                break
            
            current_board = next_board
            frames.append(self.render_board(current_board))

        if not frames:
            print("No frames generated. Cannot create GIF.")
            return

        # Construct frame durations: standard speed for all steps, extended pause for the final step
        durations = [FRAME_DURATION_MS] * (len(frames) - 1) + [FINAL_FRAME_PAUSE_MS]

        print(f"Saving GIF: {output_filename} ({len(frames)} frames)...")
        frames[0].save(
            output_filename,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=durations,  # Pass the list of frame delays
            loop=0
        )
        print("Done.")