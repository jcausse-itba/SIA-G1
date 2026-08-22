from ..board import Board, SokobanBoard
from typing import cast
import numpy as np
from scipy.optimize import linear_sum_assignment

def min_goal_distance_heuristic(board: Board) -> float:
    if not isinstance(board, SokobanBoard):
        return 0.0
        
    total_cost = 0.0
    goals = board.goal_coordinates
    
    for box in board.box_positions:
        if box in goals:
            continue
        
        min_dist = min(box.manhattan_distance(goal) for goal in goals)
        total_cost += min_dist
        
    return float(total_cost)

def matching_with_player_heuristic(board: Board) -> float:
    if not isinstance(board, SokobanBoard):
        return 0.0
        
    boxes = board.box_positions
    goals = board.goal_coordinates
    
    if not boxes or not goals:
        return 0.0

    box_cost = 0.0
    for box in boxes:
        if box not in goals:
            min_dist = min(box.manhattan_distance(goal) for goal in goals)
            box_cost += min_dist

    player_pos = board.player_position
    player_to_box_cost = min(player_pos.manhattan_distance(box) for box in boxes)

    return float(box_cost + player_to_box_cost)

def unique_goal_matching_heuristic(board: Board) -> float:
    if not isinstance(board, SokobanBoard):
        return 0.0
        
    boxes = board.box_positions
    goals = board.goal_coordinates
    
    if not boxes or not goals:
        return 0.0

    unplaced_boxes = [b for b in boxes if b not in goals]
    if not unplaced_boxes:
        return 0.0

    available_goals = set(goals)
    total_cost = 0.0

    # Greedily match each unplaced box to its closest available unique goal.
    # To make it robust, we can sort boxes by their distance to their closest goal first,
    # or just iterate through them.
    for box in unplaced_boxes:
        if not available_goals:
            # If there are more boxes than goals, fallback to nearest remaining goal
            available_goals = set(goals)
            
        closest_goal = min(available_goals, key=lambda g: box.manhattan_distance(g))
        total_cost += box.manhattan_distance(closest_goal)
        available_goals.remove(closest_goal)

    return float(total_cost)


def hungarian_matching_heuristic(board: Board) -> float:
    """Optimal box-to-goal assignment cost via scipy's Hungarian algorithm.
    """

    if not isinstance(board, SokobanBoard):
        return 0.0

    boxes = board.box_positions
    goals = board.goal_coordinates

    if not boxes or not goals:
        return 0.0

    unplaced_boxes = [b for b in boxes if b not in goals]
    if not unplaced_boxes:
        return 0.0

    goal_list = list(goals)
    cost_matrix = np.array([
        [box.manhattan_distance(goal) for goal in goal_list]
        for box in unplaced_boxes
    ])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return float(cost_matrix[row_ind, col_ind].sum())