import random
from typing import Dict, Tuple

from adversarial_search_problem import (
    Action,
    State as GameState,
)
from heuristic_adversarial_search_problem import HeuristicAdversarialSearchProblem


def minimax(asp: HeuristicAdversarialSearchProblem[GameState, Action], cutoff_depth=float('inf')) -> Tuple[Action, Dict[str, int]]:
    """
    Implement the minimax algorithm on ASPs, assuming that the given game is
    both 2-player and zero-sum.

    Input:
        asp - a HeuristicAdversarialSearchProblem
        cutoff_depth - the maximum search depth, where 0 is the start state. 
                    Depth 1 is all the states reached after a single action from the start state (1 ply).
                    cutoff_depth will always be greater than 0.
    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))
        a dictionary of statistics for visualization
            states_expanded: stores the number of states expanded during current search
                            A state is expanded when get_available_actions(state) is called.
    """
    best_action = None
    stats = {
        'states_expanded': 0
    }
    _, best_action, stats = minimax_helper(asp, asp.get_start_state(), stats, cutoff_depth)
    return best_action, stats


def minimax_helper(
        asp: HeuristicAdversarialSearchProblem[GameState, Action],
        current_state: GameState,
        stats: Dict[str, int],
        cutoff_depth: float) -> Tuple[int, Action, Dict[str, int]]:
    """
    Implement the minimax algorithm on ASPs, assuming that the given game is
    both 2-player and zero-sum.

    Input:
        asp - a HeuristicAdversarialSearchProblem
        current_state - the current state of the board
        stats: a dictionary containing the current states_expanded statistic
        cutoff_depth - the maximum search depth, where 0 is the start state. 
                    Depth 1 is all the states reached after a single action from the start state (1 ply).
                    cutoff_depth will always be greater than 0.
    Output:
        a float representing the value of the action returned
        an action (an element of asp.get_available_actions(asp.get_start_state()))
        a dictionary of statistics for visualization
            states_expanded: stores the number of states expanded during current search
                            A state is expanded when get_available_actions(state) is called.

    """
    if asp.is_terminal_state(current_state):
        value = asp.get_result(current_state)
        return (value, None, stats)
    if cutoff_depth < 1:
        value = asp.heuristic(current_state)
        return (value, None, stats)
    else:
        if current_state.player_to_move() == 0:
            value = float('-inf')
        else:
            value = float('inf')
        actions = asp.get_available_actions(current_state)
        stats["states_expanded"] += 1
        best_action = None  #best_action will be changed when looping through the actions
        for action in actions:
            new_state = asp.transition(current_state, action)
            new_value, new_action, new_stats = minimax_helper(
                asp, new_state, stats, cutoff_depth - 1)
            if (
                (current_state.player_to_move() == 0 and new_value >= value) or 
                (current_state.player_to_move() == 1 and new_value <= value)
            ):
                value = new_value
                best_action = action
        return (value, best_action, stats)


def alpha_beta(asp: HeuristicAdversarialSearchProblem[GameState, Action], cutoff_depth=float('inf')) -> Tuple[Action, Dict[str, int]]:
    """
    Implement the alpha-beta pruning algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum.

    Input:
        asp - an AdversarialSearchProblem
        cutoff_depth - the maximum search depth, where 0 is the start state,
                    Depth 1 is all the states reached after a single action from the start state (1 ply).
                    cutoff_depth will always be greater than 0.
    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))
         a dictionary of statistics for visualization
            states_expanded: stores the number of states expanded during current search
                            A state is expanded when get_available_actions(state) is called.
    """
    best_action = None
    stats = {
        'states_expanded': 0  # Increase by 1 for every state transition
    }
    alpha_init = float('-inf')
    beta_init = float('inf')
    _, action, stats = alpha_beta_helper(asp, asp.get_start_state(), stats, cutoff_depth, alpha_init, beta_init)
    return action, stats

def alpha_beta_helper(
        asp: HeuristicAdversarialSearchProblem[GameState, Action],
        current_state: GameState,
        stats: Dict[str, int],
        cutoff_depth: float,
        alpha: float,
        beta: float) -> Tuple[int, Action, Dict[str, int]]:
    """
    Implement the alpha-beta pruning algorithm on ASPs, assuming that the given game is
    both 2-player and zero-sum.

    Input:
        asp - a HeuristicAdversarialSearchProblem
        current_state - the current state of the board
        stats: a dictionary containing the current states_expanded statistic
        cutoff_depth - the maximum search depth, where 0 is the start state. 
                    Depth 1 is all the states reached after a single action from the start state (1 ply).
                    cutoff_depth will always be greater than 0.
        alpha: minimum score that the maximizing player is assured of
        beta: maximum score that the minimizing player is assured of
    Output:
        a float representing the value of the action returned
        an action (an element of asp.get_available_actions(asp.get_start_state()))
        a dictionary of statistics for visualization
            states_expanded: stores the number of states expanded during current search
                            A state is expanded when get_available_actions(state) is called.

    """
    if asp.is_terminal_state(current_state):
        value = asp.get_result(current_state)
        return (value, None, stats)
    if cutoff_depth < 1:
        value = asp.heuristic(current_state)
        return (value, None, stats)
    actions = asp.get_available_actions(current_state)
    stats["states_expanded"] += 1
    best_action = None  #best_action will be changed when looping through the actions
    if current_state.player_to_move() == 0: 
        #we are maximizing 
        value = float('-inf')
        for action in actions:
            new_state = asp.transition(current_state, action)
            new_value, _, _ = alpha_beta_helper(
                asp, new_state, stats, cutoff_depth - 1, alpha, beta)
            if new_value >= value:
                value = new_value
                best_action = action
            if value > beta:
                return value, best_action, stats
            alpha = max(alpha, value)
    else:
        #we are minimizing
        value = float('inf')
        for action in actions:
            new_state = asp.transition(current_state, action)
            new_value, _, _ = alpha_beta_helper(
                asp, new_state, stats, cutoff_depth - 1, alpha, beta)
            if new_value <= value:
                value = new_value
                best_action = action
            if value < alpha:
                return value, best_action, stats
            beta = min(beta, value)
    return value, best_action, stats
