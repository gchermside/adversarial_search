from abc import ABC, abstractmethod
from heuristic_adversarial_search_problem import HeuristicAdversarialSearchProblem
from asps.heuristic_connect_four import HeuristicConnectFourProblem
from asps.heuristic_ttt_problem import HeuristicTTTProblem
from adversarial_search import minimax, alpha_beta


class GameAgent():
    """
    Interface for GameAgents
    An agent is an object that takes actions given some state information.
    For the purposes of this assignment, it is a combination of search algorithm and heuristic.
    Agents with different search algorithms or heuristics can be compared by playing games.
    """
    @abstractmethod
    def get_move(self, state):
        """
        return an action (and stats) given a state
        """
        pass


class MinimaxAgent(GameAgent):
    """
    A GameAgent that uses minimax search to find the best move
    """

    def __init__(self, searchproblem, cutoff_depth=3):
        self.searchproblem = searchproblem
        self.cutoff_depth = cutoff_depth

    def get_move(self, state):
        self.searchproblem.set_start_state(state)
        return minimax(self.searchproblem, cutoff_depth=self.cutoff_depth)


class AlphaBetaAgent(GameAgent):
    """
    A GameAgent that uses alpha-beta search to find the best move
    """

    def __init__(self, searchproblem, cutoff_depth=3):
        self.searchproblem = searchproblem
        self.cutoff_depth = cutoff_depth

    def get_move(self, state):
        self.searchproblem.set_start_state(state)
        return alpha_beta(self.searchproblem, cutoff_depth=self.cutoff_depth)
