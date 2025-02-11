from heuristic_adversarial_search_problem import HeuristicAdversarialSearchProblem
from .ttt_problem import TTTState, TTTProblem

SPACE = " "
X = "X"  # Player 0 is X
O = "O"  # Player 1 is O
PLAYER_SYMBOLS = [X, O]


class HeuristicTTTProblem(TTTProblem, HeuristicAdversarialSearchProblem):

    def heuristic(self, state: TTTState) -> float:
        """
        heuristic takes in a state of a tic-tac-toe game. For each row, colum, and diagnol where a
        player could win, if there is only X's in the line, value is increased by the number of X's.
        If there are only O's in the line, value is decreased by the number of O's. Returns value
        Input:
            state - TTTState
        Output:
            value - float (represents the huerisitc measure of the board state with a positive number
            representing player one having the advantage and a negative number representing player 2
            having the advantage)
        """
        board = state.board
        dimension = len(board)
        #value is positive if player 1 is doing better and negative is player 2 is doing better
        value = 0

        #looping through each row:
        for row_num in range(0, dimension):
            # x is player 1, o is player 2
            x_count = 0
            o_count = 0
            for col_num in range(0, dimension):
                if board[row_num][col_num] == "X":
                    x_count += 1
                if board[row_num][col_num] == "O":
                    o_count += 1
            if x_count > 0 and o_count == 0:
                value += x_count
            elif o_count > 0 and x_count == 0:
                value -= o_count

        #looping through each colum:
        for col_num in range(0, dimension):
            # x is player 1, o is player 2
            x_count = 0
            o_count = 0
            for row_num in range(0, dimension):
                if board[row_num][col_num] == "X":
                    x_count += 1
                if board[row_num][col_num] == "O":
                    o_count += 1
            if x_count > 0 and o_count == 0:
                value += x_count
            elif o_count > 0 and x_count == 0:
                value -= o_count

        #looking at the diagnols:
        x_count_diag_1 = 0
        o_count_diag_1 = 0
        for i in range(0, dimension):
            if board[i][i] == "X":
                x_count_diag_1 += 1
            if board[i][i] == "O":
                o_count_diag_1 += 1
        if x_count_diag_1 > 0 and o_count_diag_1 == 0:
            value += x_count_diag_1
        elif o_count_diag_1 > 0 and x_count_diag_1 == 0:
            value -= o_count_diag_1

        x_count_diag_2 = 0
        o_count_diag_2 = 0
        for i in range(0, dimension):
            if board[dimension - 1 - i][i] == "X":
                x_count_diag_2 += 1
            if board[dimension - 1 - i][i] == "O":
                o_count_diag_2 += 1
        if x_count_diag_2 > 0 and o_count_diag_2 == 0:
            value += x_count_diag_2
        elif o_count_diag_2 > 0 and x_count_diag_2 == 0:
            value -= o_count_diag_2
        return value