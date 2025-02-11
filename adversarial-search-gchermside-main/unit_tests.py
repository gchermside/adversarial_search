import unittest

from asps.game_dag import DAGState, GameDAG
from asps.heuristic_ttt_problem import HeuristicTTTProblem, X, O, SPACE
from asps.heuristic_connect_four import HeuristicConnectFourProblem
from adversarial_search import alpha_beta, minimax
from asps.ttt_problem import TTTState

class IOTest(unittest.TestCase):
    """
    Tests IO for adversarial search implementations.
    Contains basic/trivial test cases.

    Each test function instantiates an adversarial search problem (DAG) and tests
    that the algorithm returns a valid action.

    It does NOT test whether the action is the "correct" action to take.
    """

    def _check_result(self, result, dag):
        """
        Tests whether the result is one of the possible actions of the dag.
        Input:
            result - the return value of an adversarial search problem.
                     This should be an action.
            dag - the GameDAG that was used to test the algorithm.
        """
        self.assertIsNotNone(result, "Output should not be None")
        start_state = dag.get_start_state()
        potential_actions = dag.get_available_actions(start_state)
        self.assertIn(result, potential_actions, "Output should be an available action")

    def test_minimax(self):
        """
        Test minimax on a basic GameDAG.
        """
        X = True
        _ = False
        matrix = [
            [_, X, X, _, _, _, _],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: -1, 4: -2, 5: -3, 6: -4}

        dag = GameDAG(matrix, start_state, terminal_evaluations)
        result, stats = minimax(dag)
        self._check_result(result, dag)
        print(result)
        self.assertEqual(result, 1)
        self.assertEqual(stats["states_expanded"], 3)

    def test_depth(self):
        """
        tests that minimax and alpha_beta do not go beyond the given depth
        """
        classic_ttt_problem = HeuristicTTTProblem()
        result_depth_1, stats_depth_1 = minimax(classic_ttt_problem, 1)
        self.assertEqual(stats_depth_1["states_expanded"], 1)
        self.assertEqual(result_depth_1, (1, 1))

        result_depth_2, stats_depth_2 = minimax(classic_ttt_problem, 2)
        self.assertEqual(stats_depth_2["states_expanded"], 10)
        self.assertEqual(result_depth_2, (1, 1))

        connect4 = HeuristicConnectFourProblem()
        _, stats_d_1 = minimax(connect4, 1)
        _, stats_d_2 = minimax(connect4, 2)
        _, stats_d_3 = minimax(connect4, 3)
        self.assertEqual(stats_d_1["states_expanded"], 1)
        self.assertEqual(stats_d_2["states_expanded"], 8)
        self.assertEqual(stats_d_3["states_expanded"], 57)

        _, stats_d_1 = alpha_beta(connect4, 1)
        _, stats_d_2 = alpha_beta(connect4, 2)
        _, stats_d_3 = alpha_beta(connect4, 3)
        self.assertEqual(stats_d_1["states_expanded"], 1)
        self.assertEqual(stats_d_2["states_expanded"], 8)


    def test_minimax_and_alpha_beta(self):
        """
        tests minimax and alpha_beta on edge case GameDAGs
        """
        X = True
        _ = False
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: -1, 4: -2, 5: -3, 6: -4}
        matrix_easy_win = [
            [_, X, X, X, _, _, _],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        dag_easy_win = GameDAG(matrix_easy_win, start_state, terminal_evaluations)
        result, stats = minimax(dag_easy_win)
        self._check_result(result, dag_easy_win)
        self.assertTrue(result == 3)
        self.assertEqual(stats["states_expanded"], 3)

        result, stats = alpha_beta(dag_easy_win)
        self._check_result(result, dag_easy_win)
        self.assertTrue(result == 3)
        self.assertEqual(stats["states_expanded"], 3)

        matrix_easy_tie = [
            [_, X, X, _, X, _, _],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        dag_easy_tie = GameDAG(matrix_easy_tie, start_state, terminal_evaluations)
        result, stats = minimax(dag_easy_tie)
        self._check_result(result, dag_easy_tie)
        self.assertTrue(result == 1 or result == 4)
        self.assertEqual(stats["states_expanded"], 3)

        result, stats = alpha_beta(dag_easy_tie)
        self._check_result(result, dag_easy_tie)
        self.assertTrue(result == 1 or result == 4)
        self.assertEqual(stats["states_expanded"], 3)

        matrix_easy_loss = [
            [_, X, X, _, _, X, X],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        dag_easy_loss = GameDAG(matrix_easy_loss, start_state, terminal_evaluations)
        result, stats = minimax(dag_easy_loss)
        self._check_result(result, dag_easy_loss)
        self.assertTrue(result == 1)
        self.assertEqual(stats["states_expanded"], 3)

        result, stats = alpha_beta(dag_easy_loss)
        self._check_result(result, dag_easy_loss)
        self.assertTrue(result == 1)
        self.assertEqual(stats["states_expanded"], 3)

        matrix_complex = [
            [_, X, X, _, _, _, _, _],
            [_, _, X, _, _, _, _, X],
            [_, _, _, X, X, X, _, _],
            [_, _, _, _, X, _, X, _],
            [_, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _],
            [_, _, _, _, _, _, _, _],
        ]
        complex_terminal_evaluations = {4: -1, 5: 0, 6: 2, 7: 10}
        dag_complex = GameDAG(matrix_complex, start_state, complex_terminal_evaluations)
        result, stats = minimax(dag_complex)
        self._check_result(result, dag_complex)
        self.assertEqual(result, 1)
        self.assertEqual(stats["states_expanded"], 6)

        result, stats = alpha_beta(dag_complex)
        self._check_result(result, dag_complex)
        self.assertEqual(result, 1)
        self.assertEqual(stats["states_expanded"], 6)


    def test_alpha_beta(self):
        """
        Test alpha-beta pruning on a basic GameDAG.
        """
        X = True
        _ = False
        matrix = [
            [_, X, X, _, _, _, _],
            [_, _, _, X, X, _, _],
            [_, _, _, _, _, X, X],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
            [_, _, _, _, _, _, _],
        ]
        start_state = DAGState(0, 0)
        terminal_evaluations = {3: -1, 4: -2, 5: -3, 6: -4}

        dag = GameDAG(matrix, start_state, terminal_evaluations)
        result, stats = alpha_beta(dag)
        self._check_result(result, dag)
    
    def test_heuristic(self):
        """"
        Tests that my heuristic for tic-tak-toe boards works as expected
        """
        #intializing a ttt_problem (not used in heuristic)
        my_fake_ttt_problem = HeuristicTTTProblem(3, None, 0)

        _ = SPACE
        ttt_board = [
            [_, X, X,],
            [_, _, _,],
            [_, O, _,],
        ]
        ttt_state = TTTState(ttt_board, 0)
        self.assertEqual(3, my_fake_ttt_problem.heuristic(state=ttt_state))

        O_winning = [
            [_, X, _,],
            [_, O, O,],
            [X, _, O,],
        ]
        ttt_state = TTTState(O_winning, 0)
        self.assertEqual(-4, my_fake_ttt_problem.heuristic(state=ttt_state))

        ttt_4_by_4 = [
            [_, X, X, O],
            [_, _, _, _],
            [_, O, X, _],
            [_, O, _, _]
        ]
        ttt_state_4_by_4 = TTTState(ttt_4_by_4, 0)
        self.assertEqual(-1, my_fake_ttt_problem.heuristic(state=ttt_state_4_by_4))

        empty_ttt = [
            [_, _, _,],
            [_, _, _,],
            [_, _, _,],
        ]
        ttt_state = TTTState(empty_ttt, 0)
        self.assertEqual(0, my_fake_ttt_problem.heuristic(state=ttt_state))

        full_ttt = [
            [X, X, O,],
            [O, X, X,],
            [X, O, O,],
        ]
        ttt_state = TTTState(full_ttt, 0)
        self.assertEqual(0, my_fake_ttt_problem.heuristic(state=ttt_state))

        full_4_by_4 = [
            [O, X, X, O],
            [X, _, O, _],
            [_, O, X, _],
            [X, O, _, X]
        ]
        ttt_state = TTTState(full_4_by_4, 0)
        self.assertEqual(0, my_fake_ttt_problem.heuristic(state=ttt_state))

        random_4_by_4 = [
            [O, _, X, O],
            [X, _, _, _],
            [_, O, X, _],
            [_, _, X, _]
        ]
        ttt_state = TTTState(random_4_by_4, 0)
        self.assertEqual(1, my_fake_ttt_problem.heuristic(state=ttt_state))


if __name__ == "__main__":
    unittest.main()
