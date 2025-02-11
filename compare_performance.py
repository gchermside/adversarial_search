import argparse
from matplotlib import pyplot as plt
import numpy as np
from adversarial_search_problem import AdversarialSearchProblem
from asps.heuristic_connect_four import HeuristicConnectFourProblem
from asps.heuristic_ttt_problem import HeuristicTTTProblem
from agents import MinimaxAgent, AlphaBetaAgent
import copy
import pickle
import tqdm


def run_game_with_stats(asp: AdversarialSearchProblem, agents, game_ui=None):
    """
    Inputs:
            - asp: a game to play, represented as an adversarial search problem
            - agents: a list of agents which determines the algorithm used for each player
            - game_ui (optional): a GameUI that visualizes ASPs and allows for
                    direct input in place of a bot that is None. If no argument is
                    passed, run_game() will not be interactive.
    Output:
            - the evaluation of the terminal state.
            - the total number of states expanded by each agent.
            - a list of the number of states expanded per move for each agent.
    """

    # Ensure game_ui is present if a bot is None:
    if not game_ui and any(agent is None for agent in agents):
        raise ValueError(
            "A GameUI instance must be provided if any bot is None.")

    state = asp.get_start_state()
    if game_ui:
        game_ui.update_state(state)
        game_ui.render()

    total_states_expanded = [0, 0]
    states_per_move = [[], []]

    while not asp.is_terminal_state(state):
        curr_bot = agents[state.player_to_move()]

        # Obtain decision from the bot itself, or from GameUI if bot is None:
        if curr_bot:
            decision, stats = curr_bot.get_move(state)
            total_states_expanded[state.player_to_move()
                                  ] += stats['states_expanded']
            states_per_move[state.player_to_move()].append(
                stats['states_expanded'])
            # If the bot tries to make an invalid action,
            # returns any valid action:
            available_actions = asp.get_available_actions(state)
            if decision not in available_actions:
                decision = available_actions.pop()
        else:
            decision = game_ui.get_user_input_action()

        result_state = asp.transition(state, decision)
        asp.set_start_state(result_state)
        state = result_state

        if game_ui:
            game_ui.update_state(state)
            game_ui.render()

    return asp.get_result(asp.get_start_state()), total_states_expanded, states_per_move


def run_games(asp: AdversarialSearchProblem, agents, game_ui=None):
    """
    Run a pair of games between two agents and report results
    Agents alternate going first.

    Inputs:
            - asp: a game to play, represented as an adversarial search problem
            - agents: a list of agents which determines the algorithm used for each player
    Returns:
        - a tuple of the results of the two games
        - the total number of states expanded by each agent.
        - a list of the number of states expanded per move for each agent.
    """
    curr_game = copy.deepcopy(asp)
    outcome1, states_expanded, states_per_move = run_game_with_stats(
        curr_game, agents, game_ui)

    curr_game = copy.deepcopy(asp)
    outcome2, states_expanded2, states_per_move2 = run_game_with_stats(
        curr_game, list(reversed(agents)), game_ui)

    # Add stats of second game to stats of game 1
    states_expanded[0] += states_expanded2[1]
    states_expanded[1] += states_expanded2[0]

    player1_win = 0
    player2_win = 0
    if outcome1 > 0:
        player1_win += 1
        player2_win -= 1
    elif outcome1 < 0:
        player1_win -= 1
        player2_win += 1
    # Note, bots order is reversed for second game
    if outcome2 > 0:
        player1_win -= 1
        player2_win += 1
    elif outcome2 < 0:
        player1_win += 1
        player2_win -= 1

    return (player1_win, player2_win), states_expanded, states_per_move


def run_tournament(game: AdversarialSearchProblem, max_depth_ab=4, max_depth_minimax=4, n_trials=1):
    """
    Run pairwise games between minimax agents up to some depth
    and alpha beta agents up to some depth.

    Repeat for n_trials and record results and stats.

    Returns:
        - a 2d-array of results (corresponds to the number of wins for AB - wins for minimax)
        - a dictionary of states expanded by alpha beta agent for each depth
        - a dictionary of states expanded by minimax agent for each depth
    """

    # Create bots with varying cutoff depths
    minimax_bots = []
    alpha_beta_bots = [AlphaBetaAgent(copy.deepcopy(
        game), cutoff_depth=i) for i in range(max_depth_ab)]
    minimax_bots = [MinimaxAgent(copy.deepcopy(game), cutoff_depth=i)
                    for i in range(max_depth_minimax)]

    # Create 2d-matrix of results row player (ab) vs column player (minimax)
    results = np.zeros((max_depth_minimax, max_depth_ab))

    # Keep track of states expanded for each bot
    states_expanded_ab = {i: [] for i in range(max_depth_ab)}
    states_expanded_minimax = {i: [] for i in range(max_depth_minimax)}

    for i in range(max_depth_minimax):
        print(f"Evaluating minimax with depth {i+1}")
        for j in tqdm.tqdm(range(max_depth_ab)):
            for _ in range(n_trials):
                # Run games runs 2 games, where each player goes first once
                (player1_results, _), states_expanded, states_per_move = run_games(
                    game, [alpha_beta_bots[j], minimax_bots[i]], None)
                states_expanded_minimax[i].append(states_expanded[1])
                num_moves = len(states_per_move[0])
                states_expanded_ab[j].append(
                    states_expanded[0] / num_moves)

                # Results just keeps track of ab performance (minimax is the negative of ab), so only use player1
                results[i][j] += player1_results

    return results, states_expanded_ab, states_expanded_minimax


def make_results_heatmap(results, n_trials):
    """
    Inputs: 
        - results: a 2d-array of results (corresponds to the number of wins for AB - wins for minimax)
        - n_trials: the number of trials ran for each depth
    """

    cmap = 'coolwarm'
    # reverse rows and columns for plotting
    results = np.flip(results, axis=0)
    plt.imshow(results, cmap=cmap)
    plt.xlabel("Alpha Beta Cutoff Depth")
    plt.ylabel("Minimax Cutoff Depth")
    plt.title('Alpha Beta Score (Playing Against Minimax), Varying Depth')
    # Add text to plot
    for i in range(len(results)):
        for j in range(len(results[0])):
            plt.text(j, i, f'{results[i][j]}/{2 * n_trials}',
                     ha='center', va='center', color='black')
    plt.colorbar()
    plt.xticks(range(len(results[0])-1, -1, -1),
               range(len(results[0]), 0, -1))
    plt.yticks(range(len(results)), range(len(results), 0, -1))
    plt.savefig('alpha_beta_performance.png')
    plt.clf()


def make_states_expanded_plot(states_expanded_ab, states_expanded_minimax, max_depth_ab, max_depth_minimax):
    """
    Make plot that shows the average number of states expanded by each bot for each depth.
    """
    plt.bar(np.array(range(1, max_depth_ab+1)) - 0.2,
            [np.mean(states_expanded_ab[i]) for i in range(max_depth_ab)], 0.4, label="Alpha Beta")
    plt.bar(np.array(range(1, max_depth_minimax+1)) + 0.2,
            [np.mean(states_expanded_minimax[i]) for i in range(max_depth_minimax)], 0.4, label="Minimax")
    plt.xlabel("Cutoff Depth")
    plt.ylabel("States Expanded")
    plt.title('States Expanded vs Cutoff Depth')
    # plt.yscale('log')
    plt.legend()
    plt.savefig('states_expanded.png')
    plt.clf()


def make_pareto_front(results, states_expanded_ab, states_expanded_minimax):
    """
    Make plot for pareto front between number of states expanded and performance
    """
    performance = {}
    for i in range(len(results)):
        result = results[i].sum(axis=-1)
        performance[f'minimax depth={i+1}'] = (-result, np.sum(states_expanded_minimax[i]))
    for j in range(len(results[0])):
        result = results[:, j].sum(axis=-1)
        performance[f'alpha beta depth={j+1}'] = (result, np.sum(states_expanded_ab[j]))
    shapes = ['o', 's', 'D', 'v', '^', '<', '>',
              'p', '*', 'h', 'H', '8', 'd', 'P', 'X']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for k, v in performance.items():
        # Extract depth from key
        depth = int(k.split('=')[1])
        if 'minimax' in k:
            shape = shapes[0]
        else:
            shape = shapes[1]
        plt.scatter(v[1], v[0], color=colors[depth], label=k, marker=shape)
    plt.xlabel("States Expanded")
    plt.ylabel("Performance Score")
    plt.legend()
    plt.savefig('pareto_frontier.png')
    plt.clf()


def make_states_per_move_plot(game, depth=4):
    """
    Plot number of states expanded per move in a game.
    """
    minimax_bot = MinimaxAgent(game, cutoff_depth=depth)
    alpha_beta_bot = AlphaBetaAgent(game, cutoff_depth=depth)

    results, states_expanded, states_per_move = run_games(
        game, [alpha_beta_bot, minimax_bot])
    plt.bar(np.array(
        range(len(states_per_move[0]))) - 0.2, states_per_move[0], 0.4, label="Alpha Beta")
    plt.bar(np.array(
        range(len(states_per_move[1]))) + 0.2, states_per_move[1], 0.4, label="Minimax")
    plt.xlabel("Move #")
    plt.ylabel("States Expanded")
    plt.title('States Expanded vs Move # (Cutoff=4)')
    plt.xticks(range(len(states_per_move[0])), range(
        1, len(states_per_move[0]) + 1))
    plt.xscale('log')
    plt.legend()
    plt.savefig('states_expanded_per_move.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_trials", type=int, default=1)
    parser.add_argument(
        "--max_depth", type=int, default=4)
    parser.add_argument("--game", type=str,
                        choices=["ttt", "connect4"], default="connect4")

    args = parser.parse_args()

    if args.game == "ttt":
        game = HeuristicTTTProblem()
    elif args.game == "connect4":
        game = HeuristicConnectFourProblem()
    else:
        raise ValueError("Invalid game")
    max_depth_ab = args.max_depth
    max_depth_minimax = args.max_depth
    n_trials = args.n_trials
    results = np.zeros((4, 4))
    results, states_expanded_ab, states_expanded_minimax = run_tournament(
        game, max_depth_ab=max_depth_ab, max_depth_minimax=max_depth_minimax, n_trials=n_trials)

    make_pareto_front(results, states_expanded_ab, states_expanded_minimax)
    make_results_heatmap(results, n_trials=n_trials)
    make_states_expanded_plot(states_expanded_ab, states_expanded_minimax,
                              max_depth_ab=max_depth_ab, max_depth_minimax=max_depth_minimax)

    make_states_per_move_plot(game, depth=4)
