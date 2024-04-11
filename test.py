import argparse
from problem import Problem
from search import LocalSearchStrategy

def main():
    parser = argparse.ArgumentParser(description='Run local search algorithms.')
    parser.add_argument('-s', '--strategy', choices=['RRHC', 'SAS', 'LB'], required=True,
                        help='Choose the search strategy: RRHC for random restart hill climbing, SAS for simulated annealing search, localbeam for local beam search')
    args = parser.parse_args()

    # Create a problem instance
    initial_state = (0, 0)
    goal_state = (50, 50)
    problem = Problem(initial_state, goal_state)

    # Execute the chosen search strategy
    if args.strategy == 'RRHC':
        num_trials = 10
        best_path = LocalSearchStrategy.random_restart_hill_climbing(problem, num_trials)
    elif args.strategy == 'SAS':
        def schedule(t):
            return 1 / (t + 1)  # Example cooling schedule
        best_path = LocalSearchStrategy.simulated_annealing_search(problem, schedule)
    elif args.strategy == 'LB':
        k = 1  # Example value for k
        best_path = LocalSearchStrategy.local_beam_search(problem, k)

    # Visualize the results
    problem.show()
    problem.draw_path([best_path])

if __name__ == "__main__":
    main()
