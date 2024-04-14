import argparse
from problem import Problem
from search import RandomRestartHillClimbing, LocalBeamSearch, SimulatedAnnealing

def schedule(t):
    return 1/(t + 1)

def test_random_restart_hill_climbing():
    problem = Problem('monalisa.jpg')
    hill_climber = RandomRestartHillClimbing(problem)
    path = hill_climber.search()
    
    problem.draw_path(path)

def test_beam_search():
    problem = Problem('monalisa.jpg')
    local_beam_search = LocalBeamSearch(problem, 5)
    path = local_beam_search.local_beam_search(problem, 5)
    problem.draw_path(path)

def test_simulated_annealing():
    problem = Problem('monalisa.jpg')
    simulated_annealing = SimulatedAnnealing(problem)
    path = simulated_annealing.search()
    problem.draw_path(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test different local search strategies.")
    parser.add_argument("-s", "--strategy", choices=["RRHC", "SAS", "LB"], help="Choose a strategy: RRHC, SAS, LB")
    args = parser.parse_args()

    if args.strategy == "RRHC":
        test_random_restart_hill_climbing()
    elif args.strategy == "SAS":
        test_simulated_annealing()
    elif args.strategy == "LB":
        test_beam_search()
    else:
        print("Please provide a valid strategy using the -s or --strategy argument.")
