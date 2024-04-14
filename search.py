import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math

class LocalSearch:
    def __init__(self, problem):
        self.problem = problem

    def get_neighbors(self, state):
        return self.problem.get_neighbors(state)

    def evaluate_state(self, state):
        return self.problem.evaluate_state(state)
    
class RandomRestartHillClimbing(LocalSearch):
    def __init__(self, problem, num_trials=10):
        super().__init__(problem)
        self.num_trials = num_trials

    def search(self):
        best_solution = None
        best_score = float('-inf')

        for _ in range(self.num_trials):
            x_init = np.random.randint(0, self.problem.X.size)
            y_init = np.random.randint(0, self.problem.Y.size)
            state = (x_init, y_init)
            solution = []
            max_value = self.problem.Z[state[1], state[0]]
            prev_max = -1000

            while True:
                neighbors = self.problem.get_neighbors(state)
                heuristic_values = [self.problem.Z[t[1], t[0]] for t in neighbors]

                if max_value <= prev_max:
                    break

                prev_max = max_value
                max_value = max(heuristic_values)
                solution.append((state[0], state[1], self.problem.Z[state[1], state[0]]))

                for i in range(len(neighbors)):
                    if heuristic_values[i] == max_value:
                        state = neighbors[i]
                        break
            if max_value > best_score:
                best_score = max_value
                best_solution = solution

        return best_solution

class SimulatedAnnealing(LocalSearch):
    def __init__(self, problem):
        super().__init__(problem)

    def schedule(self, t):
        return 100 * math.exp(-0.001 * t)

    def search(self):
        x_initial = np.random.randint(0, self.problem.X.size)
        y_initial = np.random.randint(0, self.problem.Y.size)
        current_state = (x_initial, y_initial)
        current_value = int(self.problem.evaluate(current_state))

        path = [current_state + (current_value,)]
        step = 1

        while True:
            t = self.schedule(step)
            if t <= 0:
                break

            neighbors = self.problem.get_neighbors(current_state)
            if not neighbors:
                break

            next_state = random.choice(neighbors)
            next_value = int(self.problem.evaluate(next_state))

            delta_E = next_value - current_value

            if delta_E > 0 or random.random() < np.exp(delta_E / t):
                current_state = next_state
                current_value = next_value
                path.append(current_state + (current_value,))

            step += 1
        return path

class LocalBeamSearch(LocalSearch):
    def __init__(self, problem, k):
        super().__init__(problem)
        self.k = k

    def get_successors(self, x, y, space):
        X, Y, val = space
        successors = []

        for dx, dy in [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0) and dx * dy == 0]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(X) and 0 <= new_y < len(Y):
                successors.append((new_x, new_y, val[new_y, new_x]))

        return successors

    def local_beam_search(self, problem, k=1):
        X, Y, Z = problem.X, problem.Y, problem.Z
        initial_x, initial_y = random.choice(X), random.choice(Y)
        initial_state = ((initial_x, initial_y, Z[initial_y, initial_x]), [(initial_x, initial_y, Z[initial_y, initial_x])])

        states = [initial_state]
        best_path = initial_state[1]

        while True:
            new_states = []
            for state in states:
                x, y, _ = state[0]
                successors = self.get_successors(x, y, (X, Y, Z))
                for successor in successors:
                    new_path = state[1] + [successor]
                    new_states.append((successor, new_path))

            new_states = sorted(new_states, key=lambda x: x[0][2], reverse=True)[:k]
            if best_path[-1][2] < new_states[0][0][2]:
                best_path = new_states[0][1]
                states = new_states
            else:
                return best_path
