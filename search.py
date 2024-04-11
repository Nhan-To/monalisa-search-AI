import random
import numpy as np

class LocalSearchStrategy:
    @staticmethod
    def random_restart_hill_climbing(problem, num_trial):
        best_path = None
        best_evaluation = float('-inf')
        
        for _ in range(num_trial):
            current_state = problem.initial_state
            current_evaluation = problem.evaluate(current_state)
            
            while True:
                neighbors = problem.neighbors(current_state)
                next_state = max(neighbors, key=lambda x: problem.evaluate(x))
                next_evaluation = problem.evaluate(next_state)
                
                if next_evaluation > current_evaluation:
                    current_state = next_state
                    current_evaluation = next_evaluation
                else:
                    break
            
            if current_evaluation > best_evaluation:
                best_path = current_state
                best_evaluation = current_evaluation
        
        return best_path
    
    @staticmethod
    def simulated_annealing_search(problem, schedule):
        max_iterations = 1000
        current_state = problem.initial_state
        current_evaluation = problem.evaluate(current_state)
        best_state = current_state
        best_evaluation = current_evaluation
        
        t = 0 
        
        while t < max_iterations:
            temperature = schedule(t)
            if temperature == 0:
                return best_state
            
            next_state = random.choice(problem.neighbors(current_state))
            next_evaluation = problem.evaluate(next_state)
            
            delta_evaluation = next_evaluation - current_evaluation
            
            if delta_evaluation > 0 or random.random() < np.exp(delta_evaluation / temperature):
                current_state = next_state
                current_evaluation = next_evaluation
                if current_evaluation > best_evaluation:
                    best_state = current_state
                    best_evaluation = current_evaluation
            
            t += 1  # Increment time step
        
        return best_state  # Return the best state found within the maximum iterations


    @staticmethod
    def local_beam_search(problem, k):
    # Initialize k random states
        max_iterations = 1000
        states = [problem.initial_state] * k
        
        for _ in range(max_iterations):
            new_states = []
            for state in states:
                neighbors = problem.neighbors(state)
                new_states.extend(random.sample(neighbors, min(k, len(neighbors))))
            
            # Evaluate new states
            evaluations = [problem.evaluate(state) for state in new_states]
            
            # Select top k states based on evaluation
            top_k_indices = np.argsort(evaluations)[-k:]
            states = [new_states[i] for i in top_k_indices]
            
            # Check if any of the states is a goal state
            for state in states:
                if problem.goal_test(state):
                    return state
        return None  # No solution found within max_iterations

