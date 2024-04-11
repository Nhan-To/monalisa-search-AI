import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

class Problem:
    def __init__(self, initial_state, goal_state):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.max_z_value = 0.0
        self.X, self.Y, self.Z = self.load_state_space('monalisa.jpg')

    def load_state_space(self, filename):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        h, w = img.shape
        X = np.arange(w)
        Y = np.arange(h)
        Z = img
        return X, Y, Z
    
    def actions(self, state):
        """
        Returns the possible actions from the given state.
        """
        # This would depend on the problem, e.g., moving in four directions or any other valid movement.
        pass

    def result(self, state, action):
        """
        Returns the state that results from taking the given action from the given state.
        """
        # Apply the action to the state to get the new state.
        pass

    def goal_test(self, state):
        """
        Returns True if the given state has the highest Z value, False otherwise.
        """
        # Extract the Z value of the current state
        x, y = state
        current_z_value = self.Z[y, x]

        # Check if the Z value of the current state is equal to the maximum Z value
        return current_z_value > self.max_z_value

    def find_max_z_value(self):
        """
        Find the maximum Z value in the state space.
        """
        self.max_z_value = np.max(self.Z)
        print(self.max_z_value)

    def path_cost(self, c, state1, action, state2):
        """
        Returns the cost of a solution path that arrives at state2 from state1 via action.
        """
        # Calculate the cost of the action.
        pass

    def evaluate(self, state):
        """
        Evaluate the given state.
        """
        x, y = state
        return self.Z[y, x]

    def neighbors(self, state):
        """
        Generate neighboring states of the given state.
        """
        x, y = state
        possible_actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Assuming moving in four directions
        max_x, max_y = self.X.shape[0], self.Y.shape[0]
        
        neighbors = [(x + dx, y + dy) for dx, dy in possible_actions 
        if 0 <= x + dx < max_x and 0 <= y + dy < max_y]
        
        return neighbors

    
    @staticmethod
    def load_state_space(filename):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        h, w = img.shape
        X = np.arange(w)
        Y = np.arange(h)
        Z = img
        return X, Y, Z

    def show(self):
        """
        Visualizes all tuples of (x, y, z).
        """
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Ensure X, Y are compatible with Z's shape
        X, Y = np.meshgrid(self.X, self.Y)
        
        ax.plot_surface(X, Y, self.Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.show()


    def draw_path(self, path):
        X, Y, Z = self.load_state_space('monalisa.jpg')
        X, Y = np.meshgrid(X, Y)

        fig = plt.figure(figsize=(8,6))
        ax = plt.axes(projection='3d')
        # draw state space (surface)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        # draw a polyline on the surface
        ax.plot(range(0, 50), range(0, 50), Z[range(0, 50), range(0, 50)], 'r-', zorder=3, linewidth=0.5)
        plt.show()
