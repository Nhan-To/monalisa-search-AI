import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
import random
import math

class Problem:
    def __init__(self, filename):
        self.X, self.Y, self.Z = self.load_state_space(filename)
        
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
        self.X, self.Y = np.meshgrid(self.X, self.Y)
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        plt.show()

    def draw_path(self, path):
        self.X, self.Y = np.meshgrid(self.X, self.Y)
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        x = [p[0] for p in path]
        y = [p[1] for p in path]
        z = self.Z[y, x]
        ax.plot(x, y, z, 'r-', zorder=3, linewidth=0.5)
        plt.show()

    def evaluate(self, state):
        x, y = state
        return self.Z[y, x]

    def get_neighbors(self, state):
        x, y = state
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    new_x = x + dx
                    new_y = y + dy
                    if 0 <= new_x < self.Z.shape[1] and 0 <= new_y < self.Z.shape[0]:
                        neighbors.append((new_x, new_y))
        return neighbors

problem = Problem('monalisa.jpg')
problem.show()