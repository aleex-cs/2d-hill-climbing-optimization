#!/usr/bin/env python3

import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import time

def matrix(print_option, N, p, seed):

    class LCG:
        def __init__(self, seed):
            self.seed = seed
            self.a = 1664525
            self.c = 1013904223
            self.m = 2**32

        def next(self):
            self.seed = (self.a * self.seed + self.c) % self.m  # Update the state
            return self.seed / self.m  # Normalize the number
        
    try:
        if print_option not in ["s", "n"] or not isinstance(N, int):
            raise ValueError
    except ValueError:
        print("Incorrect usage. The first argument must be 's' or 'n', and N must be an integer.")
        return  # Exit the function in case of error
    
    # Define ranges for x and y
    a, b = -2, 1  # New x range
    c, d = 0, 9  # New y range
    
    # Calculate evenly spaced points
    x_values = np.linspace(a, b, N)
    y_values = np.linspace(c, d, N)
    
    # Initialize the matrix with -Infinity
    t0_i = time.time()
    mat = np.full((N, N), -np.inf)
    t1_i = time.time()

    # Original f2 function
    def f2(x, y):
        r = 0
        if x != 0 and y != 0:
            r = (x + y) / 2 + (8 * x * x * y * y * math.exp(0 - math.sqrt(x * x + y * y / 4))) / math.sqrt(x * x + y * y / 4) + math.sin((x + y) / 10000)
        return r
        
    # Function to calculate value only if it hasn't been calculated before
    def calculate_if_needed(i, j):
        if mat[i, j] == -np.inf:  # Only calculate if -Infinity
            x = x_values[i]
            y = y_values[j]
            mat[i, j] = f2(x + y * 0.99, y)  # Calculate and assign
        return mat[i, j]  # Return the calculated or existing value

    # Function to perform the Hill Climbing algorithm
    def hill_climbing(x_ini, y_ini):
        current_x, current_y = x_ini, y_ini
        max_value = calculate_if_needed(current_x, current_y)  # Use the modified function
        max_position = (current_x, current_y)
        
        points_traversed = []

        while True:
            points_traversed.append((current_x, current_y, max_value))
            neighbors = {}

            if current_x > 0:  # Up
                neighbors[(current_x - 1, current_y)] = calculate_if_needed(current_x - 1, current_y)
            if current_x < N - 1:  # Down
                neighbors[(current_x + 1, current_y)] = calculate_if_needed(current_x + 1, current_y)
            if current_y > 0:  # Left
                neighbors[(current_x, current_y - 1)] = calculate_if_needed(current_x, current_y - 1)
            if current_y < N - 1:  # Right
                neighbors[(current_x, current_y + 1)] = calculate_if_needed(current_x, current_y + 1)

            new_max_value = max(neighbors.values(), default=max_value)
            if max_value >= new_max_value:
                break
            
            for pos, value in neighbors.items():
                if value == new_max_value:
                    current_x, current_y = pos
                    max_value = new_max_value
                    max_position = (current_x, current_y)
                    break
        
        return max_position, max_value, points_traversed

    # Monte Carlo: generate p random starting points
    lcg = LCG(seed)
    all_starts = []
    all_max_positions = []
    all_paths = []
    global_max_value = float('-inf')  # Initialize global maximum as very low
    global_max_position = None
    total_points_traversed = 0  # Counter for total points traversed

    t0_hc = time.time()
    for _ in range(p):
        # Generate random initial point (choose an index in the matrix)
        x_ini = math.floor(lcg.next() * N)
        y_ini = math.floor(lcg.next() * N)
        
        # Run Hill Climbing from the random starting point
        max_position, max_value, points_traversed = hill_climbing(x_ini, y_ini)
        
        # Save all starting points and maxima found
        all_starts.append((x_ini, y_ini))
        all_max_positions.append(max_position)
        all_paths.append(points_traversed)

        # Update the global maximum
        if max_value > global_max_value:
            global_max_value = max_value
            global_max_position = max_position

        # Count the number of points traversed in the current path
        total_points_traversed += len(points_traversed)

        # Print the paths if the parameter is 's'
        for idx, point in enumerate(points_traversed):
            print(f"p{idx + 1}: {point[0]} {point[1]} {point[2]:.4f}")

    t1_hc = time.time()

    T_i = t1_i - t0_i
    T_m = t1_hc - t0_hc
    
    print(f"T_i: {T_i:.6f}")
    print(f"T_m: {T_m:.6f}")
    print(f"T_i+T_m: {T_i+T_m}")
    print(f"Max_m: {global_max_position} {global_max_value:.6f}")

    # Show total number of points traversed
    print(f"Total_steps: {total_points_traversed}")

    # Print the matrix without brackets or commas
    if print_option == 's':
        for row in mat:
            print(' '.join(f'{val:.4f}' if val != -np.inf else '-Inf' for val in row))

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(mat, cmap='inferno', interpolation='nearest')
    plt.colorbar(label='Function Value')
    
    # Highlight all starting points
    for start in all_starts:
        plt.scatter(start[1], start[0], color='lime', s=100, edgecolor='black', zorder=5)
    
    # Highlight all local maxima
    for max_position in all_max_positions:
        plt.scatter(max_position[1], max_position[0], color='cyan', s=100, edgecolor='black', zorder=5)

    # Draw the paths followed for all points
    for path in all_paths:
        path_x, path_y = zip(*[(x, y) for x, y, _ in path])
        plt.plot(path_y, path_x, color='black', linewidth=1.0, zorder=4)

    # Highlight the global maximum with a larger red circle
    if global_max_position is not None:
        plt.scatter(global_max_position[1], global_max_position[0], color='red', s=300, edgecolor='white', label='Global Maximum', zorder=6)

    # Add title and axis labels
    plt.title(f'Heatmap with {p} Random Starting Points', fontsize=16)
    plt.xlabel('X Axis', fontsize=14)
    plt.ylabel('Y Axis', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.grid(False)
    
    # Show the heatmap
    plt.show()
    pass

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: ./Lab2_mc_mem1 <print_option> <N> <p> <seed>")
        sys.exit(1)

    print_option = sys.argv[1]
    N = int(sys.argv[2])
    p = int(sys.argv[3])
    seed = int(sys.argv[4])

    matrix(print_option, N, p, seed)
