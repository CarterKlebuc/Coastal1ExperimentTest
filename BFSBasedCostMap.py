from collections import deque
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import heapq
from PIL import Image
from matplotlib.widgets import Button
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle
import math
from functools import partial
import pygame
import ParticleFilterFunctions as PF
import InterpolationTest as IT
from scipy.optimize import curve_fit

import matplotlib.cbook as cbook
import matplotlib.image as image

# Define directions for movement: up, down, left, right
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, -1), (-1, 1)]
diagonal_directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]


def calculate_map_entropy(map, world_size, growth="Linear"):
    information_map = np.zeros((world_size[0], world_size[1]))
    # Initial Information Map Generation without accounting for non-wall obstacles
    for i in range(world_size[0]):
        for j in range(world_size[1]):
            if map[i, j] == 1:
                information_map[i, j] = float('inf')  # Walls have high information content
            else:
                # Distance to nearest wall
                distance_to_wall = calculate_closest_wall_distance(map, [i, j], growth)  #* 2
                if distance_to_wall > 1.0:
                    distance_to_wall *= (2 * distance_to_wall)
                information_map[i, j] = distance_to_wall  # Inverse of distance to wall

    return information_map


def calculate_closest_wall_distance(matrix, start, growth='Linear'):
    rows = len(matrix)
    cols = len(matrix[0])

    # Define the directions for moving in the grid (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Queue for BFS: stores the position and the current distance
    queue = deque([(start[0], start[1], 0)])

    # Set to track visited cells
    visited = set()
    visited.add((start[0], start[1]))
    time = 1
    # BFS to find the closest '1'
    while queue:
        x, y, dist = queue.popleft()

        # If we found a '1', return the distance
        if matrix[x][y] == 1:
            #print("Distance Returned!")
            return dist

        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Check if the new position is within bounds and not visited
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                visited.add((nx, ny))
                time = time + 1
                if growth == 'Linear':
                    queue.append((nx, ny, dist + 1))
                elif growth == "Exponential":
                    # Exponential growth (growth rate is 1 but can be changed)
                    queue.append((nx, ny, dist * np.exp(1 * time)))
                elif growth == "Cubic":
                    # Cubic growth
                    queue.append((nx, ny, dist * dist * dist))
                elif growth == "Quadratic":
                    # Quadratic Growth where the 1's are separate quadratic growth rates which can be adjusted
                    queue.append((nx, ny, (1 * dist * dist) + (1 * dist) + 1))

    # If no '1' is found, return -1 (assuming there's always a '1' in the matrix)
    return -1


def bfs_travel_cost_map(occupancy_map, start, end):
    rows, cols = len(occupancy_map), len(occupancy_map[0])

    # Initialize cost map with infinity for all cells
    travel_cost_map = [[float('inf')] * cols for _ in range(rows)]

    # Create a queue for BFS and add the starting point
    queue = deque([(start[0], start[1])])
    travel_cost_map[start[0]][start[1]] = 0  # Cost of reaching the start is 0

    # BFS loop
    while queue:
        x, y = queue.popleft()
        #print("Current X: " + str(x))
        #print("Current Y: " + str(y))
        # Explore neighbors
        for dx, dy in DIRECTIONS:
            if not (dx, dy) in diagonal_directions:
                nx, ny = x + dx, y + dy

                # Check if the neighbor is within bounds and not an obstacle
                if 0 <= nx < rows and 0 <= ny < cols and occupancy_map[nx][ny] == 0:
                    # Calculate new cost to reach neighbor
                    new_cost = travel_cost_map[x][y] + 1

                    # If the new cost is lower than the current cost in the travel_cost_map, update it
                    if new_cost < travel_cost_map[nx][ny]:
                        travel_cost_map[nx][ny] = new_cost
                        queue.append((nx, ny))
            else:
                nx, ny = x + dx, y + dy

                # Check if the neighbor is within bounds and not an obstacle
                if 0 <= nx < rows and 0 <= ny < cols and occupancy_map[nx][ny] == 0:
                    # Calculate new cost to reach neighbor
                    new_cost = travel_cost_map[x][y] + 1

                    # If the new cost is lower than the current cost in the travel_cost_map, update it
                    if new_cost < travel_cost_map[nx][ny]:
                        travel_cost_map[nx][ny] = new_cost
                        queue.append((nx, ny))

    return travel_cost_map


def dijkstra_cost_map(travel_cost_map, info_cost_map, denied_map, start, end):
    rows, cols = len(travel_cost_map), len(travel_cost_map[0])

    # Initialize the distance map with infinity for all cells
    dist_map = [[float('inf')] * cols for _ in range(rows)]
    dist_map[start[1]][start[0]] = travel_cost_map[start[1]][start[0]]

    # Priority queue (min-heap), starts with the initial position (cost, x, y)
    pq = [(travel_cost_map[start[1]][start[0]], start[0], start[1])]

    # Keep track of the path
    came_from = [[None for _ in range(cols)] for _ in range(rows)]

    while pq:
        current_dist, x, y = heapq.heappop(pq)

        # If we've reached the end point, stop the search
        if (x, y) == end:
            break

        # Explore neighbors
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy

            # Ensure the neighbor is within bounds
            if 0 <= nx < rows - 1 and 0 <= ny < cols - 1 and travel_cost_map[ny][nx] != float('inf') and \
                    travel_cost_map[ny][nx] != float('nan') and info_cost_map[ny][nx] != float('inf') and \
                    info_cost_map[ny][nx] != float('nan'):
                if not (dx, dy) in diagonal_directions:
                    new_dist = current_dist + travel_cost_map[ny][nx] + info_cost_map[ny][nx] + denied_map[ny][
                        nx]  # Accumulate the cost to reach this neighbor
                else:
                    new_dist = current_dist + (travel_cost_map[ny][nx] * np.sqrt(2)) + info_cost_map[ny][nx] + \
                               denied_map[ny][nx]  # Accumulate the cost to reach this neighbor
                # If the new cost is lower, update the dist_map and push to the priority queue
                if new_dist < dist_map[nx][ny]:
                    dist_map[nx][ny] = new_dist
                    came_from[nx][ny] = (x, y)  # Track the path
                    heapq.heappush(pq, (new_dist, nx, ny))

    # Reconstruct the path from end to start
    path = []
    cx, cy = end
    if dist_map[end[0]][end[1]] == float('inf'):
        return [], float('inf')  # No valid path

    while (cx, cy) != start:
        path.append((cx, cy))
        #print("Came From: " + str(came_from[cx][cy]))
        cx, cy = came_from[cx][cy]

    path.append(start)  # Add the start point at the end
    path.reverse()  # Reverse to get the path from start to end

    return path, dist_map[end[0]][end[1]]  # Return the path and the cost of the path


def create_experiment_new_multi_exp(start, end, world_size, occupancy_grid, k_value_array, denied_map, figure_label):
    number_of_inputs = len(k_value_array)
    paths = []
    path_properties = []
    properties = []
    info_cost_map_storage = []
    travel_cost_map_storage = []
    total_cost_map_storage = []
    robot_path_storage = []
    starting_point_storage = []

    info_cost_map = np.multiply(calculate_map_entropy(occupancy_grid, world_size, "Linear"), k_value_array[0])

    f, (ax3) = plt.subplots(1, 1)
    displayed_cost_map = ax3.imshow(occupancy_grid, 'Grays')#, extent=(0, 33, 0, 33))

    #plt.title("Please select the desired goal point")
    #goal_point = np.asarray(plt.ginput(1))
    end = end

    #plt.title("Please select the desired start points")
    #start_point = np.asarray(plt.ginput(number_of_inputs))

    start_points = []
    for i in range(0, number_of_inputs):
        start_points.append(start)

    paths_array = []
    start = start_points[0]

    ax3.set_aspect('equal')  # Maintain equal scaling
    mpl.rcParams['text.usetex'] = True

    for i in range(0, number_of_inputs):
        # Creates the respective travel and info cost maps for each input
        #print("Current K Value at " + str(i) + " is: " + str(k_value_array[i]))
        travel_cost_map = np.multiply(bfs_travel_cost_map(occupancy_grid, [start_points[i][1], start_points[i][0]], [end[1], end[0]]), k_value_array[i])
        info_cost_map = np.multiply(calculate_map_entropy(occupancy_grid, world_size, "Linear"), (1 - k_value_array[i]))
        #print("Info Cost Map: " + str(info_cost_map))

        travel_cost_map_storage.append(travel_cost_map)
        info_cost_map_storage.append(info_cost_map)

        # Creates and appends the total map to the storage
        total_map = np.add(travel_cost_map, info_cost_map)
        total_cost_map_storage.append(total_map)

        # The path between the start and goal points is generated here
        path, total_cost = dijkstra_cost_map(travel_cost_map, info_cost_map, denied_map, start_points[i], end)
        paths_array.append(path)

        #print("Path: " + str(path))

    # Handles the color bar (representing total cost) and title of the figure
    #plt.colorbar(total_cost_map_storage[0], ax=ax3)
    #plt.xticks(np.arange(world_size[0]))
    #plt.yticks(np.arange(world_size[1]))

    plt.title("UH Map Experiment ($\\textit{k}$ = 0.5)", fontdict={'fontsize': 32})
    plt.xlabel("Cells [m]", fontdict={'fontsize': 32})
    plt.ylabel("Cells [m]", fontdict={'fontsize': 32})
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    '''
    # Places total cost on each cell in the map grid
    for i in range(world_size[0]):
        for j in range(world_size[1]):
            text = plt.text(j, i, round(total_cost_map_storage[0][i, j], 2), ha="center", va="center", color="black", fontsize=3.0)
    '''
    path_labels = ["Shortest Path (k = 1)", "Balanced Path (k = 0.5)", "Information Path (k = 0)"]
    # Path is converted into separate arrays of x and y values
    colors = ["red", "green", "blue"]
    starting_point = ax3.scatter(start_points[0][0], start_points[0][1], c='white', marker='o', label='Start',
                                 zorder=10)
    for i in range(0, number_of_inputs):
        path_x_vals = []
        path_y_vals = []
        print("Path Length: " + str(len(paths_array[i])))
        for point in paths_array[i]:
            path_x_vals.append(point[0])
            path_y_vals.append(point[1])

        # Plots the start point, goal point, and the path between these two points
        robot_path = ax3.plot(path_x_vals, path_y_vals, c=colors[i], label=path_labels[i])
        #starting_point = ax3.scatter(start_points[i][0], start_points[i][1], c='white', marker='o', label='Start',
        #                          zorder=10)
        robot_path_storage.append(robot_path)
        starting_point_storage.append(starting_point)

    ax3.scatter(end[0], end[1], c='white', marker='x', label='Goal', zorder=10)

    # ax3.legend()
    #axes = plt.axes([0.80, 0.5, 0.15, 0.075])
    #axes2 = plt.axes([0.80, 0.3, 0.15, 0.075])
    #axes3 = plt.axes([0.80, 0.2, 0.15, 0.075])
    #axes4 = plt.axes([0.80, 0.000001, 0.15, 0.075])
    #bnext = Button(axes, 'Display First Cost Map', color="yellow")

    # Add Wall Coordinates as Red Circular Patches Here!
    patches = [(17.6, 4.52), (21.45, 4.52), (22.42, 5.44), (17.55, 7.42), (23.42, 8.44), (21.48, 8.44), (23.45, 6.44), (22.61, 6.42), (22.45, 10.50), (22.45, 12.45), ]
    #for point in paths_array[0]:
    #    circle = Circle((point[0], point[1]), 0.1, facecolor=(1, 0, 0),
    #                    edgecolor=(1, 0, 0), linewidth=3, alpha=1.0)
    #    ax3.add_patch(circle)

    walls = [
        # Island 1
        (17.4, 4.52, 3.85, 0.1),
        (17.4, 4.42, 0.1, 3),
        (21.3, 4.52, 0.1, 1),
        (17.4, 7.42, 4, 0.1),
        (21.42, 5.44, 0.97, 0.1),
        (22.42, 5.50, 0.1, 0.89),
        (22.42, 6.47, 0.86, 0.1),
        (23.45, 6.47, 0.1, 1.75),
        (21.53, 7.50, 0.1, 0.89),
        (21.53, 8.39, 1.92, 0.1),

        # Island 2
        (16.53, 8.55, 1.86, 0.1),
        (18.39, 8.58, 0.1, 0.87),
        (18.47, 9.45, 1.9, 0.1),
        (20.39, 9.58, 0.1, 0.92),
        (20.39, 10.50, 1.92, 0.1),
        (22.31, 10.55, 0.1, 1.9),
        (16.55, 8.58, 0.1, 1.84),
        (13.58, 10.47, 2.97, 0.1),
        (13.58, 10.47, 0.1, 2.9),
        (13.44, 13.37, 2.86, 0.1),
        (16.50, 13.50, 0.1, 0.87),
        (16.50, 14.37, 2.7, 0.1),
        (19.56, 11.53, 0.1, 2.84),
        (19.56, 11.53, 0.89, 0.1),
        (20.47, 11.50, 0.1, 0.84),
        (20.47, 12.47, 2, 0.1),

        # Island 3
        (12.55, 16.50, 2.03, 0.1),
        (12.55, 16.50, 0.1, 1.89),
        (12.47, 18.42, 1, 0.1),
        (13.53, 18.42, 0.1, 1),
        (13.47, 19.40, 1, 0.1),
        (14.47, 19.53, 0.1, 1),
        (14.58, 20.40, 3, 0.1),
        (17.50, 20.53, 0.1, 1),
        (17.50, 21.37, 1, 0.1),
        (14.45, 16.58, 0.1, 1),
        (14.45, 17.53, 2, 0.1),
        (16.47, 17.56, 0.1, 1),
        (16.50, 18.45, 2, 0.1),
        (18.47, 18.56, 0.1, 3),

        # Island 4
        (23.56, 12.64, 2, 0.1),
        (23.56, 12.64, 0.1, 3),
        (25.56, 12.64, 0.1, 3),
        (25.56, 15.64, 1, 0.1),
        (26.56, 15.64, 0.1, 2),
        (26.56, 17.64, 1, 0.1),
        (27.56, 17.64, 0.1, 2),
        (21.50, 15.58, 2, 0.1),
        (21.50, 15.58, 0.1, 5),
        (20.50, 20.58, 1, 0.1),
        (20.50, 20.58, 0.1, 2),
        (20.50, 22.58, 2, 0.1),
        (22.48, 21.40, 0.1, 1),
        (22.48, 21.40, 2, 0.1),
        (24.45, 20.45, 0.1, 1),
        (24.45, 20.45, 2, 0.1),
        (26.48, 19.48, 1, 0.1),
        (26.48, 19.48, 0.1, 1),

        # Island 5
        (11.52, 20.56, 3, 0.1),
        (11.52, 20.56, 0.1, 1),
        (10.52, 21.56, 1, 0.1),
        (10.52, 21.56, 0.1, 3),
        (10.52, 24.56, 2, 0.1),
        (12.52, 24.56, 0.1, 1),
        (12.52, 25.56, 4, 0.1),
        (16.52, 25.56, 0.1, 1),
        (16.52, 26.56, 2, 0.1),
        (18.52, 26.56, 0.1, 1),
        (18.52, 27.56, 2, 0.1),
        (20.52, 27.56, 0.1, 1),
        (20.52, 28.56, 2, 0.1),
        (14.52, 20.56, 0.1, 1),
        (14.52, 21.56, 3, 0.1),
        (17.52, 21.56, 0.1, 1),
        (17.52, 22.56, 2, 0.1),
        (19.52, 22.56, 0.1, 1),
        (19.52, 23.56, 2, 0.1),
        (21.52, 23.56, 0.1, 3),
        (21.52, 26.56, 1, 0.1),
        (22.52, 26.56, 0.1, 2),

        # Island 6
        (5.58, 6.61, 3, 0.1),
        (8.58, 6.61, 0.1, 1),
        (8.58, 7.61, 2, 0.1),
        (10.58, 7.61, 0.1, 1),
        (10.58, 8.61, 2, 0.1),
        (12.58, 8.61, 0.1, 4),
        (5.58, 6.61, 0.1, 4),
        (5.58, 10.61, 2, 0.1),
        (7.58, 10.61, 0.1, 1),
        (7.58, 11.61, 3, 0.1),
        (10.58, 11.61, 0.1, 1),
        (10.58, 12.61, 2, 0.1),

        # Island 7
        (26.59, 2.47, 1, 0.1),
        (27.59, 2.47, 0.1, 1),
        (27.59, 3.47, 4, 0.1),
        (31.59, 3.47, 0.1, 1),
        (31.59, 4.47, 1, 0.1),
        (32.58, 4.47, 0.1, 1),
        (32.57, 5.47, 1, 0.1),
        (33.57, 5.47, 0.1, 4),
        (26.59, 2.47, 0.1, 6),
        (25.59, 8.47, 1, 0.1),
        (25.59, 8.47, 0.1, 3),
        (25.59, 11.47, 2, 0.1),
        (27.59, 11.47, 0.1, 1),
        (27.58, 12.47, 3, 0.1),
        (30.58, 12.47, 0.1, 1),
        (30.58, 13.47, 1, 0.1),
        (32.58, 9.50, 1, 0.1),
        (32.58, 9.50, 0.1, 2),
        (31.55, 11.55, 1, 0.1),
        (31.55, 11.55, 0.1, 2),

        # Island 8
        (3.60, 12.45, 5, 0.1),
        (8.44, 12.58, 0.1, 1),
        (8.44, 13.58, 2, 0.1),
        (10.44, 13.58, 0.1, 3),
        (9.54, 16.53, 1, 0.1),
        (9.54, 16.53, 0.1, 4),
        (1.54, 19.45, 1, 0.1),
        (2.54, 19.45, 0.1, 1),
        (2.54, 20.45, 1, 0.1),
        (3.54, 20.45, 0.1, 1),
        (3.54, 21.45, 2, 0.1),
        (5.54, 21.45, 0.1, 1),
        (5.54, 22.45, 1, 0.1),
        (6.54, 22.45, 0.1, 1),
        (6.54, 23.45, 1, 0.1),
        (7.54, 21.50, 1, 0.1),
        (7.54, 21.50, 0.1, 2),
        (8.46, 20.50, 1, 0.1),
        (8.46, 20.50, 0.1, 1),
        (2.57, 15.58, 1, 0.1),
        (3.54, 12.61, 0.1, 3),
        (2.59, 15.64, 0.1, 2),
        (1.57, 17.64, 1, 0.1),
        (1.57, 17.64, 0.1, 2),

        # Island 9
        (29.61, 0.52, 4, 0.1),
        (33.61, 0.52, 0.1, 2),
        (29.61, 0.52, 0.1, 1),
        (29.61, 1.52, 1, 0.1),
        (30.61, 1.52, 0.1, 1),
        (30.61, 2.52, 3, 0.1),

        # Island 10
        (8.54, 0.60, 5, 0.1),
        (13.54, 0.60, 0.1, 2),
        (8.54, 0.60, 0.1, 2),
        (7.59, 2.52, 1, 0.1),
        (7.59, 2.52, 0.1, 2),
        (7.59, 4.52, 1, 0.1),
        (8.59, 4.52, 0.1, 1),
        (8.59, 5.52, 3, 0.1),
        (11.57, 4.50, 1, 0.1),
        (11.57, 4.50, 0.1, 1),
        (12.65, 2.50, 1, 0.1),
        (12.65, 2.50, 0.1, 2),

        # Island 11
        (0.61, 0.52, 2, 0.1),
        (0.61, 0.52, 0.1, 6),
        (0.61, 6.52, 2, 0.1),
        (2.56, 3.44, 1, 0.1),
        (2.56, 3.44, 0.1, 3),
        (2.51, 0.55, 0.1, 2),
        (2.53, 2.55, 1, 0.1),
        (3.45, 2.47, 0.1, 1),

        # Island 12
        (8.56, 29.62, 5, 0.1),
        (13.56, 29.62, 0.1, 2),
        (12.54, 31.45, 1, 0.1),
        (12.54, 31.45, 0.1, 1),
        (8.56, 29.62, 0.1, 3),
        (8.56, 32.62, 4, 0.1),

        # Island 13 (Final Island!)
        (30.58, 28.59, 3, 0.1),
        (30.58, 28.59, 0.1, 2),
        (30.58, 30.59, 1, 0.1),
        (31.58, 30.59, 0.1, 1),
        (31.58, 31.59, 1, 0.1),
        (32.58, 30.53, 1, 0.1),
        (32.58, 30.53, 0.1, 1),
        (33.47, 28.59, 0.1, 2)
    ]
    #for wall in walls:
    #    rect = plt.Rectangle((wall[0], wall[1]), wall[2], wall[3], color="blue")
    #    ax3.add_patch(rect)

    # Setup Corners for Particle Filter Feature Matching
    corners = list(set([(wall[0], wall[1]) for wall in walls] + [(wall[0] + wall[2], wall[1] + wall[3]) for wall in walls]))

    print(str(IT.interpolate_coordinates(paths_array[0])))

    #coordinate_paths = [IT.interpolate_coordinates(paths_array[0]), IT.interpolate_coordinates(paths_array[1]), IT.interpolate_coordinates(paths_array[2])]
    coordinate_paths = paths_array

    my_robot = PF.Quadcopter(paths_array[0][0][0], paths_array[0][1][1])
    return_array = PF.run_robot_path(my_robot, coordinate_paths[0], corners)
    uncertainty_array = return_array[0]
    position_array = return_array[1]
    #for point in position_array:
    #    circle = Circle((point[0], point[1]), 0.1, facecolor=(1, 0, 0),
    #                    edgecolor=(1, 0, 0), linewidth=3, alpha=1.0)
    #    ax3.add_patch(circle)


    #plt.figure(2)
    #for i in range(len(uncertainty_array)):
    #    plt.scatter(i, uncertainty_array[i], color='red')
    #shortest_path = plt.plot(uncertainty_array, color='red', label='Plot 1')
    my_robot_2 = PF.Quadcopter(paths_array[1][0][0], paths_array[1][1][1])
    return_array_2 = PF.run_robot_path(my_robot, coordinate_paths[1], corners)
    #for point in return_array_2[1]:
    #    circle = Circle((point[0], point[1]), 0.1, facecolor=(1, 0, 0),
    #                    edgecolor=(1, 0, 0), linewidth=3, alpha=1.0)
    #    ax3.add_patch(circle)
    uncertainty_array_2 = return_array_2[0]
    position_array_2 = return_array_2[1]
    #for i in range(len(uncertainty_array_2)):
    #    plt.scatter(i, uncertainty_array_2[i], color='green')
    #balanced_path = ax3.plot(uncertainty_array_2, color='green', label="Plot 2")
    my_robot_3 = PF.Quadcopter(paths_array[2][0][0], paths_array[2][1][1])
    return_array_3 = PF.run_robot_path(my_robot, coordinate_paths[2], corners)
    #for point in return_array_3[1]:
    #    circle = Circle((point[0], point[1]), 0.1, facecolor=(1, 0, 0),
    #                    edgecolor=(1, 0, 0), linewidth=3, alpha=1.0)
    #    ax3.add_patch(circle)
    uncertainty_array_3 = return_array_3[0]
    position_array_3 = return_array_3[1]
    #for i in range(len(uncertainty_array_3)):
    #    plt.scatter(i, uncertainty_array_3[i], color='blue')


    #information_path = ax3.plot(uncertainty_array_3, color='blue', label="Plot 3")

    #patches = [shortest_path, balanced_path, information_path]

    displayed_cost_map = ax3.imshow(total_cost_map_storage[1])
    ax3.legend(loc="upper left", fontsize=12)
    plt.show()

    print("Uncertainty Array 1: " + str(uncertainty_array))
    print("Uncertainty Array 2: " + str(uncertainty_array_2))
    print("Uncertainty Array 3: " + str(uncertainty_array_3))

    exported_position_values = []

    for i in range(len(position_array_3)):
        one_val = 0
        two_val = 0
        three_val = 0
        if i < len(position_array):
            one_val = position_array[i]
        if i < len(position_array_2):
            two_val = position_array_2[i]
        three_val = position_array_3[i]

        exported_position_values.append([one_val, two_val, three_val])

    import csv

    with open('ExportedPositionValuesExperiment1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(exported_position_values)



    #callback_with_args = partial(display_first_cost_map, selected_ax=ax3, displayed_cost_map=total_cost_map_storage[0], paths=robot_path_storage, start_points=starting_point_storage, k_values = k_value_array)

    '''
    
        Add moving robot function here!
        (Done!) For the first path in the paths, gather all of the points:
            For each point: plot a scatter plot at that point (Each point will represent a motion step!)
        
        (Done!) Once that is done:
            Copy the feature matching algorithm made yesterday and test it out to see how it will function with steps of 
            one (may need to adjust any gains present in the algorithm)
                - You should largely adjust the motion model as the sensor model should be the same between this and the
                feature matching program
        
        (Save for later!) Then modify the walls so there is a better fit of the islands (of the UH map)
        
        (Done) Then implement the corner detection algorithm found present in the feature matching algorithm
        
        (Done) Then implement the particle filter algorithm (in a separate file) and determine if it will work for the
        first path derived in step 1
            - Test this based on a graph of the uncertainty over the elapsed time (in iterations!)
            
        Replace the line plot with a very accurate line of best fit along with using the scatter plot
        
        Work on experiment 2 where there should be a very short but open path and a very long but information
        rich path (that will be your ideal case while the UH map will be a more realistic scenario)
        
        
    
    
    '''











    # Connect the button to the partial function
    #bnext.on_clicked(callback_with_args)

    #bnext2 = Button(axes2, 'Display Second Cost Map', color="yellow")

    #callback_with_args2 = partial(display_second_cost_map, selected_ax=ax3, displayed_cost_map=total_cost_map_storage[1], paths=robot_path_storage, start_points=starting_point_storage, k_values = k_value_array)

    # Connect the button to the partial function
    #bnext2.on_clicked(callback_with_args2)

    #bnext3 = Button(axes3, 'Display Third Cost Map', color="yellow")

    #callback_with_args3 = partial(display_third_cost_map, selected_ax=ax3,isplayed_cost_map=total_cost_map_storage[2], paths=robot_path_storage, start_points=starting_point_storage, k_values = k_value_array)

    # Connect the button to the partial function
    #bnext3.on_clicked(callback_with_args3)

    #bnext4 = Button(axes4, 'Display All Lines', color="yellow")

    #callback_with_args4 = partial(display_all_lines, selected_map=occupancy_grid, selected_ax=ax3, paths=robot_path_storage,start_points=starting_point_storage)

    # Connect the button to the partial function
    #bnext4.on_clicked(callback_with_args4)

def display_first_cost_map(event, selected_ax, displayed_cost_map, paths, start_points, k_values):
    displayed_cost_map = selected_ax.imshow(displayed_cost_map)
    for point in start_points:
        #print("Point: " + str(point))
        point.set_visible(False)
    for path in paths:
        #print("Path: " + str(path))
        path[0].set_linestyle('None')
    start_points[0].set_visible(True)
    paths[0][0].set_linestyle('solid')
    third_k_val_string = "Path with K Value: " + str(k_values[0])
    plt.title(third_k_val_string)
    plt.show()

def display_second_cost_map(event, selected_ax, displayed_cost_map, paths, start_points, k_values):
    displayed_cost_map = selected_ax.imshow(displayed_cost_map)
    for point in start_points:
        #print("Point: " + str(point))
        point.set_visible(False)
    for path in paths:
        #print("Path: " + str(path))
        path[0].set_linestyle('None')
    start_points[1].set_visible(True)
    paths[1][0].set_linestyle('solid')
    third_k_val_string = "Path with K Value: " + str(k_values[1])
    plt.title(third_k_val_string)
    plt.show()

def display_third_cost_map(event, selected_ax, displayed_cost_map, paths, start_points, k_values):
    displayed_cost_map = selected_ax.imshow(displayed_cost_map)
    for point in start_points:
        #print("Point: " + str(point))
        point.set_visible(False)
    for path in paths:
        #print("Path: " + str(path))
        path[0].set_linestyle('None')
    start_points[2].set_visible(True)
    paths[2][0].set_linestyle('solid')
    third_k_val_string = "Path with K Value: " + str(k_values[2])
    plt.title(third_k_val_string)
    plt.show()

def display_all_lines(event, selected_ax, selected_map, paths, start_points):
    selected_ax.imshow(selected_map, 'Grays')
    for point in start_points:
        #print("Point: " + str(point))
        point.set_visible(False)
        point.set_visible(True)
    for path in paths:
        #print("Path: " + str(path))
        path[0].set_linestyle('None')
        path[0].set_linestyle('solid')
    plt.title("Displaying All Lines")
    plt.show()


def create_experiment_uniform_cost(start, end, world_size, occupancy_grid, travel_weight, info_weight, denied_map, figure_label):
    number_of_inputs = 10
    paths = []
    path_properties = []
    properties = []
    info_cost_map = np.multiply(calculate_map_entropy(occupancy_grid, world_size, "Linear"), info_weight)

    f, (ax3) = plt.subplots(1, 1)
    displayed_cost_map = ax3.imshow(info_cost_map)#, extent=(0, 33, 0, 33))
    img = np.asarray(Image.open('UH_Sat_View.png'))
    #ax3.imshow(img, extent=(0, 33, 0, 33))

    plt.title("Please select the desired goal point")
    goal_point = np.asarray(plt.ginput(1))
    end = (int(round(goal_point[0][0])), int(round(goal_point[0][1])))

    plt.title("Please select the desired start points")
    start_point = np.asarray(plt.ginput(number_of_inputs))
    #start = (int(round(start_point[0][0])), int(round(start_point[0][1])))

    print("Start : " + str(start) + " Goal : " + str(end))



    start_points = []
    for i in range(0, number_of_inputs):
        start_points.append((int(round(start_point[i][0])), int(round(start_point[i][1]))))

    paths_array = []
    start = start_points[0]

    #displayed_cost_map = ax3.imshow(info_cost_map, extent=(0, 10, 0, 10))


    ax3.set_aspect('equal')  # Maintain equal scaling

    for i in range(0, number_of_inputs):
        travel_cost_map = np.multiply(bfs_travel_cost_map(occupancy_grid, [start_points[i][1], start_points[i][0]], [end[1], end[0]]), travel_weight)
        total_map = np.add(travel_cost_map, info_cost_map)

        # The path between the start and goal points is generated here
        path, total_cost = dijkstra_cost_map(travel_cost_map, info_cost_map, denied_map, start_points[i], end)
        paths_array.append(path)

        print("Path: " + str(path))



    # Handles the color bar (representing total cost) and title of the figure
    plt.colorbar(displayed_cost_map, ax=ax3)
    plt.xticks(np.arange(world_size[0]))
    plt.yticks(np.arange(world_size[1]))

    plt.title("Experiment 1")


    # Places total cost on each cell in the map grid
    for i in range(world_size[0]):
         for j in range(world_size[1]):
             text = plt.text(j, i, round(total_map[i, j], 2), ha="center", va="center", color="black", fontsize=3.0)



    # Path is converted into separate arrays of x and y values
    for i in range(0, number_of_inputs):
        path_x_vals = []
        path_y_vals = []
        for point in paths_array[i]:
            path_x_vals.append(point[0])
            path_y_vals.append(point[1])

        # Plots the start point, goal point, and the path between these two points
        new_plot = ax3.plot(path_x_vals, path_y_vals, c='red')
        new_scatter = ax3.scatter(start_points[i][0], start_points[i][1], c='green', marker='o', label='Start', zorder=10)

    ax3.scatter(end[0], end[1], c='white', marker='x', label='Goal', zorder=10)

    #ax3.legend()

    plt.show()

def create_experiment_interface(start, end, world_size, occupancy_grid, travel_weight, info_weight, denied_map, figure_label):
    number_of_inputs = 1
    paths = []
    path_properties = []
    properties = []
    info_cost_map = np.multiply(calculate_map_entropy(occupancy_grid, world_size, "Linear"), info_weight)

    f, (ax3) = plt.subplots(1, 1)
    displayed_cost_map = ax3.imshow(info_cost_map)#, extent=(0, 33, 0, 33))
    img = np.asarray(Image.open('UH_Sat_View.png'))
    #ax3.imshow(img, extent=(0, 33, 0, 33))

    #plt.title("Please select the desired goal point")
    #goal_point = np.asarray(plt.ginput(1))
    #end = (int(round(goal_point[0][0])), int(round(goal_point[0][1])))
    #print("end: " + str(end))

    #plt.title("Please select the desired start points")
    #start_point = np.asarray(plt.ginput(number_of_inputs))
    #start = (int(round(start_point[0][0])), int(round(start_point[0][1])))

    print("Start : " + str(start) + " Goal : " + str(end))



    start_points = []
    #for i in range(0, number_of_inputs):
    #    start_points.append((int(round(start_point[i][0])), int(round(start_point[i][1]))))
    start_points.append(start)

    paths_array = []
    start = start_points[0]

    #displayed_cost_map = ax3.imshow(info_cost_map, extent=(0, 10, 0, 10))


    ax3.set_aspect('equal')  # Maintain equal scaling

    for i in range(0, 1):
        travel_cost_map = np.multiply(bfs_travel_cost_map(occupancy_grid, [start_points[i][1], start_points[i][0]], [end[1], end[0]]), travel_weight)
        total_map = np.add(travel_cost_map, info_cost_map)

        # The path between the start and goal points is generated here
        path, total_cost = dijkstra_cost_map(travel_cost_map, info_cost_map, denied_map, start_points[i], end)
        paths_array.append(path)

        print("Path: " + str(path))



    # Handles the color bar (representing total cost) and title of the figure
    plt.colorbar(displayed_cost_map, ax=ax3)
    plt.xticks(np.arange(world_size[0]))
    plt.yticks(np.arange(world_size[1]))

    plt.title("Experiment 1")


    # Places total cost on each cell in the map grid
    for i in range(world_size[0]):
         for j in range(world_size[1]):
             text = plt.text(j, i, round(total_map[i, j], 2), ha="center", va="center", color="black", fontsize=3.0)



    # Path is converted into separate arrays of x and y values
    for i in range(0, number_of_inputs):
        path_x_vals = []
        path_y_vals = []
        for point in paths_array[i]:
            path_x_vals.append(point[0])
            path_y_vals.append(point[1])

        # Plots the start point, goal point, and the path between these two points
        new_plot = ax3.plot(path_x_vals, path_y_vals, c='red')
        new_scatter = ax3.scatter(start_points[i][0], start_points[i][1], c='green', marker='o', label='Start', zorder=10)

    ax3.scatter(end[0], end[1], c='white', marker='x', label='Goal', zorder=10)

    #ax3.legend()

    # Add Wall Coordinates as Red Circular Patches Here!
    patches = [[5, 5], [6, 6], [20, 20]]
    for patch in patches:
        circle = Circle((patch[0], patch[1]), 1, facecolor='none',
                        edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        ax3.add_patch(circle)

    plt.show()

    print(str(info_cost_map))

def return_total_map(start, world_size, occupancy_grid, travel_weight, info_weight, denied_map):
    pass