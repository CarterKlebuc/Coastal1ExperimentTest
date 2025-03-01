from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import heapq
from PIL import Image
from matplotlib.widgets import Button
from matplotlib.markers import MarkerStyle
import math
from functools import partial

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


def bfs_travel_cost_map(occupancy_map, start, end=None):
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
        print("Came From: " + str(came_from[cx][cy]))
        cx, cy = came_from[cx][cy]

    path.append(start)  # Add the start point at the end
    path.reverse()  # Reverse to get the path from start to end

    return path, dist_map[end[0]][end[1]]  # Return the path and the cost of the path


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

    plt.title("Travel + Information Cost Map")


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

    print(str(info_cost_map))

def return_total_map(start, world_size, occupancy_grid, travel_weight, info_weight):
    info_cost_map = np.multiply(calculate_map_entropy(occupancy_grid, world_size, "Linear"), info_weight)


    travel_cost_map = np.multiply(bfs_travel_cost_map(occupancy_grid, [start[1], start[0]]), travel_weight)
    total_map = np.add(travel_cost_map, info_cost_map)

    #f, (ax3) = plt.subplots(1, 1)
    #print(str(total_map))
    #ax3.imshow(total_map)
    #plt.colorbar(total_map, ax=ax3)
    #plt.show()
    return total_map

def return_info_map(start, world_size, occupancy_grid, travel_weight, info_weight):
    info_cost_map = np.multiply(calculate_map_entropy(occupancy_grid, world_size, "Linear"), info_weight)


    #travel_cost_map = np.multiply(bfs_travel_cost_map(occupancy_grid, [start[1], start[0]]), travel_weight)
    #total_map = np.add(travel_cost_map, info_cost_map)

    #f, (ax3) = plt.subplots(1, 1)
    #print(str(total_map))
    #ax3.imshow(total_map)
    #plt.colorbar(total_map, ax=ax3)
    #plt.show()
    return info_cost_map
