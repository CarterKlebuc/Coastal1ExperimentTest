import heapq

# Define movement directions: up, down, left, right
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def dijkstra_cost_map(cost_map, start, end):
    rows, cols = len(cost_map), len(cost_map[0])

    # Initialize the distance map with infinity for all cells
    dist_map = [[float('inf')] * cols for _ in range(rows)]
    dist_map[start[0]][start[1]] = cost_map[start[0]][start[1]]

    # Priority queue (min-heap), starts with the initial position (cost, x, y)
    pq = [(cost_map[start[0]][start[1]], start[0], start[1])]

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
            if 0 <= nx < rows and 0 <= ny < cols:
                new_dist = current_dist + cost_map[nx][ny]  # Accumulate the cost to reach this neighbor

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
        cx, cy = came_from[cx][cy]

    path.append(start)  # Add the start point at the end
    path.reverse()  # Reverse to get the path from start to end

    return path, dist_map[end[0]][end[1]]  # Return the path and the cost of the path


# Example usage:
cost_map = [
    [1, 2, 3, 4, 1],
    [4, 1, 1, 3, 1],
    [2, 3, 1, 1, 1],
    [3, 1, 2, 1, 2],
    [2, 2, 1, 3, 1]
]

start = (0, 0)
end = (4, 4)

path, total_cost = dijkstra_cost_map(cost_map, start, end)

# Display the path and total cost
print("Path:", path)
print("Total Cost:", total_cost)
