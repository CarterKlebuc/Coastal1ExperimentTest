import numpy as np


def interpolate_coordinates(points, num_interpolations=5):
    """
    Interpolates between each pair of consecutive points.

    :param points: List of tuples [(x1, y1), (x2, y2), ...]
    :param num_interpolations: Number of points to insert between each pair.
    :return: List of tuples with interpolated points.
    """
    if not points or len(points) < 2:
        return points

    interpolated_points = []

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        interpolated_points.append((x1, y1))

        # Generate interpolated points
        for j in range(1, num_interpolations + 1):
            t = j / (num_interpolations + 1)
            new_x = x1 + t * (x2 - x1)
            new_y = y1 + t * (y2 - y1)
            interpolated_points.append((new_x, new_y))

    # Append the last point
    interpolated_points.append(points[-1])

    return interpolated_points
