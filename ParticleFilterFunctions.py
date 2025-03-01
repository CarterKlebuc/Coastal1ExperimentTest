import numpy as np
import random
import math


# Robot class
class Quadcopter:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.lidar_range = 1 # 5 is a good starting value to mess around with

    def sense(self, corners):
        readings = []
        for lx, ly in corners:
            dist = np.hypot(lx - self.x, ly - self.y)
            if dist <= self.lidar_range:
                readings.append((lx, ly, dist))
        return readings

    def move(self, dx, dy):
        new_x = self.x + dx
        new_y = self.y + dy



# Particle filter class
class ParticleFilter:
    def __init__(self, num_particles, initial_x, initial_y):
        self.num_particles = num_particles
        print("Initial X: " + str(initial_x))
        self.particles = np.random.randn(num_particles, 2) * 10 + [initial_x, initial_y]
        self.weights = np.ones(num_particles) / num_particles

    def motion_update(self, dx, dy):
        noise = np.random.randn(self.num_particles, 2) * 2
        new_particles = self.particles + [dx, dy] + noise
        for i in range(len(new_particles)):
            self.particles[i] = new_particles[i]
        #self.particles[:, 0] = np.clip(self.particles[:, 0], 0, WIDTH)
        #self.particles[:, 1] = np.clip(self.particles[:, 1], 0, HEIGHT)

    def sensor_update(self, lidar_readings):
        for i, particle in enumerate(self.particles):
            weight = 1.0
            for lx, ly, dist in lidar_readings:
                pred_dist = np.hypot(lx - particle[0], ly - particle[1])
                weight *= np.exp(-((pred_dist - dist) ** 2) / (2 * 20 ** 2))
            self.weights[i] = weight
        self.weights += 1.e-300  # Avoid zeros
        self.weights /= np.sum(self.weights)

    def compute_uncertainty(self):
        spread = np.std(self.particles, axis=0)
        uncertainty = spread[0] + spread[1]
        return uncertainty

    def resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)




def run_robot_path(robot, path, corners):
    particle_filter = ParticleFilter(500, robot.x, robot.y)
    uncertainty_array = []
    position_array = []
    ending_point = path[-1]
    for point in path:
        #print("Point: " + str(point[0]) + " " + str(point[1]))
        position_array.append([robot.x, robot.y])
        #print("Position: " + str(robot.x) + " " + str(robot.y))
        euclidean_distance = math.sqrt(math.pow((robot.x - ending_point[0]), 2) + math.pow((robot.y - ending_point[1]), 2))
        manhattan_distance = abs(robot.x - ending_point[0]) + abs(robot.y - ending_point[1])
        chebyshev_distance = max(abs(robot.x - ending_point[0]), abs(robot.y - ending_point[1]))
        position_array.append(euclidean_distance)
        # Give movement commands based on the difference between the robot's current position and the next desired
        # position in terms of dx and dy
        # Plot this new position to make sure the robot is following the correct path
        dx = point[0] - robot.x
        dy = point[1] - robot.y
        robot.x = point[0]
        robot.y = point[1]
        particle_filter.motion_update(dx, dy)

        # Sense and update particles
        lidar_readings = robot.sense(corners)
        particle_filter.sensor_update(lidar_readings)
        particle_filter.resample()

        # Compute and print uncertainty
        uncertainty = particle_filter.compute_uncertainty()
        uncertainty_array.append(uncertainty)
    return uncertainty_array, position_array
    # Make graph of uncertainity over iterations