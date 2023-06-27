import random
import math
import matplotlib.pyplot as plt

def approximate_pi(num_points):
    points_in_circle = 0
    total_points = 0
    pi_approximations = []

    for _ in range(num_points):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        distance = math.sqrt(x**2 + y**2)

        if distance <= 1:
            points_in_circle += 1

        total_points += 1
        pi_approximation = 4 * (points_in_circle / total_points)
        pi_approximations.append(pi_approximation)

    return pi_approximations

# Set the number of points range for approximation
start_num_points = 1000
end_num_points = 100000
step_size = 1000

# Generate a range of number of points
num_points_range = range(start_num_points, end_num_points+1, step_size)

# Initialize lists to store the number of points and corresponding approximations
num_points_list = []
pi_approx_list = []

# Calculate the approximation for each number of points in the range
for num_points in num_points_range:
    pi_approximations = approximate_pi(num_points)
    last_approximation = pi_approximations[-1]
    
    num_points_list.append(num_points)
    pi_approx_list.append(last_approximation)

# Plot the results
plt.plot(num_points_list, pi_approx_list)
plt.axhline(y=math.pi, color='r', linestyle='--', label='Actual Value of Pi')
plt.xlabel('Number of Points')
plt.ylabel('Approximation of Pi')
plt.title('Approximation of Pi vs. Number of Points')
plt.legend()
plt.show()
