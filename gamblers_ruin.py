import matplotlib.pyplot as plt
import numpy as np

def solve_equation(N, p):
    x = [0] * (N + 1)
    x[0] = 1
    x[N] = N

    for i in range(1, N):
        x[i+1] = (x[i] - (1-p)*x[i-1]) / p

    return x[1:N]

# Range of p values
p_values = np.linspace(0.5, 1, num=100)

# Empty lists to store results
x0_values = []

# Solve the equation for different p values
N = 10  # Number of elements
point = int(N/2)
for p in p_values:
    solution = solve_equation(N, p)
    x0_values.append(solution[point])

# Plotting
plt.plot(p_values, x0_values)
plt.xlabel('p')
plt.ylabel(f'x[{point}]')
plt.title(f'Plot of x[{point}] against different values of p')
plt.grid(True)
plt.show()
