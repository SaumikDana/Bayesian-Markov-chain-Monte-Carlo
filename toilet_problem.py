import numpy as np
import matplotlib.pyplot as plt

# Generate x values from 0.001 to 1 (excluding 0)
x = np.linspace(0.001, 1, 500)

# Calculate y values for the function x * ln(1/x)
y = x * np.log(1 / x)

# Find the index of the maximum value
max_index = np.argmax(y)
max_x = x[max_index]
max_y = y[max_index]

# Create the plot
plt.plot(x, y)
plt.scatter(max_x, max_y, color='red', label=f'Maximum: ({max_x:.3f}, {max_y:.3f})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of x * ln(1/x)')
plt.legend()

# Display the plot
plt.show()
