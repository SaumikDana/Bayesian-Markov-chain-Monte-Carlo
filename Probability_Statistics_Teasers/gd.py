import numpy as np
import matplotlib.pyplot as plt

# Define the function and gradients
f = lambda x, y: x**2 + y**2
dfdx = lambda x, y: 2*x
dfdy = lambda x, y: 2*y

# Define the batch gradient descent function
def batch_gradient_descent(x_start, y_start, dfdx, dfdy, learning_rate, epochs):
    xs = np.zeros(epochs+1)
    ys = np.zeros(epochs+1)
    xs[0], ys[0] = x_start, y_start
    
    for i in range(epochs):
        dx, dy = dfdx(xs[i], ys[i]), dfdy(xs[i], ys[i])
        xs[i+1] = xs[i] - learning_rate * dx
        ys[i+1] = ys[i] - learning_rate * dy
    
    return xs, ys

# Define the stochastic gradient descent function
def stochastic_gradient_descent(x_start, y_start, dfdx, dfdy, learning_rate, epochs):
    xs = np.zeros(epochs+1)
    ys = np.zeros(epochs+1)
    xs[0], ys[0] = x_start, y_start
    
    for i in range(epochs):
        dx, dy = dfdx(xs[i], ys[i]), dfdy(xs[i], ys[i])
        # Only apply gradient from one of the dimensions randomly
        if np.random.rand() < 0.5:
            xs[i+1] = xs[i] - learning_rate * dx
            ys[i+1] = ys[i]
        else:
            xs[i+1] = xs[i]
            ys[i+1] = ys[i] - learning_rate * dy
    
    return xs, ys

# Set the starting point and other parameters
x_start, y_start = 2.0, 2.5
learning_rate = 0.1
epochs = 20

# Perform the gradient descents
xs_batch, ys_batch = batch_gradient_descent(x_start, y_start, dfdx, dfdy, learning_rate, epochs)
xs_stochastic, ys_stochastic = stochastic_gradient_descent(x_start, y_start, dfdx, dfdy, learning_rate, epochs)

# Plot the function with a contour plot
Y, X = np.mgrid[-3:3:100j, -3:3:100j]
Z = f(X, Y)
plt.contour(X, Y, Z, colors='black')

# Plot the path taken by gradient descent with arrows
for i in range(1, len(xs_batch)):
    plt.arrow(xs_batch[i-1], ys_batch[i-1], xs_batch[i]-xs_batch[i-1], ys_batch[i]-ys_batch[i-1], 
              shape='full', color='blue', lw=0.5, length_includes_head=True, head_width=.05)
plt.plot(xs_batch, ys_batch, color='blue', label='Batch')

for i in range(1, len(xs_stochastic)):
    plt.arrow(xs_stochastic[i-1], ys_stochastic[i-1], xs_stochastic[i]-xs_stochastic[i-1], ys_stochastic[i]-ys_stochastic[i-1], 
              shape='full', color='red', lw=0.5, length_includes_head=True, head_width=.05)
plt.plot(xs_stochastic, ys_stochastic, color='red', label='Stochastic')

plt.legend()
plt.title('Convergence of Batch and Stochastic Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
