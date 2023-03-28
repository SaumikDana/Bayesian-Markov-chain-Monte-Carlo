import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def gradient_descent(gradient, start, learn_rate, n_iter=50, tol=1e-8):
    """
    Gradient descent optimization algorithm.
    
    Parameters:
        gradient (function): Function that takes a 2-element numpy array as input and returns
                             a 2-element numpy array representing the gradient of the function
                             to be optimized.
        start (numpy array): 2-element numpy array representing the starting point for optimization.
        learn_rate (float):  Learning rate for gradient descent.
        n_iter (int):        Maximum number of iterations for gradient descent. Default is 50.
        tol (float):         Tolerance for convergence. If the absolute value of all elements in the
                             gradient is less than or equal to tol, the algorithm stops. Default is 1e-8.
                             
    Returns:
        numpy array: 2-element numpy array representing the optimal point found by the algorithm.
        int:         Number of iterations required to converge.
        list:        List of x-coordinates visited during optimization.
        list:        List of y-coordinates visited during optimization.
    """
    vector = start
    count = 0
    lst0 = []
    lst1 = []
    for _ in range(n_iter):
        lst0.append(vector[0])
        lst1.append(vector[1])
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff)<=tol):
            break
        vector += diff
        count += 1
    return vector,count,lst0,lst1

if __name__ == '__main__':
    
    rate = 0.05
    
    # Call gradient_descent function with function gradient, starting point start, 
    # learning rate learn_rate, and tolerance for convergence tol as input.
    # The function gradient is defined as a lambda function that takes a 2-element numpy array as input and returns
    # a 2-element numpy array representing the gradient of the function to be optimized.
    opt,count,lst0,lst1=gradient_descent(gradient=lambda v: np.array([2 * v[0], 4 * v[1]**3]),
            start=np.array([1.0, 1.0]), learn_rate=rate, tol=1e-08)

    # Create a meshgrid of x and y values and compute the corresponding z-values for the function to be optimized.
    x = np.meshgrid(np.linspace(-1.1,1.1,100),np.linspace(-1.1,1.1,100))
    y = x[0]**2 + x[1]**4
    
    # Create a 3D plot of the function.
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(x[0], x[1], y)
    
    # Compute the z-values for the optimization path and plot it on the 3D plot.
    yp = [lst0[i]**2+lst1[i]**4 for i in range(len(lst0))]
    plt.plot(lst0,lst1,yp,'ro-')
    
    # Add a title to the plot that shows the learning rate used for the optimization.
    plt.title('Learning rate %s' %rate)
    plt.show()

