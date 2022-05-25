import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def gradient_descent(gradient, start, learn_rate, n_iter=50, tol=1e-8):
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
    
    opt,count,lst0,lst1=gradient_descent(gradient=lambda v: np.array([2 * v[0], 4 * v[1]**3]),
            start=np.array([1.0, 1.0]), learn_rate=rate, tol=1e-08)

    x = np.meshgrid(np.linspace(-1.1,1.1,100),np.linspace(-1.1,1.1,100))
    y = x[0]**2 + x[1]**4
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(x[0], x[1], y)
    yp = [lst0[i]**2+lst1[i]**4 for i in range(len(lst0))]
    plt.plot(lst0,lst1,yp,'ro-')
    plt.title('Learning rate %s' %rate)
    plt.show()


