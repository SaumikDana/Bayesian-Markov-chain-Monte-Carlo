import numpy as np
import matplotlib.pyplot as plt
def gradient_descent(gradient, start, learn_rate, n_iter=50, tol=1e-6):
    vector = start
    count = 0
    lst = []
    for _ in range(n_iter):
        lst.append(vector)
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff)<=tol):
            break
        vector += diff
        count += 1
    return vector,count,lst

if __name__ == '__main__':
    rate = 0.8
    opt,count,lst=gradient_descent(gradient=lambda v: 2 * v, start=10.0,learn_rate=rate)
    x = np.linspace(-11,11,100)
    y = pow(x,2)
    plt.plot(x,y)
    yp = [number ** 2 for number in lst]
    plt.plot(lst,yp,'o-')    
    plt.plot()
    plt.xlabel('x')
    plt.ylabel('$f(x)=x^2$')
    plt.show()
