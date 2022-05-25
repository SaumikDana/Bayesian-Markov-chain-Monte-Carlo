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

    rate = 0.2

    opt,count,lst=gradient_descent(gradient=lambda v: 2 * v, start=10.0,learn_rate=rate)
    x = np.linspace(-11,11,100)
    y = [number**2 for number in x]    
    yp = [number ** 2 for number in lst]

#    opt,count,lst=gradient_descent(gradient=lambda v: 4 * v**3 - 10 * v - 3,start=0.0,learn_rate=rate)
#    x = np.linspace(-3,3,100)
#    y = [number**4-5*number**2-3*number for number in x]
#    yp = [number**4-5*number**2-3*number for number in lst]

    plt.plot(x,y)
    plt.plot(lst,yp,'o-')    
    plt.plot()
    plt.xlabel('x')
    plt.ylabel('$f(x)$')
    plt.title('Learning rate %s' %rate)
    plt.show()
