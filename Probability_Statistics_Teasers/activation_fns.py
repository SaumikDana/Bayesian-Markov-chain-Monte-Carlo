import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_gradient(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_gradient(x):
    return 1 - np.tanh(x)**2

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_gradient(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def plot_activation_functions():
    x = np.linspace(-10, 10, 100)
    
    # Sigmoid
    plt.subplot(2, 2, 1)
    plt.plot(x, sigmoid(x), label='Sigmoid')
    plt.plot(x, sigmoid_gradient(x), label='Gradient')
    plt.legend()
    plt.title('Sigmoid')
    
    # ReLU
    plt.subplot(2, 2, 2)
    plt.plot(x, relu(x), label='ReLU')
    plt.plot(x, relu_gradient(x), label='Gradient')
    plt.legend()
    plt.title('ReLU')
    
    # Tanh
    plt.subplot(2, 2, 3)
    plt.plot(x, tanh(x), label='Tanh')
    plt.plot(x, tanh_gradient(x), label='Gradient')
    plt.legend()
    plt.title('Tanh')
    
    # Leaky ReLU
    plt.subplot(2, 2, 4)
    plt.plot(x, leaky_relu(x), label='Leaky ReLU')
    plt.plot(x, leaky_relu_gradient(x), label='Gradient')
    plt.legend()
    plt.title('Leaky ReLU')

    plt.tight_layout()
    plt.show()

plot_activation_functions()
