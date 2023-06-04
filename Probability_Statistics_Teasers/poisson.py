import numpy as np
import matplotlib.pyplot as plt

# Set the lambda parameter for the Poisson distribution
lambda_param = 3

# Generate a range of x values
x = np.arange(0, 15)

# Calculate the probability mass function (PMF) using the Poisson formula
factorial = np.vectorize(np.math.factorial)  # Vectorize the factorial function
pmf = np.exp(-lambda_param) * (lambda_param ** x.astype(int)) / factorial(x)

# Plot the PMF
plt.bar(x, pmf, alpha=0.5)
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.title('Poisson Distribution (λ=3)')
plt.show()
