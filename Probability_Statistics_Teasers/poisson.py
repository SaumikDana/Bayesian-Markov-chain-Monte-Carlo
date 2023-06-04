import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# Set the lambda parameter for the Poisson distribution
lambda_param = 3

# Generate a range of x values
x = np.arange(0, 15)

# Calculate the probability mass function (PMF) using the Poisson formula
pmf = np.exp(-lambda_param) * (lambda_param ** x.astype(int)) / special.factorial(x)

# Calculate the cumulative probabilities iteratively
cdf = np.cumsum(pmf)

# Plot the PMF
plt.subplot(2, 1, 1)
plt.bar(x, pmf, alpha=0.5)
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.title('Poisson Distribution (λ=3) - PMF')

# Plot the cumulative probabilities
plt.subplot(2, 1, 2)
plt.plot(x, cdf, '-o')
plt.xlabel('Number of Events')
plt.ylabel('Cumulative Probability')
plt.title('Poisson Distribution (λ=3) - CDF (Explicit Integration)')

plt.tight_layout()
plt.show()
