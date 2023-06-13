import numpy as np
import matplotlib.pyplot as plt

# True parameter value
true_theta = 1.5

# Generate observed data
np.random.seed(0)
X = np.random.exponential(true_theta, size=100)

# Likelihood function
def likelihood(theta, data):
    return np.prod(theta * np.exp(-theta * data))

# Prior distribution
def prior(theta):
    return np.exp(-theta)

# MLE estimation
theta_range = np.linspace(0.01, 5, 100)
likelihood_values = [likelihood(theta, X) for theta in theta_range]
mle_estimate = theta_range[np.argmax(likelihood_values)]

# MAP estimation
posterior_values = [likelihood(theta, X) * prior(theta) for theta in theta_range]
posterior_values /= np.trapz(posterior_values, theta_range)  # Normalize posterior
map_estimate = theta_range[np.argmax(posterior_values)]

# Plot likelihood and posterior
plt.figure(figsize=(8, 6))
plt.plot(theta_range, likelihood_values, label='Likelihood')
plt.plot(theta_range, posterior_values, label='Posterior')
plt.axvline(x=true_theta, color='red', linestyle='--', label='True Theta')
plt.axvline(x=mle_estimate, color='blue', linestyle='--', label='MLE Estimate')
plt.axvline(x=map_estimate, color='green', linestyle='--', label='MAP Estimate')
plt.xlabel('Theta')
plt.ylabel('Density')
plt.title('MLE vs MAP Estimation')
plt.legend()
plt.grid(True)
plt.show()
