
import matplotlib.pyplot as plt
from scipy.stats import norm

# generate random numbers from a normal probability density function (PDF) with zero mean and a standard deviation of 1: N(0,1)
norm_pdf = norm.rvs(size=10000,loc=0,scale=1)
plt.hist(norm_pdf, bins=100, density=1)
plt.xlabel('Random variable value')
plt.ylabel('Relative Frequency')
plt.title('Probability Density Function (PDF) for a Normal Random Variable')
plt.show()
