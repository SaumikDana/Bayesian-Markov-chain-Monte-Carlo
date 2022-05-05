from scipy.stats import poisson
import matplotlib.pyplot as plt

poisson_rv1 = poisson.rvs(mu=2, size=10000)
plt.hist(poisson_rv1,bins=100, density = 1)
plt.title('Poisson Distribution $p(x=k)=\\frac{\lambda^k e^{-\\lambda}}{k!}$ with $\\lambda=2$')
plt.xlabel('Value of Poisson RV')
plt.ylabel('Frequency of occurrence')
plt.show()


poisson_rv2 = poisson.rvs(mu=10, size=10000)
plt.hist(poisson_rv2,bins=100, density = 1)
plt.title('Poisson Distribution $p(x=k)=\\frac{\lambda^k e^{-\\lambda}}{k!}$ with $\\lambda=10$')
plt.xlabel('Value of Poisson RV')
plt.ylabel('Frequency of occurrence')
plt.show()
