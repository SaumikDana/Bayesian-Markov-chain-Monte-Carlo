import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import scipy.stats as stats

colors = ["#348ABD", "#A60628"]
a = np.linspace(0, 4, 100)
expo = stats.expon
lambda_ = [0.5, 1]
for l, c in zip(lambda_, colors):
  plt.plot(a, expo.pdf(a, scale=1./l), lw=3, color=c, label="$\lambda = %.1f$" % l)
  plt.fill_between(a, expo.pdf(a, scale=1./l), color=c, alpha=.33)
plt.legend()
plt.ylabel("Probability density function at $z$")
plt.xlabel("$z$")
plt.ylim(0,1.2)
plt.title("$f(z|\lambda)=\lambda e^{-\lambda z}$, differing $\lambda$ values");
plt.show()
