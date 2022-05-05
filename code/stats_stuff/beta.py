
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

params = [0.5, 1, 2, 3]
x = np.linspace(0, 1, 100)
f, ax = plt.subplots(len(params), len(params), sharex=True, sharey=True, figsize=(8, 7))
for i in range(4):
  for j in range(4):
    a = params[i]
    b = params[j]
    y = stats.beta(a, b).pdf(x)
    ax[i,j].plot(x, y)
    ax[i,j].plot(0, 0, label="$\\alpha$ = {:2.1f}\n$\\beta$ = {:2.1f}".format(a,b), alpha=0)
    ax[i,j].legend(frameon=False)
ax[1,0].set_yticks([])
ax[1,0].set_xticks([0, 0.5, 1])
f.text(0.5, 0.05, '$\\theta$', ha='center')
f.text(0.07, 0.5, '$p(\\theta)$', va='center', rotation=0)
f.suptitle('Beta distribution $p(\\theta)=\\frac{\Gamma(\\alpha+\\beta)}{\Gamma(\\alpha)\Gamma(\\beta)}\\theta^{\\alpha-1}(1-\\theta)^{\\beta-1}$')

# Close it out
plt.show()
plt.close('all')
