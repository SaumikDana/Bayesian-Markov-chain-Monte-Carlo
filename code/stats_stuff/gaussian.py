import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

mu_params = [-1, 0, 1]
sd_params = [0.5, 1, 1.5]
x = np.linspace(-7, 7, 200)
fig, ax = plt.subplots(len(mu_params), len(sd_params), sharex=True, sharey=True, figsize=(9, 7))
for i in range(3):
  for j in range(3):
     mu = mu_params[i]
     sd = sd_params[j]
     y = stats.norm(mu, sd).pdf(x)
     ax[i,j].plot(x, y)
     ax[i,j].plot([], label="$\mu$ = {:3.2f}\n $\sigma$ = {:3.2f}".format(mu,sd), alpha=0)
     ax[i,j].legend(loc=1,frameon=False)
ax[2,1].set_xlabel('x')
ax[1,0].set_ylabel('p(x)', rotation=0, labelpad=20)
ax[1,0].set_yticks([])
fig.suptitle('Gaussian distribution $p(x)=\\frac{1}{\sigma\sqrt{2\pi}}e^{-\\frac{(x-\mu)^2}{2\sigma^2}}$ with different mean and standard deviation')

# Close it out
plt.show()
plt.close('all')
