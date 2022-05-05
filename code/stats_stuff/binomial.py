## Binomial Random variable
import numpy as np
import seaborn as sns
from scipy.stats import binom
import matplotlib.pyplot as plt

## n corresponds to the number of trials in each experiment, size refers to total runs of the experiment, p is the probability of success.
binom_rv = binom.rvs(n=20,p=0.5,loc=0,size=100)

print('Number of successes in each trial having 20 coin tosses =', binom_rv)
## distplot from seaborn library is used to visualize probability distributions
ax = sns.distplot(binom_rv, color='blue', hist_kws={"linewidth": 10,'alpha':1})
# hist_kws specifies keywords to be used with histogram, linewidth specifies the width of bars and alpha is used to adjust the color strength
ax.set(xlabel='Values of Binomial RV', ylabel='Relative Frequency (Probability)', title ='Binomial Distribution, 20 trials//100 experiments, success probability 0.5')

# Close it out
plt.show()
plt.close('all')

