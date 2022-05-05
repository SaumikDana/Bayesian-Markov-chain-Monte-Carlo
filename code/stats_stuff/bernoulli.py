from scipy.stats import bernoulli
from matplotlib import pyplot as plt

num_tosses = 1000
# p=0.5 is for fair coin, any other value of p results in unfair coin
fair_coin = bernoulli.rvs(p=0.5,size=num_tosses)
plt.hist(fair_coin)

plt.title('Bernouli Distribution for a fair coin $f=p^x(1-p)^{1-x},x=0,1$')
plt.xlabel('Value of Bernouli RV')
plt.ylabel('Frequency of occurrence')
plt.show()

# plotting distribution for an unfair coin
unfair_coin = bernoulli.rvs(p=0.2,size=num_tosses)
plt.hist(unfair_coin)

plt.title('Bernoulli Distribution for an unfair coin $f=p^x(1-p)^{1-x},x=0,1$')
plt.xlabel('Value of Bernoulli RV')
plt.ylabel('Frequency of occurrence')
plt.show()
