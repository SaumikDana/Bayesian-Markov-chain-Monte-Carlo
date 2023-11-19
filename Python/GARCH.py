__author__ = "Saumik Dana"
__purpose__ = "Demonstrate Bayesian inference using RSF model"

import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import os

# Load the dataset
output_dir = '/Users/saumikdana/Bayesian_MCMC_Deep-Learning/data'
csv_path = os.path.join(output_dir, 'all_stocks_5yr.csv')
data = pd.read_csv(csv_path)

# Count the occurrences of each stock symbol
stock_counts = data['Name'].value_counts()

# Select the top n stocks based on occurrences
n = 20
top_n_stocks = stock_counts.head(n).index

# Prepare the returns series for each top n stock
returns = {}
for stock in top_n_stocks:
    stock_data = data[data['Name'] == stock]
    stock_returns = stock_data['close'].pct_change().dropna()
    returns[stock] = stock_returns

# Specify and fit the GARCH(1, 1) model for each top 50 stock
alpha_results = {}

# GARCH stands for Generalized Autoregressive Conditional Heteroskedasticity.
# It is a statistical model used to capture the time-varying nature of volatility
# in financial time series data. It's especially useful for stock prices.
# The GARCH(1, 1) model specifies that the current variance is a function of
# the immediately preceding variance and the immediately preceding squared residual.
for stock, stock_returns in returns.items():
    # Specify and fit the GARCH(1, 1) model
    # Here, p=1 and q=1 specify the order of the GARCH model.
    model = arch_model(stock_returns, vol='Garch', p=1, q=1)
    results = model.fit(disp='off')

    # Store the estimated alpha parameter
    # The alpha parameter in a GARCH(1, 1) model measures the effect of past squared residuals
    # (or shocks) on the current conditional variance. It reflects how responsive the variance
    # is to sudden shocks.
    alpha_results[stock] = results.params['alpha[1]']

# Sort the alpha results in descending order
sorted_alpha_results = sorted(alpha_results.items(), key=lambda x: x[1], reverse=True)
sorted_stocks, sorted_alphas = zip(*sorted_alpha_results)

# Plot the estimated alpha parameters for the top n stocks
plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_alphas)), sorted_alphas, align='center')
plt.xticks(range(len(sorted_alphas)), sorted_stocks)
plt.xlabel('Stock')
plt.ylabel('Estimated Alpha Parameter')
plt.title(f'Estimated Alpha Parameter for the Top {n} Stocks')
plt.xticks(rotation=90)

# Annotate stock names on the plot
for i, alpha in enumerate(sorted_alphas):
    plt.annotate(f'{alpha:.4f}', (i, alpha), ha='center', va='bottom')

plt.tight_layout()
plt.show()
