import os
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

class StockAnalysis:
    """
    A class for analyzing stock data using the GARCH model.
    """

    def __init__(self, data_path, n=20):
        """
        Initialize the StockAnalysis class.

        Parameters:
        - data_path (str): The path to the CSV file containing stock data.
        - n (int): The number of top stocks to analyze.
        """
        self.data_path = data_path
        self.n = n
        self.data = pd.read_csv(data_path)
        self.returns = {}
        self.alpha_results = {}

    def calculate_returns(self):
        """
        Calculate and store returns for the top n stocks.
        """
        # Count the occurrences of each stock symbol
        stock_counts = self.data['Name'].value_counts()
        
        # Select the top n stocks based on occurrences
        top_n_stocks = stock_counts.head(self.n).index

        # Prepare the returns series for each top n stock
        for stock in top_n_stocks:
            stock_data = self.data[self.data['Name'] == stock]
            stock_returns = stock_data['close'].pct_change().dropna()
            self.returns[stock] = stock_returns

    def fit_garch_model(self):
        """
        Fit the GARCH model for each stock and store the alpha parameters.
        """
        for stock, returns in self.returns.items():
            # Specify and fit the GARCH(1, 1) model
            model = arch_model(returns, vol='Garch', p=1, q=1)
            results = model.fit(disp='off')
            
            # Store the estimated alpha parameter
            self.alpha_results[stock] = results.params['alpha[1]']

    def plot_results(self):
        """
        Plot the estimated alpha parameters for each stock.
        """
        # Sort the alpha results in descending order
        sorted_alpha_results = sorted(self.alpha_results.items(), key=lambda x: x[1], reverse=True)
        sorted_stocks, sorted_alphas = zip(*sorted_alpha_results)

        # Plotting setup
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(sorted_alphas)), sorted_alphas, align='center')
        plt.xticks(range(len(sorted_alphas)), sorted_stocks)
        plt.xlabel('Stock')
        plt.ylabel('Estimated Alpha Parameter')
        plt.title(f'Estimated Alpha Parameters for the Top {self.n} Stocks')
        plt.xticks(rotation=90)

        # Annotating the plot
        for i, alpha in enumerate(sorted_alphas):
            plt.annotate(f'{alpha:.4f}', (i, alpha), ha='center', va='bottom')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Set up the file path for the data
    output_dir = '/Users/saumikdana/Bayesian_MCMC_Deep-Learning/data'
    csv_path = os.path.join(output_dir, 'all_stocks_5yr.csv')

    # Initialize the StockAnalysis class and execute its methods
    stock_analysis = StockAnalysis(data_path=csv_path)
    stock_analysis.calculate_returns()
    stock_analysis.fit_garch_model()
    stock_analysis.plot_results()
