import os
import pandas as pd
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Function to download and extract the dataset
def download_and_extract_dataset(api, dataset_name, output_dir):
    """
    Downloads and extracts the dataset from Kaggle.

    :param api: Authenticated Kaggle API object.
    :param dataset_name: Name of the Kaggle dataset.
    :param output_dir: Directory to store the downloaded files.
    """
    zip_file = os.path.join(output_dir, 'sandp500.zip')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.isfile(zip_file):
        api.dataset_download_files(dataset_name, path=output_dir, unzip=False)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already downloaded and extracted.")

# Function to authenticate with Kaggle API
def authenticate_kaggle_api():
    """
    Authenticates with the Kaggle API and returns the API object.

    :return: Authenticated Kaggle API object.
    """
    api = KaggleApi()
    api.authenticate()
    return api

# Function to load and prepare the dataset
def load_and_prepare_data(csv_path):
    """
    Loads the dataset from a CSV file, handles missing data, and prepares it for analysis.

    :param csv_path: Path to the CSV file.
    :return: Prepared DataFrame.
    """
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.groupby('Name').resample('M').mean(numeric_only=True).reset_index()
    return df

# Function to plot stock data
def plot_stock_data(df, stock_names, plot_type):
    """
    Plots stock data for the given stock names and plot type.

    :param df: DataFrame containing stock data.
    :param stock_names: List of stock names to plot.
    :param plot_type: Type of plot ('close', 'first_derivative', 'second_derivative').
    """
    plt.figure(figsize=(12, 6))

    for stock_name in stock_names:
        stock = df[df['Name'] == stock_name].copy()
        stock.set_index('date', inplace=True)
        if plot_type == 'close':
            plt.plot(stock.index, stock['close'], label=f'{stock_name} Monthly Mean Closing Price')
        elif plot_type == 'first_derivative':
            first_derivative = stock['close'].diff()
            plt.plot(stock.index, first_derivative, label=f'{stock_name} First Derivative')
        elif plot_type == 'second_derivative':
            second_derivative = stock['close'].diff().diff()
            plt.plot(stock.index, second_derivative, label=f'{stock_name} Second Derivative')

    plt.xlabel('Date')
    plt.ylabel('Price' if plot_type == 'close' else 'Change in Price')
    plt.title(f'{plot_type.replace("_", " ").title()} of Stock Prices for Multiple Stocks')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main function to execute the script
def main():
    """
    Main function to execute the script.
    """
    dataset_name = 'camnugent/sandp500'
    output_dir = '/path/to/your/data/directory'
    csv_path = os.path.join(output_dir, 'all_stocks_5yr.csv')
    stock_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']

    api = authenticate_kaggle_api()
    download_and_extract_dataset(api, dataset_name, output_dir)
    df = load_and_prepare_data(csv_path)
    
    plot_stock_data(df, stock_names, 'close')
    plot_stock_data(df, stock_names, 'first_derivative')
    plot_stock_data(df, stock_names, 'second_derivative')

# Checks if the script is run as the main program and calls the main function
if __name__ == "__main__":
    main()
