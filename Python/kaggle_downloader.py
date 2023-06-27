import os

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError as e:
    print(f"Error importing libraries: {e}")
    exit()

# Set Kaggle API credentials
api = KaggleApi()
try:
    api.authenticate()
except Exception as e:
    print(f"Error authenticating with Kaggle API: {e}")
    exit()

# Define the dataset and output directory
dataset_name = 'camnugent/sandp500'
output_dir = '/Users/saumikdana/Bayesian_MCMC_Deep-Learning/data'
zip_file = os.path.join(output_dir, 'sandp500.zip')

# Check if output directory exists, if not create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Download the dataset if it doesn't exist
if not os.path.isfile(zip_file):
    try:
        api.dataset_download_files(dataset_name, path=output_dir, unzip=False)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        exit()
else:
    print("Dataset already downloaded.")

# Unzip the dataset
csv_path = os.path.join(output_dir, 'all_stocks_5yr.csv')
if not os.path.isfile(csv_path):
    try:
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    except Exception as e:
        print(f"Error unzipping dataset: {e}")
        exit()

# Read the downloaded CSV file
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Process the time series data
try:
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.groupby('Name').resample('M').mean(numeric_only=True).reset_index()  # Group by stock name
except Exception as e:
    print(f"Error processing time series data: {e}")
    exit()

# Specify the stock names here
stock_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']

# First plot: Monthly Mean Closing Price z-scored
plt.figure(figsize=(12, 6))

for stock_name in stock_names:
    try:
        stock = df[df['Name'] == stock_name].copy()
        stock.set_index('date', inplace=True)
        monthly_mean = stock['close']
        zscored_monthly_mean = (monthly_mean - monthly_mean.mean()) / monthly_mean.std()
        plt.plot(stock.index, zscored_monthly_mean, label=f'{stock_name} Z-Scored Monthly Mean Closing Price')
    except Exception as e:
        print(f"Error plotting data for {stock_name}: {e}")

plt.xlabel('Date')
plt.ylabel('Z-Scored Price')
plt.title(f'Z-Scored Stock Price Time Series for Multiple Stocks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Second plot: Z-Scored First Derivatives
plt.figure(figsize=(12, 6))

for stock_name in stock_names:
    try:
        stock = df[df['Name'] == stock_name].copy()
        stock.set_index('date', inplace=True)
        monthly_mean = stock['close']
        zscored_monthly_mean = (monthly_mean - monthly_mean.mean()) / monthly_mean.std()
        first_derivative = zscored_monthly_mean.diff()
        plt.plot(stock.index, first_derivative, label=f'{stock_name} Z-Scored First Derivative')
    except Exception as e:
        print(f"Error plotting data for {stock_name}: {e}")

plt.xlabel('Date')
plt.ylabel('Z-Scored Change in Price')
plt.title(f'Z-Scored First Derivatives of Stock Prices for Multiple Stocks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Third plot: Z-Scored Second Derivatives
plt.figure(figsize=(12, 6))

for stock_name in stock_names:
    try:
        stock = df[df['Name'] == stock_name].copy()
        stock.set_index('date', inplace=True)
        monthly_mean = stock['close']
        zscored_monthly_mean = (monthly_mean - monthly_mean.mean()) / monthly_mean.std()
        first_derivative = zscored_monthly_mean.diff()
        zscored_first_derivative = (first_derivative - first_derivative.mean()) / first_derivative.std()
        second_derivative = zscored_first_derivative.diff()
        plt.plot(stock.index, second_derivative, label=f'{stock_name} Z-Scored Second Derivative')
    except Exception as e:
        print(f"Error plotting data for {stock_name}: {e}")

plt.xlabel('Date')
plt.ylabel('Z-Scored Change in Change in Price')
plt.title(f'Z-Scored Second Derivatives of Stock Prices for Multiple Stocks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()