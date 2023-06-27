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
    # Check for missing values
    missing_values = df.isnull().sum()
    # Handle missing values (if any)
    df = df.dropna()  # Drop rows with missing values
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
        plt.plot(stock.index, monthly_mean, label=f'{stock_name} Monthly Mean Closing Price')
    except Exception as e:
        print(f"Error plotting data for {stock_name}: {e}")

plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'Stock Price Time Series for Multiple Stocks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Second plot: First Derivatives
plt.figure(figsize=(12, 6))

for stock_name in stock_names:
    try:
        stock = df[df['Name'] == stock_name].copy()
        stock.set_index('date', inplace=True)
        monthly_mean = stock['close']
        first_derivative = monthly_mean.diff()
        plt.plot(stock.index, first_derivative, label=f'{stock_name} First Derivative')
    except Exception as e:
        print(f"Error plotting data for {stock_name}: {e}")

plt.xlabel('Date')
plt.ylabel('Change in Price')
plt.title(f'First Derivatives of Stock Prices for Multiple Stocks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Third plot: Second Derivatives
plt.figure(figsize=(12, 6))

for stock_name in stock_names:
    try:
        stock = df[df['Name'] == stock_name].copy()
        stock.set_index('date', inplace=True)
        monthly_mean = stock['close']
        first_derivative = monthly_mean.diff()
        second_derivative = first_derivative.diff()
        plt.plot(stock.index, second_derivative, label=f'{stock_name} Second Derivative')
    except Exception as e:
        print(f"Error plotting data for {stock_name}: {e}")

plt.xlabel('Date')
plt.ylabel('Change in Change in Price')
plt.title(f'Second Derivatives of Stock Prices for Multiple Stocks')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()