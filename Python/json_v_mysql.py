import json
import time
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector
from rsf import rsf
from RateStateModel import RateStateModel

# Function to measure the execution time of a function
def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return execution_time
    return wrapper

# Function to store data in JSON format
@measure_execution_time
def store_data_in_json(data):
    with open("data.json", "w") as file:
        json.dump(data.tolist(), file)

# Function to retrieve data from JSON
@measure_execution_time
def retrieve_data_from_json():
    with open("data.json", "r") as file:
        data = json.load(file)
    return np.array(data)

# Function to store data in MySQL database
@measure_execution_time
def store_data_in_mysql(data, host, user, password, database):
    cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
    cursor = cnx.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS my_table (id INT PRIMARY KEY, value FLOAT)")

    cursor.execute("SELECT MAX(id) FROM my_table")
    max_id = cursor.fetchone()[0]
    if max_id is None:
        max_id = 0

    data_to_insert = [(max_id + 1 + i, float(value)) for i, value in enumerate(data)]
    cursor.executemany("INSERT INTO my_table (id, value) VALUES (%s, %s)", data_to_insert)

    cnx.commit()
    cursor.close()
    cnx.close()

# Function to retrieve data from MySQL database
@measure_execution_time
def retrieve_data_from_mysql(host, user, password, database):
    cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
    cursor = cnx.cursor()
    cursor.execute("SELECT value FROM my_table")
    data = cursor.fetchall()
    cursor.close()
    cnx.close()
    return np.array(data).flatten()

# Generate data with varying sizes
problem = rsf(number_slip_values=5,lowest_slip_value=100.,largest_slip_value=1000.)

data_sizes = [100,500,1000,2000]
json_store_times = []
json_retrieve_times = []
mysql_store_times = []
mysql_retrieve_times = []
actual_size = []

# MySQL connection details
mysql_host = 'localhost'  # replace with your MySQL host
mysql_user = 'my_user'    # replace with your MySQL user
mysql_password = 'my_password'  # replace with your MySQL password
mysql_database = 'my_database'   # replace with your MySQL database

for size in data_sizes:

    # RSF model
    problem.model = RateStateModel(number_time_steps=size)
    
    # Generate the time series for the RSF model
    data = problem.generate_time_series()

    actual_size.append(len(data))

    # Measure JSON execution time for storing and retrieving
    json_store_time = store_data_in_json(data)
    json_retrieve_time = retrieve_data_from_json()
    json_store_times.append(json_store_time)
    json_retrieve_times.append(json_retrieve_time)

    # Measure MySQL execution time for storing and retrieving
    mysql_store_time = store_data_in_mysql(data, mysql_host, mysql_user, mysql_password, mysql_database)
    mysql_retrieve_time = retrieve_data_from_mysql(mysql_host, mysql_user, mysql_password, mysql_database)
    mysql_store_times.append(mysql_store_time)
    mysql_retrieve_times.append(mysql_retrieve_time)

# Plot the comparison for storing times
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(actual_size, json_store_times, '-o', label='JSON')
plt.plot(actual_size, mysql_store_times, '-o', label='MySQL')
plt.xlabel('Data Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Store Execution Time')
plt.legend()

# Plot the comparison for retrieving times
plt.subplot(1, 2, 2)
plt.plot(actual_size, json_retrieve_times, '-o', label='JSON')
plt.plot(actual_size, mysql_retrieve_times, '-o', label='MySQL')
plt.xlabel('Data Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Retrieve Execution Time')
plt.legend()

plt.tight_layout()
plt.show()
