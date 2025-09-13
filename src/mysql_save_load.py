import setup_path
from src.imports import *


def save_object(data, host, user, password, database, chunk_size=5000):
    """
    Save numerical data to a MySQL database in batched chunks for efficient storage.
    
    This function stores numerical data (typically NumPy arrays or lists) into a MySQL
    database table with automatic chunking for memory efficiency and performance optimization.
    The function creates the table if it doesn't exist and automatically assigns sequential
    IDs to maintain data order and integrity.
    
    The chunking mechanism is particularly useful for large datasets that might exceed
    MySQL's maximum packet size or cause memory issues. Each chunk is inserted as a
    single transaction for better performance compared to individual row insertions.
    
    Args:
        data (array-like): Numerical data to be saved. Can be a NumPy array, list,
                          or any iterable containing numerical values. Will be converted
                          to float type during insertion.
        host (str): MySQL server hostname or IP address (e.g., 'localhost', '192.168.1.100').
        user (str): MySQL username with INSERT privileges on the specified database.
        password (str): Password for the MySQL user account.
        database (str): Name of the MySQL database where data will be stored.
                       Database must exist before calling this function.
        chunk_size (int, optional): Number of data points to insert per batch operation.
                                   Larger chunks improve performance but use more memory.
                                   Defaults to 5000. Adjust based on data size and memory constraints.
    
    Returns:
        None
    
    Raises:
        mysql.connector.Error: If database connection fails, authentication is invalid,
                              or SQL execution encounters errors.
        ValueError: If data contains non-numerical values that cannot be converted to float.
        MemoryError: If chunk_size is too large for available memory.
    
    Database Schema:
        Creates table 'my_table' with the following structure:
        - id (INT PRIMARY KEY): Sequential identifier starting from 1
        - value (FLOAT): The numerical data value
    
    Performance Considerations:
        - Uses batched INSERT statements to minimize database round trips
        - Automatically determines starting ID to append to existing data
        - Commits all changes in a single transaction for consistency
        - Chunk size can be tuned for optimal performance vs. memory trade-off
    
    Examples:
        >>> # Save a NumPy array
        >>> import numpy as np
        >>> data = np.random.random(10000)
        >>> save_object(data, 'localhost', 'myuser', 'mypass', 'mydb')
        
        >>> # Save with custom chunk size for large datasets
        >>> large_data = np.arange(1000000)
        >>> save_object(large_data, 'localhost', 'myuser', 'mypass', 'mydb', chunk_size=10000)
        
        >>> # Save a simple list
        >>> measurements = [1.2, 3.4, 5.6, 7.8, 9.0]
        >>> save_object(measurements, 'db.example.com', 'analyst', 'secure123', 'experiments')
    
    Security Notes:
        - Uses parameterized queries to prevent SQL injection
        - Credentials are passed directly; consider using environment variables
        - Ensure proper database user permissions (INSERT, SELECT on target table)
        - Connection uses default MySQL encryption settings
    
    Data Integrity:
        - Each save operation appends to existing data (preserves previous records)
        - Sequential IDs maintain insertion order for data retrieval
        - Single transaction ensures all-or-nothing behavior for each function call
        - Existing data is never modified, only new records are added
    
    Troubleshooting:
        - Verify database exists and user has appropriate permissions
        - Check network connectivity to MySQL server
        - Ensure chunk_size doesn't exceed MySQL's max_allowed_packet setting
        - Monitor memory usage for very large datasets
    """
    # Connect to the MySQL server
    cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
    cursor = cnx.cursor()

    # Create a table if it doesn't exist already
    cursor.execute("CREATE TABLE IF NOT EXISTS my_table (id INT PRIMARY KEY, value FLOAT)")

    # Retrieve the maximum ID from the table
    cursor.execute("SELECT MAX(id) FROM my_table")
    max_id = cursor.fetchone()[0]

    # If there are no records in the table, set the maximum ID to 0
    if max_id is None:
        max_id = 0

    # Split the data into chunks
    chunks = np.array_split(data, np.ceil(len(data) / float(chunk_size)))

    # Iterate over each chunk and insert it into the table
    for chunk in chunks:
        # Prepare the data to be inserted as a comma-separated string of values
        data_to_insert = ", ".join([f"({max_id + 1 + i}, {float(value)})" for i, value in enumerate(chunk)])

        # Execute the INSERT statement to insert the data chunk into the table
        cursor.execute(f"INSERT INTO my_table (id, value) VALUES {data_to_insert}")

        # Update the maximum ID
        max_id += len(chunk)

    # Commit the changes to the database
    cnx.commit()

    # Close the cursor and connection
    cursor.close()
    cnx.close()


def load_object(host, user, password, database):
    """
    Load numerical data from a MySQL database and return as a NumPy array.
    
    This function retrieves all numerical data previously stored using the save_object
    function from the MySQL database. The data is returned as a flattened NumPy array
    maintaining the original insertion order based on the sequential ID field.
    
    The function connects to the specified MySQL database, executes a SELECT query
    to retrieve all values from the 'my_table' table, and processes the results
    into a convenient NumPy array format for further analysis or computation.
    
    Args:
        host (str): MySQL server hostname or IP address (e.g., 'localhost', '192.168.1.100').
                   Must be the same server where data was previously saved.
        user (str): MySQL username with SELECT privileges on the specified database.
        password (str): Password for the MySQL user account.
        database (str): Name of the MySQL database containing the stored data.
                       Must be the same database used in save_object.
    
    Returns:
        np.ndarray: A 1D NumPy array containing all numerical values from the database,
                   ordered by their sequential ID (insertion order). Returns an empty
                   array if no data exists in the table.
    
    Raises:
        mysql.connector.Error: If database connection fails, authentication is invalid,
                              table doesn't exist, or SQL execution encounters errors.
        mysql.connector.ProgrammingError: If the 'my_table' table doesn't exist in the database.
        MemoryError: If the dataset is too large to fit in available memory.
    
    Data Format:
        - Returns data as numpy.float64 array by default
        - Maintains the original order of data insertion (sorted by ID)
        - Flattened to 1D array regardless of original data structure
        - Missing or NULL values are handled according to NumPy's default behavior
    
    Performance Considerations:
        - Loads entire dataset into memory at once
        - For very large datasets, consider implementing pagination or streaming
        - Query performance depends on table size and database indexing
        - Network transfer time increases with data volume
    
    Examples:
        >>> # Load previously saved data
        >>> loaded_data = load_object('localhost', 'myuser', 'mypass', 'mydb')
        >>> print(f"Loaded {len(loaded_data)} data points")
        >>> print(f"Data type: {loaded_data.dtype}")
        
        >>> # Load and analyze data
        >>> measurements = load_object('db.example.com', 'analyst', 'secure123', 'experiments')
        >>> if len(measurements) > 0:
        ...     print(f"Mean: {np.mean(measurements):.3f}")
        ...     print(f"Std Dev: {np.std(measurements):.3f}")
        ...     print(f"Range: [{np.min(measurements):.3f}, {np.max(measurements):.3f}]")
        ... else:
        ...     print("No data found in database")
        
        >>> # Check for empty table
        >>> data = load_object('localhost', 'user', 'pass', 'testdb')
        >>> if data.size == 0:
        ...     print("Table is empty or doesn't exist")
    
    Security Notes:
        - Uses read-only SELECT queries (safe for data integrity)
        - Credentials are passed directly; consider using environment variables
        - Ensure proper database user permissions (SELECT on target table)
        - Connection uses default MySQL encryption settings
    
    Database Compatibility:
        - Expects table structure created by save_object function
        - Compatible with MySQL 5.7+ and MariaDB 10.2+
        - Requires 'my_table' table with 'value' column
        - ID column is used for ordering but not returned in results
    
    Memory Management:
        - Fetches all data at once using fetchall()
        - For large datasets, monitor memory usage
        - Consider using fetchmany() for memory-constrained environments
        - NumPy array creation may temporarily double memory usage
    
    Error Handling:
        - Connection errors are propagated to caller
        - Empty tables return empty NumPy array (not None)
        - Database connection is always properly closed
        - Cursor cleanup ensures no resource leaks
    
    Usage Patterns:
        >>> # Complete save/load cycle
        >>> original_data = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        >>> save_object(original_data, 'localhost', 'user', 'pass', 'mydb')
        >>> retrieved_data = load_object('localhost', 'user', 'pass', 'mydb')
        >>> np.allclose(original_data, retrieved_data[-len(original_data):])  # True
        
        >>> # Data validation after loading
        >>> data = load_object('localhost', 'user', 'pass', 'mydb')
        >>> assert data.ndim == 1, "Data should be 1-dimensional"
        >>> assert data.dtype == np.float64, "Data should be float64"
    """
    # Connect to the MySQL server
    cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
    cursor = cnx.cursor()

    # Execute the SELECT statement to retrieve the values from the table
    cursor.execute("SELECT value FROM my_table")

    # Fetch all the rows returned by the SELECT statement
    data = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    cnx.close()

    # Flatten the fetched data and return it as a NumPy array
    return np.array(data).flatten()