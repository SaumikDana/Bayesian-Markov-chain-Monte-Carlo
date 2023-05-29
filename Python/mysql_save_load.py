import mysql.connector
import numpy as np

def save_object(data, host, user, password, database, chunk_size=5000):
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
