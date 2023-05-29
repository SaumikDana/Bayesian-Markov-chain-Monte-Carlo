import mysql.connector
import numpy as np

def save_object(data, host, user, password, database, chunk_size=5000):
    cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
    cursor = cnx.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS my_table (id INT PRIMARY KEY, value FLOAT)")

    cursor.execute("SELECT MAX(id) FROM my_table")
    max_id = cursor.fetchone()[0]
    if max_id is None:
        max_id = 0

    # Split the data into chunks
    chunks = np.array_split(data, np.ceil(len(data) / float(chunk_size)))

    for chunk in chunks:
        data_to_insert = ", ".join([f"({max_id + 1 + i}, {float(value)})" for i, value in enumerate(chunk)])
        cursor.execute(f"INSERT INTO my_table (id, value) VALUES {data_to_insert}")
        max_id += len(chunk)

    cnx.commit()
    cursor.close()
    cnx.close()

def load_object(host, user, password, database):
    cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
    cursor = cnx.cursor()
    cursor.execute("SELECT value FROM my_table")
    data = cursor.fetchall()
    cursor.close()
    cnx.close()
    return np.array(data).flatten()
