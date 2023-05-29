import sqlite3
import numpy as np

def save_object(data, filename):
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS my_table (id INTEGER PRIMARY KEY, value REAL)")
    
    cursor.execute("SELECT MAX(id) FROM my_table")
    max_id = cursor.fetchone()[0]
    if max_id is None:
        max_id = 0

    for i, value in enumerate(data):
        cursor.execute("INSERT INTO my_table (id, value) VALUES (?, ?)", (max_id + 1 + i, float(value)))
    
    conn.commit()
    conn.close()

def load_object(filename):
    conn = sqlite3.connect(filename)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM my_table")
    data = cursor.fetchall()
    conn.close()
    return np.array(data).flatten()

