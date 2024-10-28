import psycopg2
import csv

try:
    # Establish the connection to PostgreSQL
    conn = psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="rootpo"
    )
    # Create a cursor object
    cur = conn.cursor()
    print("Connection to the database established successfully.")


except Exception as e:
    print("An error occurred:", e)

finally:
    # Close the cursor and connection
    if cur:
        cur.close()
    if conn:
        conn.close()



