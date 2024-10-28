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

    # Open your CSV file
    with open(r'C:\Users\HP\Fraud use case\Datagen\fraud.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row if it exists

        # Prepare an SQL query to insert data
        query = """
        INSERT INTO transactions (
            transaction_id, customer_id, transaction_date, amount, merchant_id,
            payment_method, location, device_id, ip_address, fraud_label,
            customer_age, gender, income_bracket, total_transactions,
            avg_transaction_amount, churn_risk, segment_label, last_transaction_date
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Insert each row into the database
        for row in reader:
            cur.execute(query, row)

    # Commit the transaction
    conn.commit()
    print("Data inserted successfully!")

except Exception as e:
    print("An error occurred:", e)

finally:
    # Close the cursor and connection
    if cur:
        cur.close()
    if conn:
        conn.close()



