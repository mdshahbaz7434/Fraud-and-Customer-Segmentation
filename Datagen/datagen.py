from faker import Faker
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
from sqlalchemy import create_engine
import psycopg2


from datetime import datetime, timedelta  # For insertaion date column 

# Initialize Faker instance
fake = Faker()

# Database credentials
db_username = 'postgres'
db_password = 'rootpo'
db_host = 'localhost'  # or your database host/IP
db_port = '5432'       # default PostgreSQL port
db_name = 'postgres'

# Create the database engine
engine = create_engine(
    f'postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
)

# Define the size of the dataset per batch
num_records = 1000

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate Transaction_ID and Customer_ID
transaction_ids = [f"TX{str(i).zfill(7)}" for i in range(num_records)]
customer_ids = [f"CUST{str(random.randint(1, 200000)).zfill(6)}" for _ in range(num_records)]

# Corrected Transaction_Date with seasonality (e.g., more transactions in December)
def generate_transaction_date():
    # Assign higher probability to December
    base_prob = 0.90 / 11  # Remaining probability after allocating to December
    month_probabilities = [base_prob for _ in range(11)]
    december_prob = 1.0 - sum(month_probabilities)
    month_probabilities.append(december_prob)  # December probability

    # Verify the sum
    total_prob = sum(month_probabilities)
    assert abs(total_prob - 1.0) < 1e-6, "Probabilities do not sum to 1."

    month = np.random.choice(
        range(1, 13),
        p=month_probabilities
    )
    day = np.random.randint(1, 28)  # Simplify to avoid month-end complexities
    hour = np.random.randint(0, 24)
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    return datetime(2023, month, day, hour, minute, second)

transaction_dates = [generate_transaction_date() for _ in range(num_records)]

# Placeholder list for Amounts
amounts = np.zeros(num_records)

# Placeholder list for Fraud_Label (to be assigned later)
fraud_labels = np.zeros(num_records, dtype=int)

# Generate preliminary Fraud_Label based on some criteria (to be refined later)
# For initial assignment, set fraud_prob to 3%
base_fraud_prob = 0.03

for i in range(num_records):
    # Base fraud probability
    fraud_prob = base_fraud_prob

    # Temporarily assign fraud_label
    fraud_labels[i] = np.random.choice([0, 1], p=[1 - fraud_prob, fraud_prob])

# Generate Merchant_ID with some popularity (some merchants have more transactions)
# Precompute weights for merchants to ensure consistency
merchant_weights = np.random.randint(1, 100, size=1000)
merchant_ids_list = [f"MERCH{str(i).zfill(4)}" for i in range(1, 1001)]
merchant_ids = random.choices(merchant_ids_list, weights=merchant_weights, k=num_records)

# Payment_Method with realistic distribution
payment_methods = np.random.choice(
    ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"],
    num_records,
    p=[0.6, 0.3, 0.08, 0.02]
)

# Location based on realistic city and state combinations
locations = [f"{fake.city()}, {fake.state_abbr()}" for _ in range(num_records)]

# Device_ID with some reuse (customers use multiple devices)
device_ids = [f"DEV{str(random.randint(1, 10000)).zfill(5)}" for _ in range(num_records)]

# IP_Address with realistic distribution (more common IPs)
ip_addresses = [fake.ipv4_private() if random.random() < 0.95 else fake.ipv4_public() for _ in range(num_records)]

# Customer_Age based on realistic demographics
customer_ages = np.random.normal(loc=40, scale=12, size=num_records).astype(int)
customer_ages = np.clip(customer_ages, 18, 80)

# Gender with balanced distribution
genders = np.random.choice(["Male", "Female", "Other"], num_records, p=[0.48, 0.48, 0.04])

# Income_Bracket based on age
income_brackets = []
for age in customer_ages:
    if age < 25:
        income_brackets.append(np.random.choice(["Low", "Medium"], p=[0.7, 0.3]))
    elif age < 60:
        income_brackets.append(np.random.choice(["Low", "Medium", "High"], p=[0.3, 0.5, 0.2]))
    else:
        income_brackets.append(np.random.choice(["Low", "Medium"], p=[0.6, 0.4]))

# Total_Transactions correlated with income and age
lam_values = []
for income in income_brackets:
    if income == "High":
        lam = 4  # Higher lambda for high income
    elif income == "Medium":
        lam = 3  # Medium lambda
    else:
        lam = 2  # Lower lambda for low income
    lam_values.append(lam)

total_transactions = np.random.poisson(lam=lam_values)
total_transactions = np.clip(total_transactions, 1, 500)

# Avg_Transaction_Amount correlated with income_bracket
avg_transaction_amount = []
for income in income_brackets:
    if income == "Low":
        avg_transaction_amount.append(np.round(np.random.uniform(10, 100), 2))
    elif income == "Medium":
        avg_transaction_amount.append(np.round(np.random.uniform(100, 500), 2))
    else:
        avg_transaction_amount.append(np.round(np.random.uniform(500, 1000), 2))

# Churn_Risk inversely related to Total_Transactions
churn_risk = []
for transactions in total_transactions:
    if transactions > 300:
        churn_risk.append("Low")
    elif transactions > 100:
        churn_risk.append("Medium")
    else:
        churn_risk.append("High")

# Segment_Label based on Total_Transactions and Avg_Transaction_Amount
segment_labels = []
for transactions, avg_amount in zip(total_transactions, avg_transaction_amount):
    if transactions > 300 and avg_amount > 300:
        segment_labels.append("High Spender")
    elif transactions > 100:
        segment_labels.append("Frequent User")
    else:
        segment_labels.append("Dormant")

# Last_Transaction_Date related to Transaction_Date
last_transaction_dates = [
    (dt + timedelta(days=np.random.randint(0, 30))).date()
    for dt in transaction_dates
]

# Now, refine Fraud_Label based on updated churn_risk and other features
for i in range(num_records):
    fraud_prob = base_fraud_prob
    if churn_risk[i] == "High":
        fraud_prob += 0.02  # Increase fraud probability
    if payment_methods[i] in ["Bank Transfer"]:
        fraud_prob += 0.02

    # Assign Fraud_Label based on updated fraud_prob
    fraud_prob = min(fraud_prob, 1)  # Ensure probability does not exceed 1
    fraud_labels[i] = np.random.choice([0, 1], p=[1 - fraud_prob, fraud_prob])

# Now, generate Amounts based on Fraud_Label
for i in range(num_records):
    if fraud_labels[i] == 1:
        # For fraudulent transactions, use a higher log-normal distribution
        amounts[i] = np.round(np.random.lognormal(mean=4.0, sigma=1.5), 2)  # Increased mean and sigma
        amounts[i] = min(amounts[i], 10000)  # Cap the amount
    else:
        # For normal transactions, use a lower log-normal distribution
        amounts[i] = np.round(np.random.lognormal(mean=3.0, sigma=1.0), 2)

# Clip amounts to ensure they are within realistic bounds
amounts = np.clip(amounts, 5, 10000)


#Generate Dates 
insertion_dates = [datetime.now() for _ in range(num_records)]


# Create the DataFrame with updated Amounts and Fraud_Label
data = {
    "transaction_id": transaction_ids,
    "customer_id": customer_ids,
    "transaction_date": transaction_dates,
    "amount": amounts,
    "merchant_id": merchant_ids,
    "payment_method": payment_methods,
    "location": locations,
    "device_id": device_ids,
    "ip_address": ip_addresses,
    "fraud_label": fraud_labels,
    "customer_age": customer_ages,
    "gender": genders,
    "income_bracket": income_brackets,
    "total_transactions": total_transactions,
    "avg_transaction_amount": avg_transaction_amount,
    "churn_risk": churn_risk,
    "segment_label": segment_labels,
    "last_transaction_date": last_transaction_dates,
    "insertion_dates" : insertion_dates


}

df = pd.DataFrame(data)

# Convert data types to match the database schema
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['last_transaction_date'] = pd.to_datetime(df['last_transaction_date'])
df['amount'] = df['amount'].astype(float)
df['fraud_label'] = df['fraud_label'].astype(int)
df['customer_age'] = df['customer_age'].astype(int)
df['total_transactions'] = df['total_transactions'].astype(int)
df['avg_transaction_amount'] = df['avg_transaction_amount'].astype(float)

# # Write DataFrame to PostgreSQL table
# try:
#     df.to_sql('transactions', engine, if_exists='append', index=False)
#     print("Data inserted successfully into the 'transactions' table.")
# except Exception as e:
#     print(f"An error occurred: {e}")

#     # Write DataFrame to PostgreSQL table using a a connection 

try:
    with engine.connect() as connection:
        df.to_sql('transactions', con=connection, if_exists='append', index=False)
    print("Data inserted successfully into the 'transactions' table.")
except Exception as e:
    print(f"An error occurred: {e}")





