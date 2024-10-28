# data_generation.py

from faker import Faker
import numpy as np
import random
from datetime import datetime, timedelta

# Initialize Faker instance
fake = Faker()

# Define the size of the dataset per batch
num_records = 500

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate Transaction_ID and Customer_ID
transaction_ids = [f"TX{str(i).zfill(7)}" for i in range(num_records)]
customer_ids = [f"CUST{str(random.randint(1, 200000)).zfill(6)}" for _ in range(num_records)]

# Corrected Transaction_Date with seasonality
def generate_transaction_date():
    base_prob = 0.90 / 11
    month_probabilities = [base_prob for _ in range(11)]
    month_probabilities.append(1.0 - sum(month_probabilities))  # December
    month = np.random.choice(range(1, 13), p=month_probabilities)
    day = np.random.randint(1, 28)
    hour = np.random.randint(0, 24)
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    return datetime(2023, month, day, hour, minute, second)

transaction_dates = [generate_transaction_date() for _ in range(num_records)]

# Generate other data: Amounts, Merchant_IDs, etc.
amounts = np.zeros(num_records)
fraud_labels = np.zeros(num_records, dtype=int)
base_fraud_prob = 0.03
for i in range(num_records):
    fraud_labels[i] = np.random.choice([0, 1], p=[1 - base_fraud_prob, base_fraud_prob])

merchant_weights = np.random.randint(1, 100, size=1000)
merchant_ids_list = [f"MERCH{str(i).zfill(4)}" for i in range(1, 1001)]
merchant_ids = random.choices(merchant_ids_list, weights=merchant_weights, k=num_records)
payment_methods = np.random.choice(
    ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"],
    num_records, p=[0.6, 0.3, 0.08, 0.02]
)
locations = [f"{fake.city()}, {fake.state_abbr()}" for _ in range(num_records)]
device_ids = [f"DEV{str(random.randint(1, 10000)).zfill(5)}" for _ in range(num_records)]
ip_addresses = [fake.ipv4_private() if random.random() < 0.95 else fake.ipv4_public() for _ in range(num_records)]
customer_ages = np.random.normal(loc=40, scale=12, size=num_records).astype(int)
customer_ages = np.clip(customer_ages, 18, 80)
genders = np.random.choice(["Male", "Female", "Other"], num_records, p=[0.48, 0.48, 0.04])
income_brackets = []
for age in customer_ages:
    if age < 25:
        income_brackets.append(np.random.choice(["Low", "Medium"], p=[0.7, 0.3]))
    elif age < 60:
        income_brackets.append(np.random.choice(["Low", "Medium", "High"], p=[0.3, 0.5, 0.2]))
    else:
        income_brackets.append(np.random.choice(["Low", "Medium"], p=[0.6, 0.4]))

lam_values = []
for income in income_brackets:
    lam_values.append(4 if income == "High" else 3 if income == "Medium" else 2)
total_transactions = np.random.poisson(lam=lam_values)
total_transactions = np.clip(total_transactions, 1, 500)

avg_transaction_amount = []
for income in income_brackets:
    if income == "Low":
        avg_transaction_amount.append(np.round(np.random.uniform(10, 100), 2))
    elif income == "Medium":
        avg_transaction_amount.append(np.round(np.random.uniform(100, 500), 2))
    else:
        avg_transaction_amount.append(np.round(np.random.uniform(500, 1000), 2))

churn_risk = []
for transactions in total_transactions:
    if transactions > 300:
        churn_risk.append("Low")
    elif transactions > 100:
        churn_risk.append("Medium")
    else:
        churn_risk.append("High")

segment_labels = []
for transactions, avg_amount in zip(total_transactions, avg_transaction_amount):
    if transactions > 300 and avg_amount > 300:
        segment_labels.append("High Spender")
    elif transactions > 100:
        segment_labels.append("Frequent User")
    else:
        segment_labels.append("Dormant")

last_transaction_dates = [(dt + timedelta(days=np.random.randint(0, 30))).date() for dt in transaction_dates]

# Bundle all generated data into a dictionary to pass to the insert script
generated_data = {
    "transaction_ids": transaction_ids,
    "customer_ids": customer_ids,
    "transaction_dates": transaction_dates,
    "amounts": amounts,
    "merchant_ids": merchant_ids,
    "payment_methods": payment_methods,
    "locations": locations,
    "device_ids": device_ids,
    "ip_addresses": ip_addresses,
    "fraud_labels": fraud_labels,
    "customer_ages": customer_ages,
    "genders": genders,
    "income_brackets": income_brackets,
    "total_transactions": total_transactions,
    "avg_transaction_amount": avg_transaction_amount,
    "churn_risk": churn_risk,
    "segment_labels": segment_labels,
    "last_transaction_dates": last_transaction_dates
}

# You can now import and use `generated_data` in the insert script
