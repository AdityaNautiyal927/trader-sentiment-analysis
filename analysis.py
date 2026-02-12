import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

print("===== ANALYSIS STARTED =====")



DATA_DIR = "data"

sentiment_path = os.path.join(DATA_DIR, "fear_greed_index.csv")
trades_path = os.path.join(DATA_DIR, "historical_data.csv")

if not os.path.exists(sentiment_path):
    print("Sentiment file not found.")
    sys.exit(1)

if not os.path.exists(trades_path):
    print("Trades file not found.")
    sys.exit(1)

sentiment = pd.read_csv(sentiment_path)
trades = pd.read_csv(trades_path)


sentiment["Date"] = pd.to_datetime(sentiment["date"]).dt.date
trades["Date"] = pd.to_datetime(trades["Timestamp IST"], dayfirst=True).dt.date

print("\n--- DATA SHAPES ---")
print("Sentiment:", sentiment.shape)
print("Trades:", trades.shape)



print("\n--- MISSING VALUES ---")
print("\nSentiment:\n", sentiment.isnull().sum())
print("\nTrades:\n", trades.isnull().sum())

sentiment.drop_duplicates(inplace=True)
trades.drop_duplicates(inplace=True)

# Parse dates
sentiment["Date"] = pd.to_datetime(sentiment["date"]).dt.date
trades["Date"] = pd.to_datetime(trades["Timestamp IST"], dayfirst=True).dt.date

# Standardize column names
trades.rename(columns={
    "Closed PnL": "closed_pnl",
    "Size USD": "size_usd",
    "Side": "side",
    "Account": "account"
}, inplace=True)


data = trades.merge(
    sentiment[["Date", "classification"]],
    on="Date",
    how="left"
)

print("\nMerged dataset shape:", data.shape)


data["win"] = (data["closed_pnl"] > 0).astype(int)
data["abs_pnl"] = data["closed_pnl"].abs()

daily_pnl = data.groupby(["account", "Date"])["closed_pnl"].sum().reset_index()


print("\n===== PERFORMANCE BY SENTIMENT =====")
perf = data.groupby("classification")["closed_pnl"].agg(
    ["mean", "std", "sum", "count"]
)
print(perf)

print("\n===== WIN RATE BY SENTIMENT =====")
print(data.groupby("classification")["win"].mean())

print("\n===== AVG TRADE SIZE BY SENTIMENT =====")
print(data.groupby("classification")["size_usd"].mean())

print("\n===== TRADE COUNT BY SENTIMENT =====")
print(data.groupby("classification").size())


print("\n===== RISK ADJUSTED (MEAN / STD) =====")
risk_adj = data.groupby("classification")["closed_pnl"].mean() / \
           data.groupby("classification")["closed_pnl"].std()
print(risk_adj)



if "Fee" in data.columns:
    print("\nNote: Leverage column not available in dataset.")


trader_freq = data.groupby("account").size()
median_freq = trader_freq.median()

data["freq_segment"] = data["account"].map(
    lambda x: "Frequent" if trader_freq[x] > median_freq else "Infrequent"
)

vol = data.groupby("account")["closed_pnl"].std()
median_vol = vol.median()

data["consistency"] = data["account"].map(
    lambda x: "Consistent" if vol[x] < median_vol else "Inconsistent"
)

print("\n===== SEGMENT DISTRIBUTION =====")
print("\nFrequency Segment:")
print(data["freq_segment"].value_counts())

print("\nConsistency Segment:")
print(data["consistency"].value_counts())


os.makedirs("outputs", exist_ok=True)

perf.to_csv("outputs/performance_by_sentiment.csv")
daily_pnl.to_csv("outputs/daily_pnl_per_trader.csv")

print("\nOutputs saved in 'outputs/' folder.")

print("\n===== ANALYSIS COMPLETE =====")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Prepare daily trader level dataset
daily = data.groupby(["account", "Date"]).agg({
    "closed_pnl": "sum",
    "size_usd": "mean",
    "win": "mean",
    "classification": "first"
}).reset_index()

# Target: next day profitability
daily["target"] = (daily["closed_pnl"].shift(-1) > 0).astype(int)

# Encode sentiment
daily["sentiment_code"] = daily["classification"].astype("category").cat.codes

features = ["size_usd", "win", "sentiment_code"]

X = daily[features].fillna(0)
y = daily["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("\n=== BONUS MODEL PERFORMANCE ===")
print(classification_report(y_test, preds))