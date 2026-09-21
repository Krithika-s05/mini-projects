import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Financial Anomaly Detection",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI-Powered Financial Anomaly Detection")
st.subheader("Smart Spending Recommendation System")

st.write(
    "This application detects unusual financial transactions, "
    "analyzes customer spending behavior, and provides smart "
    "spending recommendations using Machine Learning."
)


# =========================================================
# SIDEBAR - DATASET UPLOAD
# =========================================================

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Financial Transactions CSV",
    type=["csv"]
)


# =========================================================
# LOAD DATA
# =========================================================

if uploaded_file is None:

    st.info(
        "Please upload your Financial_Transactions_Clean.csv "
        "file using the sidebar."
    )

    st.stop()


df = pd.read_csv(uploaded_file)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "transaction_id",
    "transaction_date",
    "customer_id",
    "account_type",
    "transaction_type",
    "amount",
    "currency",
    "merchant",
    "transaction_status"
]


missing_columns = [
    column for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "The following required columns are missing:\n\n"
        + ", ".join(missing_columns)
    )

    st.stop()


# =========================================================
# DATA PREPROCESSING
# =========================================================

st.header("🔄 Data Preprocessing")


# Remove duplicate records
df = df.drop_duplicates()


# Convert date
df["transaction_date"] = pd.to_datetime(
    df["transaction_date"],
    errors="coerce"
)


# Convert amount to numeric
df["amount"] = pd.to_numeric(
    df["amount"],
    errors="coerce"
)


# Remove invalid rows
df = df.dropna(
    subset=[
        "transaction_date",
        "amount",
        "customer_id"
    ]
)


# Remove negative/zero amounts
df = df[df["amount"] > 0]


# Standardize text columns
df["account_type"] = (
    df["account_type"]
    .astype(str)
    .str.strip()
    .str.title()
)

df["transaction_type"] = (
    df["transaction_type"]
    .astype(str)
    .str.strip()
    .str.title()
)

df["merchant"] = (
    df["merchant"]
    .astype(str)
    .str.strip()
    .str.title()
)

df["transaction_status"] = (
    df["transaction_status"]
    .astype(str)
    .str.strip()
    .str.title()
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

df["hour"] = df["transaction_date"].dt.hour

df["day"] = df["transaction_date"].dt.day

df["month"] = df["transaction_date"].dt.month

df["day_of_week"] = (
    df["transaction_date"].dt.dayofweek
)


# Customer average transaction amount
customer_average = (
    df.groupby("customer_id")["amount"]
    .mean()
    .rename("customer_average_amount")
)


df = df.merge(
    customer_average,
    on="customer_id",
    how="left"
)


# Customer transaction frequency
customer_frequency = (
    df.groupby("customer_id")
    .size()
    .rename("transaction_frequency")
)


df = df.merge(
    customer_frequency,
    on="customer_id",
    how="left"
)


# Customer total spending
customer_total = (
    df.groupby("customer_id")["amount"]
    .sum()
    .rename("customer_total_spending")
)


df = df.merge(
    customer_total,
    on="customer_id",
    how="left"
)


# =========================================================
# DASHBOARD METRICS
# =========================================================

st.header("📊 Financial Dashboard")


total_transactions = len(df)

total_customers = df["customer_id"].nunique()

total_amount = df["amount"].sum()

average_amount = df["amount"].mean()


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col3.metric(
    "Total Transaction Value",
    f"${total_amount:,.2f}"
)

col4.metric(
    "Average Transaction",
    f"${average_amount:,.2f}"
)


# =========================================================
# ISOLATION FOREST
# =========================================================

st.header("🚨 Anomaly Detection")

st.write(
    "Isolation Forest is used to identify transactions "
    "that are significantly different from normal financial behavior."
)


anomaly_features = [
    "amount",
    "customer_average_amount",
    "transaction_frequency",
    "customer_total_spending"
]


X = df[anomaly_features].copy()


# Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Isolation Forest model
isolation_forest = IsolationForest(
    contamination=0.05,
    random_state=42,
    n_estimators=100
)


df["anomaly_prediction"] = (
    isolation_forest.fit_predict(X_scaled)
)


# -1 = anomaly
#  1 = normal

df["status"] = df[
    "anomaly_prediction"
].map({
    1: "Normal",
    -1: "Anomaly"
})


# Anomaly score
df["anomaly_score"] = (
    isolation_forest
    .decision_function(X_scaled)
)


# =========================================================
# ANOMALY METRICS
# =========================================================

normal_count = (
    df["status"] == "Normal"
).sum()


anomaly_count = (
    df["status"] == "Anomaly"
).sum()


col1, col2 = st.columns(2)


col1.metric(
    "Normal Transactions",
    f"{normal_count:,}"
)


col2.metric(
    "Anomalous Transactions",
    f"{anomaly_count:,}"
)


# =========================================================
# ANOMALY TABLE
# =========================================================

st.subheader("⚠️ Detected Transactions")


result_columns = [
    "transaction_id",
    "transaction_date",
    "customer_id",
    "account_type",
    "transaction_type",
    "amount",
    "merchant",
    "status",
    "anomaly_score"
]


st.dataframe(
    df[result_columns]
    .sort_values(
        "anomaly_score"
    )
    .head(20),
    use_container_width=True
)


# =========================================================
# ANOMALY VISUALIZATION
# =========================================================

fig_anomaly = px.scatter(
    df,
    x="transaction_date",
    y="amount",
    color="status",
    hover_data=[
        "customer_id",
        "merchant",
        "transaction_type"
    ],
    color_discrete_map={
        "Normal": "green",
        "Anomaly": "red"
    },
    title="Financial Transaction Anomalies"
)


st.plotly_chart(
    fig_anomaly,
    use_container_width=True
)


# =========================================================
# TRANSACTION TYPE ANALYSIS
# =========================================================

st.header("💳 Transaction Analysis")


transaction_summary = (
    df.groupby("transaction_type")["amount"]
    .agg(["count", "sum", "mean"])
    .reset_index()
)


transaction_summary.columns = [
    "Transaction Type",
    "Number of Transactions",
    "Total Amount",
    "Average Amount"
]


st.dataframe(
    transaction_summary,
    use_container_width=True
)


fig_transaction = px.bar(
    transaction_summary,
    x="Transaction Type",
    y="Total Amount",
    color="Transaction Type",
    title="Credit vs Debit Transaction Value"
)


st.plotly_chart(
    fig_transaction,
    use_container_width=True
)


# =========================================================
# MERCHANT ANALYSIS
# =========================================================

st.header("🏪 Merchant Spending Analysis")


merchant_summary = (
    df.groupby("merchant")["amount"]
    .sum()
    .reset_index()
    .sort_values(
        "amount",
        ascending=False
    )
)


fig_merchant = px.bar(
    merchant_summary,
    x="merchant",
    y="amount",
    color="merchant",
    title="Spending by Merchant"
)


st.plotly_chart(
    fig_merchant,
    use_container_width=True
)


# =========================================================
# K-MEANS CUSTOMER CLUSTERING
# =========================================================

st.header("👥 Customer Spending Behavior")


customer_data = (
    df.groupby("customer_id")
    .agg(
        total_spending=("amount", "sum"),
        average_transaction=("amount", "mean"),
        transaction_count=("amount", "count")
    )
    .reset_index()
)


cluster_features = [
    "total_spending",
    "average_transaction",
    "transaction_count"
]


cluster_scaler = StandardScaler()


cluster_X = cluster_scaler.fit_transform(
    customer_data[cluster_features]
)


# K-Means
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)


customer_data["cluster"] = (
    kmeans.fit_predict(cluster_X)
)


# Give clusters meaningful names
cluster_average = (
    customer_data.groupby("cluster")
    ["total_spending"]
    .mean()
    .sort_values()
)


cluster_names = {}

names = [
    "Low Spending",
    "Medium Spending",
    "High Spending"
]


for position, cluster in enumerate(
    cluster_average.index
):

    cluster_names[
        cluster
    ] = names[position]


customer_data["spending_group"] = (
    customer_data["cluster"]
    .map(cluster_names)
)


st.dataframe(
    customer_data,
    use_container_width=True
)


fig_cluster = px.scatter(
    customer_data,
    x="average_transaction",
    y="total_spending",
    color="spending_group",
    hover_data=[
        "customer_id",
        "transaction_count"
    ],
    title="Customer Spending Behavior"
)


st.plotly_chart(
    fig_cluster,
    use_container_width=True
)


# =========================================================
# SMART RECOMMENDATION SYSTEM
# =========================================================

st.header("💡 Smart Spending Recommendations")


customer_list = sorted(
    df["customer_id"]
    .unique()
)


selected_customer = st.selectbox(
    "Select Customer",
    customer_list
)


selected_data = df[
    df["customer_id"] == selected_customer
]


# Customer statistics
customer_total_spending = (
    selected_data["amount"].sum()
)


customer_average_spending = (
    selected_data["amount"].mean()
)


customer_transaction_count = (
    len(selected_data)
)


customer_anomalies = (
    selected_data["status"] == "Anomaly"
).sum()


recommendations = []


# Recommendation 1
if customer_anomalies > 0:

    recommendations.append(
        "🚨 Unusual spending detected. "
        "Review the flagged transactions "
        "carefully."
    )


# Recommendation 2
if customer_average_spending > average_amount:

    recommendations.append(
        "💰 Your average transaction is above "
        "the overall average. Consider setting "
        "a spending limit."
    )


# Recommendation 3
if customer_transaction_count > 10:

    recommendations.append(
        "📊 You have a high transaction frequency. "
        "Review recurring expenses and unnecessary "
        "transactions."
    )


# Recommendation 4
if customer_total_spending > (
    df["amount"].sum() / df["customer_id"].nunique()
):

    recommendations.append(
        "📉 Your total spending is relatively high. "
        "Consider creating a monthly budget."
    )


# Recommendation 5
if "Debit" in selected_data[
    "transaction_type"
].values:

    recommendations.append(
        "💳 Monitor debit transactions regularly "
        "and maintain sufficient account balance."
    )


# Default
if len(recommendations) == 0:

    recommendations.append(
        "✅ Your spending pattern appears stable. "
        "Continue monitoring your transactions."
    )


for recommendation in recommendations:

    st.info(
        recommendation
    )


# =========================================================
# CUSTOMER SUMMARY
# =========================================================

st.subheader(
    f"📋 Financial Summary — Customer {selected_customer}"
)


col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Total Spending",
    f"${customer_total_spending:,.2f}"
)


col2.metric(
    "Average Transaction",
    f"${customer_average_spending:,.2f}"
)


col3.metric(
    "Transactions",
    customer_transaction_count
)


col4.metric(
    "Anomalies",
    customer_anomalies
)


# =========================================================
# DOWNLOAD RESULTS
# =========================================================

st.header("📥 Download Results")


csv_data = df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Anomaly Detection Results",
    data=csv_data,
    file_name="financial_anomaly_results.csv",
    mime="text/csv"
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI-Powered Financial Anomaly Detection and "
    "Smart Spending Recommendation System | "
    "Machine Learning Mini Project"
)