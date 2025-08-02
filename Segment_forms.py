import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

st.title("🛒 Wholesale Customer Segmentation App")
st.write("""
Upload your wholesale customer data to automatically assign each customer to a segment based on annual spending.
Segments may include: High Spender, Budget, or Mid-Range.
""")

# Demo data link or use your own
sample = pd.read_csv('Wholesale customers data.csv')  # Ensure this file is in your app folder

uploaded_file = st.file_uploader("Upload your customer CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
else:
    st.info("No file uploaded. Using sample dataset for demonstration.")
    df = sample.copy()

st.write("### Data Preview", df.head())

# Select features
features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = df[features]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method (hidden from user, optional)
# Uncomment to display elbow plot
# inertia = []
# for k in range(1, 11):
#     km = KMeans(n_clusters=k, random_state=42)
#     km.fit(X_scaled)
#     inertia.append(km.inertia_)
# st.line_chart(pd.DataFrame({'k': range(1,11), 'inertia': inertia}).set_index('k'))

# Cluster
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Label clusters by mean spending
cluster_means = df.groupby('Cluster')[features].mean()
labels = {}
for idx in cluster_means.index:
    if cluster_means.loc[idx].mean() == cluster_means.mean(axis=1).max():
        labels[idx] = 'High Spender'
    elif cluster_means.loc[idx].mean() == cluster_means.mean(axis=1).min():
        labels[idx] = 'Budget'
    else:
        labels[idx] = 'Mid-Range'
df['Segment'] = df['Cluster'].map(labels)

st.write("### Segmentation Result")
st.dataframe(df[['Channel', 'Region', 'Cluster', 'Segment'] + features].head(10))

st.write("#### Segment Counts")
st.bar_chart(df['Segment'].value_counts())

# Download segmented data
csv = df.to_csv(index=False).encode()
st.download_button("Download Segmented Data", csv, "segmented_customers.csv", "text/csv")

st.info("Interpretation: High Spenders are likely valuable customers; Budgets are cost-sensitive; Mid-Range are balanced buyers.")
