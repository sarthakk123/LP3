import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ---------- 1. Load ----------
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")   # file has special characters
print("Shape:", df.shape)

# ---------- 2. Select numeric columns useful for grouping ----------
cols = ["QUANTITYORDERED", "PRICEEACH", "SALES", "MSRP"]
X = df[cols].dropna()
print("\nMissing values in chosen columns:", df[cols].isnull().sum().sum())

# scale so that big numbers (SALES) don't dominate small ones
X_scaled = StandardScaler().fit_transform(X)

# ---------- 3. Elbow method ----------
print("\n k    inertia (WCSS)   silhouette")
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertia.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_) if k > 1 else float("nan")
    print("%2d   %12.2f   %8.4f" % (k, km.inertia_, sil))

plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of clusters (k)"); plt.ylabel("WCSS (inertia)")
plt.title("Elbow method"); plt.grid(True)
plt.savefig("elbow.png", dpi=100, bbox_inches="tight"); plt.close()

# ---------- 4. Final K-Means ----------
K = 3
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X_scaled)
X = X.copy()
X["Cluster"] = km.labels_
print("\nCluster sizes:\n", X["Cluster"].value_counts().sort_index().to_string())
print("\nAverage values per cluster:\n", X.groupby("Cluster").mean().round(2))

# ---------- 5. Hierarchical clustering (same k) ----------
hc = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(X_scaled)
print("\nHierarchical cluster sizes:\n", pd.Series(hc.labels_).value_counts().sort_index().to_string())
