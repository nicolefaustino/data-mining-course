import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the data
data = pd.read_csv("Pokemon.csv")

# Standardize the features (Attack and Defense)
scaler = StandardScaler()
X = scaler.fit_transform(data[["Base Attack", "Base Defense"]])

# Initialize lists to store metrics
inertia = []
silhouette_scores = []
k_range = range(2, 8)  # Test k from 2 to 7 (k=1 is invalid for silhouette)

# Calculate metrics for each k
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot the Elbow Method
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

# Plot the Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', color='orange')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Method")

plt.tight_layout()
plt.show()

# Find the optimal k based on Silhouette Score (higher is better)
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"Optimal k (Silhouette Score): {optimal_k_silhouette}")

# Optional: Visualize clusters for the optimal k
kmeans_optimal = KMeans(n_clusters=optimal_k_silhouette, random_state=42)
data["Cluster"] = kmeans_optimal.fit_predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(data["Base Attack"], data["Base Defense"], c=data["Cluster"], cmap="viridis", s=50)
plt.xlabel("Base Attack")
plt.ylabel("Base Defense")
plt.title(f"Pokémon Clusters (k={optimal_k_silhouette})")
plt.colorbar(label="Cluster")
plt.show()