#! /usr/bin/env python

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# Generate data
X, y_true = make_blobs(n_samples=1000, centers=[(-5, -5), (5, 5)], cluster_std=[1.0, 1.5], random_state=42)

# Cluster using K-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Calculate ARI
ari = adjusted_rand_score(y_true, kmeans.labels_)
print(f"Adjusted Rand Index: {ari}")

n_samples_list = [100, 500, 1000, 5000, 10000]

for n_samples in n_samples_list:
    # Generate data
    X, y_true = make_blobs(n_samples=n_samples, centers=[(-5, -5), (5, 5)], cluster_std=[1.0, 1.5], random_state=42)

    # Cluster using K-means
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)

    # Calculate ARI
    ari = adjusted_rand_score(y_true, kmeans.labels_)
    print(f"Adjusted Rand Index for n_samples={n_samples}: {ari}")
    
    # Generate data with 3 mixture components
X, y_true = make_blobs(n_samples=1000, centers=[(-5, -5), (5, 5), (0, 0)], cluster_std=[1.0, 1.5, 2.0], random_state=42)

# Cluster using K-means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Calculate ARI
ari = adjusted_rand_score(y_true, kmeans.labels_)
print(f"Adjusted Rand Index for 3 mixture components: {ari}")

# Scatter plot of generated data
plt.scatter(X[:, 0], X[:, 1], c=y_true)
plt.title("Generated data")
plt.show()

# Scatter plot of clustered data
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
