import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=100, centers=3, n_features=3,
random_state=42)
df = pd.DataFrame(X, columns=["Feature1", "Feature2", "Feature3"])
df.dropna(inplace=True)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
linked = linkage(scaled_data, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=45.,
leaf_font_size=10., show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
hc_labels = fcluster(linked, t=3, criterion='maxclust')
df['HC_Cluster'] = hc_labels
db = DBSCAN(eps=0.5, min_samples=5)
db_labels = db.fit_predict(scaled_data)
df['DBSCAN_Cluster'] = db_labels
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Feature1", y="Feature2", hue="HC_Cluster",
palette="Set2")
plt.title("Hierarchical Clustering Result")
plt.show()
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Feature1", y="Feature2", hue="DBSCAN_Cluster",
palette="Set1")
plt.title("DBSCAN Clustering Result")
plt.show()
print("\nInterpretation Guide:")
print("- Hierarchical clustering builds a tree (dendrogram). Cut the tree to decide number of clusters.")
print("- DBSCAN identifies clusters based on density. Noise points (outliers) arelabeled as -1.")
print("- You can adjust `eps` and `min_samples` to tune DBSCAN for yourdataset.")
