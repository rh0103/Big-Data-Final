import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from Autoencoder import Autoencoder
from Autoencoder import DenoisingAutoencoder
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 載入資料
X = pd.read_csv("public_data.csv").values  # or private_data.csv
n_samples, n_features = X.shape

# 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# # Autoencoder 訓練與特徵提取
# autoencoder = Autoencoder(input_dim=n_features, encoding_dim=3)
# autoencoder.fit(X_scaled, epochs=100, batch_size=16)

# X_latent = autoencoder.transform(X_scaled)  # shape: (n_samples, latent_dim)
# dae = DenoisingAutoencoder(
#     input_dim=n_features,
#     encoding_dim=4,       # 比3稍寬一點點，保有些資訊
#     noise_factor=0.08    # 非常微小的 noise，防止照抄
# )
# dae.fit(X_scaled, epochs=30, batch_size=16)
# X_latent = dae.transform(X_scaled)

ae = Autoencoder(input_dim=n_features, encoding_dim=4)
ae.fit(X_scaled, epochs=34, batch_size=16)
X_latent = ae.transform(X_scaled)

# Clustering (4n - 1)
n_clusters = 4 * n_features - 1
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_latent)
cluster_labels = kmeans.labels_

# 輸出結果
output = pd.DataFrame({
    "id": np.arange(len(cluster_labels)),
    "label": cluster_labels
})
output.to_csv("public_submission.csv", index=False)

# X_vis = TSNE(n_components=2, random_state=0).fit_transform(X_latent)
# plt.scatter(X_vis[:, 0], X_vis[:, 1], c=cluster_labels, cmap='tab10', s=10)
# plt.title("Clustering Result in Latent Space (via t-SNE)")
# plt.savefig("tsne_clustering.png")
# plt.show()