import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = np.array([
    [6.6, 3.9, 4.7],
    [7.0, 5.6, 1.3],
    [5.4, 6.8, 5.1]
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print("sklearn PCA результат:")
print(np.round(X_pca, 4))
print("\nОбъясненная дисперсия:")
print(np.round(pca.explained_variance_ratio_, 4))