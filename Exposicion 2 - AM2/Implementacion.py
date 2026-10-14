# Librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

# 1️⃣ Cargar dataset de ejemplo (imágenes de dígitos 8x8)
digits = load_digits()
X = digits.data  # cada fila es una imagen vectorizada (64 pixeles)
y = digits.target  # etiquetas reales (0-9)

print("Forma de la matriz de datos:", X.shape)

# 2️⃣ Aplicar NMF con 16 componentes
nmf = NMF(n_components=16, init='nndsvda', random_state=42, max_iter=500)
W = nmf.fit_transform(X)  # Representación de imágenes en términos de componentes
H = nmf.components_       # Componentes base

print("Forma de W (representación):", W.shape)
print("Forma de H (componentes):", H.shape)

# 3️⃣ Visualización de componentes base (intuición visual)
plt.figure(figsize=(8,4))
for i, comp in enumerate(H):
    plt.subplot(2, 8, i+1)
    plt.imshow(comp.reshape(8,8), cmap='viridis')
    plt.axis('off')
    plt.title(f'C{i+1}')
plt.suptitle("Componentes base descubiertos por NMF")
plt.show()

# 4️⃣ Proyección a 2D usando t-SNE para ver clustering
tsne = TSNE(n_components=2, random_state=42, init='pca')
X_2D = tsne.fit_transform(W)

plt.figure(figsize=(6,6))
for digit in range(10):
    idx = y == digit
    plt.scatter(X_2D[idx,0], X_2D[idx,1], label=str(digit), alpha=0.6)
plt.legend()
plt.title("Proyección 2D de las imágenes usando NMF + t-SNE")
plt.show()
