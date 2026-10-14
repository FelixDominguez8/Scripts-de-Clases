import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('./wine_fraud.csv')

# Preparacion
df['quality_target'] = df['quality'].map({'Fraud': 1, 'Legit': 0})

variables_continuas = df.select_dtypes(include=['float64', 'int64']).columns
variables_continuas = variables_continuas.drop('quality_target')

X = df[variables_continuas]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Varianza
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8,5))
plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o', linestyle='--')
plt.title('Varianza por componente principal')
plt.xlabel('PC')
plt.ylabel('Varianza')
plt.show()

# Reducción a 2 dimensiones
pca_2 = PCA(n_components=2)
X_pca2 = pca_2.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
sns.scatterplot(x=X_pca2[:,0], y=X_pca2[:,1], hue=df['quality'], palette=['green','red'])
plt.title('PCA 2D')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Tipo de Vino')
plt.show()

# Reducción a 3 dimensiones

pca_3 = PCA(n_components=3)
X_pca3 = pca_3.fit_transform(X_scaled)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca3[:,0], X_pca3[:,1], X_pca3[:,2], c=df['quality_target'], cmap='RdYlGn_r', s=50)
ax.set_title('PCA 3D')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
legend1 = ax.legend(handles=[plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Fraud'),
                             plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Legit')],
                    title="Tipo de Vino")
ax.add_artist(legend1)
plt.show()

print("¿los datos son linealmente separables o no?")
print("No")
