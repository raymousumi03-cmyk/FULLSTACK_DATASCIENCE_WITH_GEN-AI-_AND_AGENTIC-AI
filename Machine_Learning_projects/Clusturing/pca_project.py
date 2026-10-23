from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target

# Instantiate the PCA Model
pca = PCA(n_components=2)

# Fit and Transform data
x_pca = pca.fit_transform(x)

# Visualize the reduced-dimensional data
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1],hue=y, palette='viridis', s=50)
plt.title('PCA: Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()