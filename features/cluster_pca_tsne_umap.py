import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# 1. Data Simulation and Preprocessing (based on your provided JSON fields: Age, BMI, Bilirubin, ALT, AST, Steatosis, etc.)
# Assume N=39 and contains TX(1)/NTX(0) labels
np.random.seed(42)
n_samples = 39
features = ['Age', 'BMI', 'Bilirubin_tot', 'ALAT', 'ASAT', 'INR', 'Steatosis_pct', 'Creatinine']

X = np.random.randn(n_samples, len(features))
y = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]) # TX vs NTX

X_scaled = StandardScaler().fit_transform(X)

# PCA
pca_res = PCA(n_components=2).fit_transform(X_scaled)

# t-SNE and UMAP
tsne_res = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(X_scaled)
umap_res = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42).fit_transform(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ['(a) PCA', '(b) t-SNE', '(c) UMAP']
data_res = [pca_res, tsne_res, umap_res]
colors = ['#E64B35FF', '#4DBBD5FF'] # ICML/Nature style
labels = ['Non-Transplantable (NTX)', 'Transplantable (TX)']

for i, ax in enumerate(axes):
    for target in [0, 1]:
        idx = (y == target)
        ax.scatter(data_res[i][idx, 0], data_res[i][idx, 1], 
                   c=colors[target], label=labels[target], 
                   alpha=0.8, edgecolors='white', s=80)
    
    ax.set_title(titles[i], fontweight='bold')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True, linestyle='--', alpha=0.6)
    if i == 2:
        ax.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig('clustering_comparison_icml.png', dpi=300)
plt.show()
