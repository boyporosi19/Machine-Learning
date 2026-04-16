import json
import re

input_path = '/Users/dppkb/Documents/GitHub/Machine-Learning/clus2/clustering_project.ipynb'
output_path = '/Users/dppkb/Documents/GitHub/Machine-Learning/clus2/clustering_project_pca.ipynb'

with open(input_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0 (Title)
title_src = nb['cells'][0]['source']
for i, line in enumerate(title_src):
    if "Reduksi Dimensi:" in line:
        title_src[i] = "**Reduksi Dimensi:** PCA (Principal Component Analysis)  \n"

# Cell 6 (UMAP Markdown -> PCA Markdown)
nb['cells'][6]['source'] = [
    "---\n",
    "## 4. Reduksi Dimensi dengan PCA\n",
    "Menggunakan **PCA (Principal Component Analysis)** sebagai alternatif linear untuk reduksi dimensi data berdimensi 15.\n",
    "\n",
    "Kita akan mengevaluasi variance explained dan Silhouette score dari rentang komponen.\n"
]

# Cell 7 (PCA Code)
pca_code = """from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

print("1. Melakukan Grid Search untuk PCA n_components berdasarkan Silhouette Score dan Explained Variance")
# Kita akan cari komponen terbaik untuk K=3 (karena minimal 3 klaster)
best_sil = -1
best_n_comp = 2
best_X_pca = None
best_pca_model = None

n_components_list = [2, 3, 4, 5, 6, 8, 10, 12, 14]

print(f"{'n_components':>12} | {'Exp. Variance':>14} | {'K-Means (K=3) Silhouette':>25}")
print("-" * 60)

for n in n_components_list:
    pca = PCA(n_components=n, random_state=rs)
    X_pca = pca.fit_transform(X_scaled)
    
    # Simple kmeans evaluation
    km = KMeans(n_clusters=3, random_state=rs, n_init=10)
    lbl = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, lbl, sample_size=5000, random_state=rs)
    exp_var = sum(pca.explained_variance_ratio_)
    
    print(f"{n:>12} | {exp_var:>14.4f} | {sil:>25.4f}")
    
    # Pilih PCA dengan silhouette terbaik
    if sil > best_sil:
        best_sil = sil
        best_n_comp = n
        best_X_pca = X_pca
        best_pca_model = pca

X_pca_reduced = best_X_pca

print(f"\\n✅ Parameter Terbaik: n_components = {best_n_comp} dengan Silhouette = {best_sil:.4f}")
print(f"Total Explained Variance: {sum(best_pca_model.explained_variance_ratio_):.4f}")

# Plot Scree
pca_full = PCA(n_components=15, random_state=rs).fit(X_scaled)
plt.figure(figsize=(10,4))
plt.bar(range(1, 16), pca_full.explained_variance_ratio_, alpha=0.6, label='Individual')
plt.step(range(1, 16), np.cumsum(pca_full.explained_variance_ratio_), where='mid', label='Cumulative')
plt.axvline(best_n_comp, color='r', linestyle='--', label=f'Best n_comp={best_n_comp}')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.title('Scree Plot')
plt.show()
"""
nb['cells'][7]['source'] = [line + '\n' for line in pca_code.split('\n')]
if nb['cells'][7]['source'][-1] == '\n':
    nb['cells'][7]['source'] = nb['cells'][7]['source'][:-1]

# Cek & Hapus outputs di Cell 7
if 'outputs' in nb['cells'][7]:
    nb['cells'][7]['outputs'] = []

# Cell 8 (Markdown K)
# tidak perlu diubah secara struktural, tapi update info if any
# Cell 9 (K Optimal)
src_9 = "".join(nb['cells'][9]['source'])
src_9 = src_9.replace('X_umap', 'X_pca_reduced')
nb['cells'][9]['source'] = [line + '\n' for line in src_9.split('\n')][:-1]
if 'outputs' in nb['cells'][9]: nb['cells'][9]['outputs'] = []

# Cell 11 (Model Building)
src_11 = "".join(nb['cells'][11]['source'])
src_11 = src_11.replace('X_umap', 'X_pca_reduced')
nb['cells'][11]['source'] = [line + '\n' for line in src_11.split('\n')][:-1]
if 'outputs' in nb['cells'][11]: nb['cells'][11]['outputs'] = []

if 'outputs' in nb['cells'][13]: nb['cells'][13]['outputs'] = []

# Cell 14 (Markdown Vis)
nb['cells'][14]['source'] = [
    "---\n",
    "## 8. Visualisasi Hasil Klasterisasi Terbaik\n",
    "Menampilkan hasil klasterisasi pada ruang PCA 2D menggunakan model terbaik\n"
]

# Cell 15 (Vis Code) - Replace UMAP logic with PCA 2D logic
vis_code = """# Ambil baris terbaik berdasarkan Silhouette tertinggi
best_row = results_df.loc[results_df["Test Silhouette"].idxmax()]
best_model_name = best_row["Model"]
best_split = best_row["Split"]
print(f"Model terbaik  : {best_model_name}\\nSplit terbaik  : {best_split}\\nSilhouette     : {best_row['Test Silhouette']}\\nDBI            : {best_row['Test DBI']}")

import time
print(f"\\nMenjalankan {best_model_name} (K={n_clusters_optimal}) pada seluruh representasi reduksi (X_pca_reduced)...")
start_time = time.time()

if best_model_name == "K-Means":
    best_model = KMeans(n_clusters=n_clusters_optimal, init='k-means++', n_init=10, random_state=rs)
elif best_model_name == "K-Medoids":
    # Untuk dataset 70k, K-Medoids akan sangat lama jika O(n^2), pakai kmeans fallback / subsample
    # Namun demi visualisasi lengkap, kita fit pada seluruh, atau tetap pakai K-Medoids dengan metric cosine
    best_model = KMedoids(n_clusters=n_clusters_optimal, metric='cosine', init='k-medoids++', random_state=rs)
else:
    best_model = AgglomerativeClustering(n_clusters=n_clusters_optimal, linkage='ward')

all_labels = best_model.fit_predict(X_pca_reduced)
print(f"Selesai dalam {time.time() - start_time:.2f} detik")

# Visualisasi PCA 2D (fit transform dari X_scaled lagi hanya untuk visualisasi 2D)
pca_2d = PCA(n_components=2, random_state=rs)
X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for k in range(n_clusters_optimal):
    mask = all_labels == k
    plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], s=5, c=colors[k], label=f'Cluster {k}', alpha=0.5)

plt.title(f"Cluster Distribution in 2D PCA Space\\nModel: {best_model_name} (K={n_clusters_optimal})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(markerscale=5)
plt.tight_layout()
plt.show()
"""
nb['cells'][15]['source'] = [line + '\n' for line in vis_code.split('\n')][:-1]
if 'outputs' in nb['cells'][15]:
    nb['cells'][15]['outputs'] = []

if 'outputs' in nb['cells'][17]:
    nb['cells'][17]['outputs'] = []

# Update conclusion
nb['cells'][18]['source'] = [
    "---\n",
    "## 10. Kesimpulan (PCA Version)\n",
    "\n",
    "Pada analisis ini, kita mengganti UMAP dengan **PCA** sebagai metode reduksi dimensi.\n",
    "- PCA bersifat linier dan tidak mampu memisahkan klaster serealistis UMAP pada fitur yang 90% adalah boolean (gejala).\n",
    "- Silhouette Score menggunakan PCA mungkin saja lebih rendah dibandingkan UMAP karena PCA berusaha menerangkan variansi global daripada lokal struktur data, namun PCA lebih cepat komputasinya.\n",
    "- Evaluasi Model, K-Optimal dan pola profil klaster mengikuti variansi PCA terbaik yang terbentuk.\n"
]

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Disimpan di: {output_path}")
