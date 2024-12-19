import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

# Data palsu baru tentang pengelolaan sampah
data = {
    'Daerah': ['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Semarang', 'Yogyakarta', 'Makassar', 'Palembang'],
    'Jumlah_Penduduk': [10000000, 3000000, 2500000, 2000000, 1600000, 1000000, 1500000, 1200000],
    'Sampah_PerOrang': [1.8, 1.5, 2.0, 1.3, 1.4, 1.2, 1.6, 1.4],  # dalam kg
    'Partisipasi_Kebersihan': [0.9, 0.8, 0.5, 0.6, 0.7, 0.4, 0.3, 0.5]  # skala 0-1
}

# Membuat DataFrame
df = pd.DataFrame(data)

# Mengambil fitur untuk clustering
X = df[['Jumlah_Penduduk', 'Sampah_PerOrang', 'Partisipasi_Kebersihan']].values

# Menggunakan KMeans untuk clustering
kmeans = KMeans(n_clusters=3)  # Mengelompokkan menjadi 3 cluster
df['Cluster'] = kmeans.fit_predict(X)

# Menghitung total sampah yang dihasilkan
df['Total_Sampah'] = df['Jumlah_Penduduk'] * df['Sampah_PerOrang']

# Menampilkan hasil
print(df[['Daerah', 'Cluster', 'Total_Sampah']])
print("\nPusat Cluster: ")
print(kmeans.cluster_centers_)