# ML Workflow Studio

Aplikasi Streamlit untuk analisis data dan membangun model machine learning dari file CSV apa pun,
dalam tujuh langkah berurutan.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Alur kerja

| # | Langkah | Isi | Syarat |
|---|---|---|---|
| 1 | **Data** | Unggah CSV, ringkasan & profil kolom | — |
| 2 | **Eksplorasi** *(opsional)* | EDA lanjutan (lihat di bawah) | Data |
| 3 | **Visualisasi** *(opsional)* | Pembuat grafik 17 jenis + dashboard otomatis | Data |
| 4 | **Persiapan** | Target, fitur, fitur turunan, preprocessing, seleksi fitur, cek kebocoran data | Data |
| 5 | **Training** | Banyak algoritma, repeated k-fold CV, baseline, tuning, riwayat eksperimen | Konfigurasi tersimpan |
| 6 | **Evaluasi** | Kinerja, diagnostik, interpretasi (SHAP/PDP), kalibrasi & threshold | Model |
| 7 | **Prediksi** | Prediksi manual/batch, unduh model `.pkl` | Model |

Mengubah langkah awal otomatis mengosongkan hasil langkah sesudahnya (data baru → konfigurasi & model dihapus;
konfigurasi baru → model dihapus), sehingga hasil yang ditampilkan selalu konsisten.

### Eksplorasi (EDA lanjutan)
- **Ringkasan**: temuan otomatis (missing, kolom ID/konstan, skewness, outlier, korelasi tinggi, relasi pasti, target tidak seimbang).
- **Kualitas**: skewness, kurtosis, outlier IQR, kategori langka, tipe campuran.
- **Missing**: persentase, pola missing antar baris, dan hubungan kosong-nya data dengan target (MAR/MNAR).
- **Distribusi**: histogram + box, Q–Q plot, uji Shapiro–Wilk, opsi log.
- **Outlier**: box plot terstandar semua kolom + daftar baris outlier.
- **Korelasi**: Pearson / Spearman / Kendall, dan asosiasi campuran (|r|, Cramér's V, η).
- **Target**: distribusi & rasio kelas, ranking *mutual information*, fitur vs target.
- **Multikolinearitas**: VIF dan deteksi kolom yang merupakan kombinasi linear pasti dari kolom lain.
- **Dimensi**: PCA (scree plot, proyeksi 2D, loadings) dan t-SNE.
- **Uji statistik**: t-test/ANOVA + Mann–Whitney/Kruskal–Wallis (dengan effect size), chi-square, normalitas.

### Training & evaluasi
- Metrik utama bisa dipilih (F1, akurasi, balanced accuracy, ROC-AUC / R², RMSE, MAE, MAPE).
- Ranking memakai **repeated k-fold CV** (mean ± std) pada data latih; skor data uji tetap ditampilkan.
- **Baseline Dummy** sebagai pembanding; peringatan jika tidak ada model yang mengalahkannya.
- **Tuning** `RandomizedSearchCV`, *class weight* seimbang, HistGradientBoosting dan XGBoost.
- Preprocessing: fitur turunan (A ÷ − + × B), log-transform kolom, potong outlier (IQR), Standard/Min-Max/Robust
  scaler, One-Hot/Ordinal/Target encoding, penggabungan kategori langka, log-transform target, SelectKBest.
- **Cek kebocoran data**: mendeteksi fitur yang (hampir) menentukan target, termasuk target yang merupakan
  kombinasi linear fitur, dengan tombol untuk membuangnya.
- Evaluasi: confusion matrix, ROC (OvR), residual; learning curve + diagnosis bias/varians; analisis error per
  baris & per segmen; permutation importance; **SHAP** (global, beeswarm, waterfall per baris); **PDP & ICE**;
  reliability diagram, Brier score, dan pengaturan threshold untuk klasifikasi biner.
- Clustering: K-Means, Agglomerative, DBSCAN, bantuan memilih k (elbow + silhouette) dan eps (k-distance),
  profil cluster.

Preprocessing tidak mengubah data mentah: semua langkah dibungkus bersama model dalam satu `sklearn.Pipeline`,
sehingga data baru di halaman Prediksi cukup diisi dalam bentuk aslinya.

## Struktur kode

```
app.py                  titik masuk: navigasi & sidebar
mlapp/state.py          session state, urutan langkah, status (selesai/siap/terkunci)
mlapp/ml.py             preprocessing, training, CV, tuning, cek kebocoran (tanpa Streamlit)
mlapp/eda.py            statistik EDA lanjutan (tanpa Streamlit)
mlapp/explain.py        SHAP, PDP, learning curve, kalibrasi, analisis error (tanpa Streamlit)
mlapp/charts.py         pembuat grafik generik untuk halaman Visualisasi
mlapp/ui.py             sistem desain & komponen UI bersama
mlapp/views/*.py        satu file per halaman
.streamlit/config.toml  tema (warna, font, palet grafik)
```

Model yang diunduh adalah `dict` hasil `pickle` berisi `pipeline`, `inputs` (kolom mentah yang dibutuhkan),
`features`, `derived`, `target`, `classes`, metrik uji & CV, parameter terbaik, versi scikit-learn, dan hash data.
Untuk memuatnya kembali, paket `mlapp` harus bisa di-import.

`fifaDatasets.csv` hanya data uji dan tidak dipakai oleh aplikasi.
