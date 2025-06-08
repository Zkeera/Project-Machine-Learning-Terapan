
# Laporan Proyek Machine Learning - Prediksi Kualitas Udara (Benzene C6H6)

## Domain Proyek

Kualitas udara merupakan isu penting dalam kesehatan masyarakat, terutama di daerah perkotaan. Salah satu indikator pencemaran udara yang berbahaya adalah kadar Benzene (C6H6), senyawa organik volatil yang bersifat karsinogenik. Proyek ini bertujuan untuk memprediksi kadar benzene berdasarkan data sensor lingkungan, yang dapat dimanfaatkan untuk sistem monitoring polusi udara secara otomatis.

Dataset yang digunakan adalah Air Quality Data Set dari UCI Machine Learning Repository, yang terdiri dari berbagai pengukuran sensor kualitas udara dan meteorologi dari stasiun pemantau di Italia selama beberapa bulan.

Referensi:

* UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml/datasets/Air+Quality](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
* De Vito, S. et al. (2008), "Wireless Sensor Networks for Distributed Chemical Sensing: Addressing Power Consumption Limits with On-Board Intelligence"

## Business Understanding

### Problem Statements

- Bagaimana cara memprediksi kadar Benzene (C6H6) di udara berdasarkan data dari sensor lingkungan?
- Fitur-fitur lingkungan apa saja yang paling berkontribusi terhadap peningkatan kadar benzene?

### Goals

- Membangun model regresi untuk memprediksi kadar Benzene (C6H6).
- Mengidentifikasi fitur sensor yang paling relevan dengan konsentrasi Benzene.

### Solution Statements

- Membangun model regresi menggunakan algoritma **Random Forest Regressor**.
- Melakukan preprocessing data, pembersihan nilai anomali, dan normalisasi fitur.
- Menilai performa model menggunakan metrik regresi seperti **R-squared** dan **MSE**.

## Data Understanding

Dataset terdiri dari 9.357 baris dan 15 kolom, berdasarkan hasil eksplorasi awal menggunakan .shape. Target utama adalah kolom C6H6(GT) yang menunjukkan konsentrasi benzene dalam µg/m³.

### Fitur yang tersedia meliputi:

- CO(GT), NMHC(GT), NOx(GT), NO2(GT)
- Sensor: PT08.S1(CO), PT08.S2(NMHC), PT08.S3(NOx), PT08.S4(NO2), PT08.S5(O3)
- T (temperature), RH (relative humidity), AH (absolute humidity)

### Target:

- **C6H6(GT)**: Kadar Benzene di udara dalam µg/m³

Sebagian besar nilai numerik memiliki format desimal dengan koma (`,`), dan beberapa fitur memiliki nilai anomali `-200` yang menandakan data hilang.

## Data Preparation

Langkah-langkah yang dilakukan:

- Menghapus dua kolom terakhir pada dataset (RH dan AH) karena tidak relevan terhadap target dan memiliki banyak missing value, sesuai eksplorasi awal dengan .info().

- Mengganti nilai -200 dengan NaN sebagai indikasi missing value.

- Menghapus baris dengan missing value pada kolom target dan fitur utama.

- Melakukan pembagian data menjadi training dan testing (80:20 split).

- Mengubah format desimal koma ke titik (',' → '.') dan konversi ke float setelah pembagian data.

- Normalisasi fitur numerik menggunakan StandardScaler.

- Memilih fitur numerik yang relevan sebagai input model.

## Modeling

### Algoritma yang digunakan:

- **Random Forest Regressor**  
  Alasan pemilihan:
  - Dapat menangani data non-linear
  - Robust terhadap noise dan outlier
  - Memberikan informasi feature importance

- **Cara Kerja:**
Random Forest Regressor membangun banyak decision tree dari subset acak data training. Setiap tree menghasilkan prediksi, dan hasil akhirnya diambil sebagai rata-rata dari semua prediksi tree. Pendekatan ini meningkatkan generalisasi dan mengurangi overfitting.

### Proses pelatihan:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
```

## Evaluation

### Metrik Evaluasi:

- **Mean Squared Error (MSE)**
- **R-squared (R²)**

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Hasil Evaluasi:

- **MSE**: 0.0043
- **R²**: mendekati 1, menunjukkan model memiliki akurasi prediksi yang tinggi terhadap target

Model dapat memprediksi kadar Benzene berdasarkan data sensor dengan baik.

### Analisis Feature Importance:
Model Random Forest memberikan nilai feature importance untuk setiap fitur input. Berdasarkan hasil analisis:

| Fitur           | Importance |
| --------------- | ---------- |
| PT08.S5(O3)     | 0.225      |
| PT08.S1(CO)     | 0.180      |
| PT08.S2(NMHC)   | 0.160      |
| T (Temperature) | 0.135      |
| PT08.S3(NOx)    | 0.130      |
| CO(GT)          | 0.090      |
| PT08.S4(NO2)    | 0.080      |

Fitur sensor seperti PT08.S5(O3) dan PT08.S1(CO) adalah yang paling berkontribusi dalam meningkatkan kadar Benzene.

## Conclusion

Model machine learning berbasis Random Forest berhasil dibangun untuk memprediksi kadar Benzene di udara. Setelah melalui proses pembersihan data, transformasi, dan pelatihan model, sistem ini mampu memberikan prediksi yang akurat.

 ** Menjawab Problem Statement: **

Model mampu memprediksi kadar Benzene berdasarkan data sensor lingkungan.

Fitur yang paling berkontribusi terhadap kadar Benzene adalah PT08.S5(O3), PT08.S1(CO), dan PT08.S2(NMHC), berdasarkan nilai feature importance dari model.
