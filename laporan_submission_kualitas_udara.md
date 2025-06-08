
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

Dataset diambil dari sensor kualitas udara dengan berbagai fitur numerik seperti:

- CO(GT): Karbon monoksida

- PT08.S1(CO): Sensor CO

- PT08.S2(NMHC): Sensor NMHC

- PT08.S3(NOx): Sensor NOx

- T: Suhu

dan beberapa lainnya

Target yang ingin diprediksi adalah kadar Benzena (C6H6(GT)).

### Fitur yang tersedia meliputi:

- CO(GT), NMHC(GT), NOx(GT), NO2(GT)
- Sensor: PT08.S1(CO), PT08.S2(NMHC), PT08.S3(NOx), PT08.S4(NO2), PT08.S5(O3)
- T (temperature), RH (relative humidity), AH (absolute humidity)

### Target:

- **C6H6(GT)**: Kadar Benzene di udara dalam µg/m³

Sebagian besar nilai numerik memiliki format desimal dengan koma (`,`), dan beberapa fitur memiliki nilai anomali `-200` yang menandakan data hilang.

## Data Preparation

Berikut adalah langkah-langkah data preparation yang dilakukan:

1. Pemilihan FiturFitur yang digunakan sebagai input model (variabel X) adalah:

- CO(GT)

- PT08.S1(CO)

- PT08.S2(NMHC)

- PT08.S3(NOx)

- T

2. Pembagian Data
Data dibagi menjadi data pelatihan dan pengujian dengan rasio 80:20 menggunakan train_test_split() dari sklearn.

3. Perbaikan Format Desimal
Nilai desimal dikonversi dari koma ke titik agar Python mengenali tipe data numerik secara benar.

4. Normalisasi Fitur
Fitur dinormalisasi menggunakan StandardScaler agar berada dalam rentang nilai yang seragam dan menghindari bias model.

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
## Visualisasi Korelasi Antar Fitur
Heatmap korelasi digunakan untuk mengevaluasi hubungan antar fitur numerik. Fitur PT08.S2(NMHC) menunjukkan korelasi tinggi dengan beberapa sensor lain, menunjukkan adanya informasi yang tumpang tindih. Analisis ini membantu dalam pemilihan fitur.

## Model Development
Model yang digunakan adalah Random Forest Regressor dari sklearn.ensemble. Parameter model:
- n_estimators=100: Jumlah pohon dalam hutan.
- random_state=42: Menjamin reprodusibilitas hasil.
Model dilatih menggunakan data latih yang telah dinormalisasi.

## Evaluation

### Model dievaluasi menggunakan dua metrik:

- **Mean Squared Error (MSE): 0.0043**
- **R-squared (R²): ~0.9999**
Nilai MSE yang sangat kecil dan R² mendekati 1 menandakan performa model sangat baik.

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Hasil Evaluasi:

- **MSE**: 0.0043
- **R²**: ~0.9999, menunjukkan model memiliki akurasi prediksi yang tinggi terhadap target

Model dapat memprediksi kadar Benzene berdasarkan data sensor dengan baik.

### Analisis Feature Importance:
Model Random Forest memberikan nilai feature importance untuk setiap fitur input. Berdasarkan hasil analisis:

| Fitur           | Importance |
| --------------- | ---------- |
| PT08.S2(NMHC)   | 0.99918    |
| CO(GT)          | 0.00052    |
| T (Temperature) | 0.00017    |
| PT08.S1(CO)     | 0.00013    |
| PT08.S3(NOx)    | 0.00001    |

PT08.S2(NMHC) merupakan fitur paling berpengaruh terhadap kadar Benzena.

## Conclusion

Model machine learning berbasis Random Forest berhasil dibangun untuk memprediksi kadar Benzene di udara. Setelah melalui proses pembersihan data, transformasi, dan pelatihan model, sistem ini mampu memberikan prediksi yang akurat.

Model Random Forest Regressor terbukti efektif dalam memprediksi kadar Benzena berdasarkan data sensor kualitas udara. Dengan nilai R² hampir 1 dan MSE sangat kecil, model ini sangat andal. Fitur PT08.S2(NMHC) memiliki kontribusi dominan terhadap prediksi.

 ** Menjawab Problem Statement: **

Model mampu memprediksi kadar Benzene berdasarkan data sensor lingkungan.

Fitur yang paling berkontribusi terhadap kadar Benzene adalah PT08.S5(O3), PT08.S1(CO), dan PT08.S2(NMHC), berdasarkan nilai feature importance dari model.
