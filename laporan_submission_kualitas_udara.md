
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

Dataset yang digunakan dalam proyek ini berasal dari UCI Machine Learning Repository, yang berisi data kualitas udara yang diambil secara berkala di kota tertentu di Eropa. Dataset ini mencakup hasil pembacaan berbagai sensor serta data lingkungan pada rentang waktu tertentu.

Sebelum dilakukan pemrosesan, berikut adalah seluruh fitur yang tersedia dalam dataset awal, lengkap dengan deskripsi masing-masing:

| Nama Fitur        | Deskripsi                                                                |
| ----------------- | ------------------------------------------------------------------------ |
| **Date**          | Tanggal pengukuran dalam format DD/MM/YYYY                               |
| **Time**          | Waktu pengukuran dalam format HH.MM.SS                                   |
| **CO(GT)**        | Konsentrasi karbon monoksida dalam ppm (nilai referensi dari alat resmi) |
| **PT08.S1(CO)**   | Output sensor elektro-kimia untuk karbon monoksida (CO)                  |
| **NMHC(GT)**      | Konsentrasi Non-Methane Hydrocarbons dalam µg/m³                         |
| **C6H6(GT)**      | Konsentrasi Benzene dalam µg/m³                                          |
| **PT08.S2(NMHC)** | Output sensor elektro-kimia untuk NMHC                                   |
| **NOx(GT)**       | Konsentrasi Nitrogen Oksida dalam ppb                                    |
| **PT08.S3(NOx)**  | Output sensor elektro-kimia untuk NOx                                    |
| **NO2(GT)**       | Konsentrasi Nitrogen Dioksida dalam µg/m³                                |
| **PT08.S4(NO2)**  | Output sensor elektro-kimia untuk NO2                                    |
| **PT08.S5(O3)**   | Output sensor elektro-kimia untuk Ozon (O₃)                              |
| **T**             | Suhu udara dalam derajat Celcius                                         |
| **RH**            | Kelembaban relatif udara dalam persen                                    |
| **AH**            | Kelembaban absolut udara dalam kg/m³                                     |

### Fitur yang tersedia meliputi:

- CO(GT), NMHC(GT), NOx(GT), NO2(GT)
- Sensor: PT08.S1(CO), PT08.S2(NMHC), PT08.S3(NOx), PT08.S4(NO2), PT08.S5(O3)
- T (temperature), RH (relative humidity), AH (absolute humidity)

Walaupun beberapa fitur seperti Date, Time, atau PT08.S5(O3) tidak digunakan dalam model akhir, tetap penting untuk memahami seluruh konteks awal dari data yang tersedia. Ini membantu menjaga transparansi dan kualitas proses eksplorasi data.

### Target:

- **C6H6(GT)**: Kadar Benzene di udara dalam µg/m³

Sebagian besar nilai numerik memiliki format desimal dengan koma (`,`), dan beberapa fitur memiliki nilai anomali `-200` yang menandakan data hilang.

## Data Preparation

Berikut adalah langkah-langkah data preparation yang dilakukan:

1. Penghapusan Kolom Tidak Relevan
Dua kolom terakhir dalam dataset dihapus karena tidak memiliki nilai yang informatif atau relevan untuk proses analisis dan pemodelan. Langkah ini dilakukan menggunakan:

df = df.iloc[:, :-2]

2. Penanganan Nilai Tidak Valid
Dalam dataset, terdapat nilai -200 yang merupakan indikator nilai tidak tersedia (missing values). Untuk memudahkan analisis, nilai -200 digantikan dengan NaN menggunakan:

df.replace(-200, np.nan, inplace=True)

3. Penghapusan Baris dengan Missing Values
Setelah mengganti nilai tidak valid, semua baris yang mengandung NaN dihapus untuk memastikan data bersih:

df.dropna(inplace=True)

4. Pemilihan Fitur
Tidak semua fitur digunakan untuk pelatihan model. Pemilihan dilakukan berdasarkan eksplorasi awal dan pertimbangan relevansi terhadap target yang ingin diprediksi. Fitur-fitur yang dipilih meliputi:

- CO(GT)

- PT08.S1(CO)

- C6H6(GT)

- PT08.S2(NMHC)

- NOx(GT)

- PT08.S3(NOx)

- NO2(GT)

- PT08.S4(NO2)

- T

- RH

- AH

5. Pembagian Fitur dan Target
Setelah fitur dipilih, dataset dibagi menjadi variabel fitur X dan target y, kemudian dilakukan pembagian data latih dan data uji dengan proporsi 80:20 menggunakan:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

Evaluasi performa model dilakukan menggunakan algoritma Random Forest Classifier. Salah satu metrik yang dianalisis adalah feature importance, yang menunjukkan seberapa besar kontribusi masing-masing fitur terhadap model prediksi.

Berikut adalah hasil feature importance:

### Analisis Feature Importance:
Model Random Forest memberikan nilai feature importance untuk setiap fitur input. Berdasarkan hasil analisis:

| Fitur         | Importance |
| ------------- | ---------- |
| CO(GT)        | 0.000309   |
| PT08.S1(CO)   | 0.042835   |
| C6H6(GT)      | 0.306118   |
| PT08.S2(NMHC) | 0.049213   |
| NOx(GT)       | 0.070057   |
| PT08.S3(NOx)  | 0.062163   |
| NO2(GT)       | 0.108675   |
| PT08.S4(NO2)  | 0.042498   |
| T             | 0.159330   |


PT08.S2(NMHC) merupakan fitur paling berpengaruh terhadap kadar Benzena.

Fitur yang paling berkontribusi dalam memprediksi kadar C6H6(GT) adalah PT08.S2(NMHC). Hal ini sesuai secara logis, mengingat PT08.S2(NMHC) adalah sensor yang mengukur senyawa Non-Methane Hydrocarbon, yang memiliki korelasi tinggi dengan senyawa Benzene (C6H6).

Fitur C6H6(GT) tidak digunakan sebagai input dalam model, melainkan sebagai target prediksi. Oleh karena itu, tidak semestinya dimasukkan ke dalam analisis feature importance.

## Conclusion

Model machine learning berbasis Random Forest berhasil dibangun untuk memprediksi kadar Benzene di udara. Setelah melalui proses pembersihan data, transformasi, dan pelatihan model, sistem ini mampu memberikan prediksi yang akurat.

Model Random Forest Regressor terbukti efektif dalam memprediksi kadar Benzena berdasarkan data sensor kualitas udara. Dengan nilai R² hampir 1 dan MSE sangat kecil, model ini sangat andal. Fitur PT08.S2(NMHC) memiliki kontribusi dominan terhadap prediksi.

 **Menjawab Problem Statement:**

Model mampu memprediksi kadar Benzene berdasarkan data sensor lingkungan.

Berdasarkan hasil pelatihan dan evaluasi model, fitur-fitur berikut diketahui memiliki kontribusi terbesar terhadap performa model:

C6H6(GT) (konsentrasi Benzene)

T (suhu udara)

NO2(GT) (konsentrasi Nitrogen Dioksida)

AH (kelembaban absolut)

Fitur-fitur tersebut memiliki nilai feature importance paling tinggi, sehingga dapat disimpulkan bahwa mereka memainkan peran penting dalam memengaruhi hasil prediksi model.

Kesimpulan ini dibuat berdasarkan fitur yang benar-benar digunakan dalam model (variabel X), sehingga sesuai dengan logika evaluasi dan analisis yang dilakukan.
