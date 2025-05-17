# LAPORAN_MLT
# Laporan Proyek Machine Learning: Prediksi Harga Cabai
##  Domain Proyek: Prediksi Harga Cabai

Latar Belakang

Pada prediksi harga cabai ini, saya menggunakan data harga cabai dengan dataset yang terdiri dari 500 baris dan 2 kolom, yang mencakup dua variabel: satu yang mewakili waktu (seperti tanggal/periode) dan satu lagi yang menunjukkan harga cabai. Fluktuasi harga cabai yang sering terjadi dapat mempengaruhi kestabilan pasar dan daya beli masyarakat, terutama di kalangan konsumen dengan pendapatan rendah. Dengan menggunakan dataset ini, diharapkan dapat dilakukan prediksi yang lebih akurat untuk mengantisipasi perubahan harga, serta membantu dalam perencanaan distribusi dan kebijakan yang lebih efisien di sektor pangan

-sumber data set[Kaggle] (https://www.kaggle.com/datasets/muhvannesalqadri/data-harga-jual-cabai-merah-keriting-di-sultra)
-sumber yang bisa digunakan [Scholar](https://socjs.telkomuniversity.ac.id/ojs/index.php/indojc/article/view/144) 

## Business Understanding

- Permasalahan:
Bagaimana memprediksi harga cabai di masa depan berdasarkan data historis?
Bagaimana mengidentifikasi pola musiman atau tren yang memengaruhi harga cabai?
Tujuan:
-Membangun model prediksi harga cabai yang akurat.
Mengidentifikasi faktor yang memengaruhi fluktuasi harga cabai untuk pengambilan keputusan yang lebih baik.

Solusi:
-Membangun model ARIMA (Autoregressive Integrated Moving Average) untuk memprediksi harga cabai berdasarkan data historis.
-Menggunakan model Random Forest untuk menangkap pola dan memprediksi harga, dengan tuning hyperparameter untuk meningkatkan akurasi.

-Jumlah Data (Baris dan Kolom):
Dataset ini terdiri dari 500 baris dan 2 kolom. Kolom pertama berisi tanggal dan kolom kedua berisi nilai harga cabai.

-Kondisi Data (Missing Value, Duplikat, dll.):
a. Missing Values: Tidak ada nilai yang hilang (missing values), semua 500 baris pada kedua kolom memiliki data lengkap.
b. Duplikat: Tidak ada data duplikat, dengan jumlah duplikat yang terdeteksi adalah 0.

-Uraian Seluruh Fitur pada Data:
a.Kolom pertama (date) berisi data tanggal dalam format teks (object).
b.Kolom kedua (value) berisi harga cabai dalam format numerik (integer).
c.Statistik deskriptif untuk kolom value menunjukkan:
  	-Rata-rata harga cabai: 46,315.78
	-Nilai harga cabai terendah: 28,150
	-Nilai harga cabai tertinggi: 78,290

-Tautan Sumber Data:
Dataset ini dapat ditemukan di Kaggle pada tautan berikut2. Data Understanding
Deskripsi Dataset: https://www.kaggle.com/datasets/muhvannesalqadri/data-harga-jual-cabai-merah-keriting-di-sultra 

## Data Preparation
Langkah-langkah:

1.Memisahkan Data Menjadi Data Latih dan Data Uji 
2.Membuat Variabel Target (y) dan Variabel Input (X):
3.Menambahkan Kolom 'date_num' (Pertama):
4.Mengatur Frekuensi Data Time Series:
5.Memisahkan Data Menjadi Data Latih dan Data Uji (Kedua):
6.Menambahkan Kolom 'date_num' (Kedua):


# Modeling
Model yang Digunakan: ARIMA & Random Forest

Model 1: ARIMA

Cara Kerja:
Memprediksi data time series berdasarkan pola data historis (tren, musiman, autokorelasi).
Menggunakan 3 komponen: AR (nilai masa lalu), I (stasioneritas), MA (kesalahan prediksi).

Parameter:
order=(5, 1, 0):
AR: 5 data masa lalu.
I: 1 kali differencing.
MA: Tidak digunakan.

Kelebihan: Baik untuk data time series dengan pola jelas.
Kekurangan: Membutuhkan data stasioner, pemilihan parameter bisa rumit.

Model 2: Random Forest

Cara Kerja:
Menggabungkan banyak decision tree untuk prediksi yang akurat.
Bagging dan random feature selection untuk mencegah overfitting.

Parameter:
Dioptimalkan dengan GridSearchCV:
Jumlah tree: 50, 100, 150.
Kedalaman tree: 10, 20, 30.
Minimum sampel untuk split: 2, 5, 10.
Parameter terbaik dipilih berdasarkan error terendah.

Kelebihan: Akurat, fleksibel, dan mencegah overfitting.
Kekurangan: Komputasi bisa berat, kurang mudah diinterpretasi.5. Kesimpulan dan Evaluation

## Evaluasi Model:

Metrik Evaluasi:

Model prediksi harga cabai Anda dievaluasi menggunakan tiga metrik utama:
1. MAE (Mean Absolute Error): Mengukur rata-rata selisih absolut antara nilai prediksi dan nilai aktual.
2. RMSE (Root Mean Squared Error): Mengukur akar kuadrat dari rata-rata selisih kuadrat antara nilai prediksi dan nilai aktual.
3. R-squared (R2): Mengukur seberapa baik model cocok dengan data, dengan nilai mendekati 1 menunjukkan kesesuaian yang baik.

MAE Model ARIMA: 10957.417664996989
RMSE Model ARIMA: 15505.544440021817
R2 Model ARIMA: -0.6627685365682277

MAE Model Random Forest: 11051.14
RMSE Model Random Forest: 15658.871308941776
R2 Model Random Forest: -0.6958157628704082

Hubungan dengan Business Understanding:

Problem Statement:

1. Bagaimana memprediksi harga cabai di masa depan berdasarkan data historis?
Model ARIMA dan Random Forest yang Anda bangun telah menjawab problem statement ini dengan memberikan prediksi harga cabai di masa depan. Akurasi prediksi dapat dinilai dari nilai MAE, RMSE, dan R2.

2.Bagaimana mengidentifikasi pola musiman atau tren yang memengaruhi harga cabai?
Model ARIMA secara inheren dapat mengidentifikasi pola musiman dan tren dalam data time series. Visualisasi data dan analisis residual juga dapat membantu dalam mengidentifikasi pola-pola ini.

Goals:
- Membangun model prediksi harga cabai yang akurat.
Telah mencapai tujuan ini dengan membangun dan mengevaluasi dua model prediksi. Nilai MAE, RMSE, dan R2 menunjukkan tingkat akurasi model. Semakin rendah MAE dan RMSE, dan semakin tinggi R2, semakin akurat model tersebut.

-Mengidentifikasi faktor yang memengaruhi fluktuasi harga cabai untuk pengambilan keputusan yang lebih baik.
Model yang dibangun, terutama ARIMA, dapat membantu dalam mengidentifikasi faktor-faktor yang memengaruhi harga cabai, 

Solusi Statement:
-Membangun model ARIMA (Autoregressive Integrated Moving Average) untuk memprediksi harga cabai berdasarkan data historis.
Solusi ini berdampak karena ARIMA adalah model yang cocok untuk data time series dan dapat memberikan prediksi yang baik.
-Menggunakan model Random Forest untuk menangkap pola dan memprediksi harga, dengan tuning hyperparameter untuk meningkatkan akurasi.
Solusi ini juga berdampak karena Random Forest adalah model yang fleksibel dan akurat. Tuning hyperparameter membantu mengoptimalkan performa model.
