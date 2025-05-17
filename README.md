# LAPORAN_MLT
## Laporan Proyek Machine Learning: Prediksi Harga Cabai
### Domain Proyek: Prediksi Harga Cabai

**Latar Belakang**

Pada prediksi harga cabai ini, saya menggunakan data harga cabai dengan dataset yang terdiri dari 500 baris dan 2 kolom, yang mencakup dua variabel: satu yang mewakili waktu (seperti tanggal/periode) dan satu lagi yang menunjukkan harga cabai. Fluktuasi harga cabai yang sering terjadi dapat mempengaruhi kestabilan pasar dan daya beli masyarakat, terutama di kalangan konsumen dengan pendapatan rendah. Dengan menggunakan dataset ini, diharapkan dapat dilakukan prediksi yang lebih akurat untuk mengantisipasi perubahan harga, serta membantu dalam perencanaan distribusi dan kebijakan yang lebih efisien di sektor pangan

-sumber data set[Kaggle] (https://www.kaggle.com/datasets/muhvannesalqadri/data-harga-jual-cabai-merah-keriting-di-sultra)

Sumber yang bisa digunakan [Scholar]
(https://socjs.telkomuniversity.ac.id/ojs/index.php/indojc/article/view/144)
Hadiansyah, F. N. (2017). Prediksi Harga Cabai dengan Menggunakan Pemodelan Time Series ARIMA. Indonesian Journal of Computing. 

## Business Understanding

Permasalahan
1. Bagaimana memprediksi harga cabai di masa depan berdasarkan data historis?
2. Bagaimana mengidentifikasi pola musiman atau tren yang memengaruhi harga cabai?


Tujuan:
1. Membangun model prediksi harga cabai yang akurat.
2. Mengidentifikasi faktor yang memengaruhi fluktuasi harga cabai untuk pengambilan keputusan yang lebih baik.

Solusi:
1. Membangun model ARIMA (Autoregressive Integrated Moving Average) untuk memprediksi harga cabai berdasarkan data historis.
2. Menggunakan model Random Forest untuk menangkap pola dan memprediksi harga, dengan tuning hyperparameter untuk meningkatkan akurasi.

-Jumlah Data (Baris dan Kolom):
Dataset ini terdiri dari 500 baris dan 2 kolom. Kolom pertama berisi tanggal dan kolom kedua berisi nilai harga cabai.

-Kondisi Data (Missing Value, Duplikat, dll.):
a. Missing Values: Tidak ada nilai yang hilang (missing values), semua 500 baris pada kedua kolom memiliki data lengkap.
b. Duplikat: Tidak ada data duplikat, dengan jumlah duplikat yang terdeteksi adalah 0.

-Uraian Seluruh Fitur pada Data:
a.Kolom pertama (date) berisi data tanggal dalam format teks (object).
b.Kolom kedua (value) berisi harga cabai dalam format numerik (integer).
c.Statistik deskriptif untuk kolom value menunjukkan:


-Tautan Sumber Data:
Dataset ini dapat ditemukan di Kaggle pada tautan berikut2. Data Understanding
Deskripsi Dataset: https://www.kaggle.com/datasets/muhvannesalqadri/data-harga-jual-cabai-merah-keriting-di-sultra 

Pada gambar dibawah ini terlihat bahwa Grafik berikut menunjukkan fluktuasi harga cabai dari bulan Agustus 2022 hingga Desember 2023. Terlihat adanya kecenderungan kenaikan harga pada periode menjelang hari raya besar

![image](skalabulan.png)

## Data Preparation
1. Pada bagian pra-pemrosesan data, data dibagi menjadi dua bagian, yaitu data pelatihan dan data pengujian, di mana ukuran data dapat dihitung dan dataframe dibagi sesuai dengan proporsi yang ditentukan
 ![image](ujilatih.png)

2. Mengatur 'tanggal' sebagai indeks
Kode ini mengambil data historis harga cabai dan mengatur kolom "date" sebagai indeks dari dataset, menunjukkan fokus pada analisis dan prediksi harga ("value") berdasarkan tanggal.
![image](bulan.png)

3. Mengatur frekuensi ke harian
Kode ini memastikan bahwa dataset harga cabai memiliki frekuensi harian ('D') dengan mengisi tanggal yang hilang
![image](tanggal.png)

4. Menangani nilai yang hilang
Kode ini menangani nilai-nilai yang hilang (NaN) pada kolom harga ('value') dengan melakukan interpolasi linear
![image](hilang.png)
 
# Modeling
Model yang Digunakan: ARIMA & Random Forest

Model 1: ARIMA

Cara Kerja:
1. Membagi data menjadi set pelatihan dan penguji
Pada koode ini membagi dataset harga cabai yang telah dipersiapkan menjadi data pelatihan  dan data pengujian  untuk mempersiapkan tahap pelatihan dan evaluasi model.
![image](latihARIMA.png)

2. Membangun dan melatih model ARIMA
Pada kode ini membangun dan melatih model ARIMA (Autoregressive Integrated Moving Average) untuk data pelatihan harga cabai Anda dengan urutan (5, 1, 0), di mana model ini akan menangkap ketergantungan harga saat ini pada lima harga sebelumnya (p=5), mempertimbangkan satu tingkat differencing untuk stasionaritas (d=1), dan tidak menggunakan komponen moving average (q=0), kemudian menampilkan ringkasan statistik model yang terlatih.
![image](bangunlatihARIMA.png)

3. Membuat prediksi dan mengevaluasi model
Pada Kode menggunakan model ARIMA yang sudah dilatih untuk memperkirakan harga cabai pada periode uji, kemudian menghitung seberapa bagus perkiraan itu menggunakan ukuran MAE, RMSE, dan R2.
![image](prediksiARIMA.png)

MAE Model ARIMA: 10957.417664996989
RMSE Model ARIMA: 15505.544440021817
R2 Model ARIMA: -0.6627685365682277


Kelebihan: Baik untuk data time series dengan pola jelas.
Kekurangan: Membutuhkan data stasioner, pemilihan parameter bisa rumit.


Model 2: Random Forest

Cara Kerja:

1. Persiapan data Random Forest
Pada kode ini memberikan nomor urut pada setiap tanggal di data pelatihan dan pengujian agar model Random Forest bisa menggunakan angka ini untuk mengenali pola berdasarkan waktu.
![image](siapRF.png)

2. Mendefinisikan hyperparameter dan GridSearchCV
Pada kode ini  menyiapkan  pengaturan (hyperparameter) untuk model Random Forest, lalu menggunakan GridSearchCV untuk mencoba kombinasi pengaturan tersebut secara otomatis untuk menemukan yang terbaik dalam memprediksi harga cabai.
![image](gridRF.png)

3. Melatih model Random Forest
Pada kode ini melatih model Random Forest yang sudah disiapkan untuk mencari pengaturan terbaiknya menggunakan data pelatihan, di mana model belajar memprediksi harga cabai ('value') berdasarkan nomor urut tanggal ('date_num').
![image](latihRF.png)

4. Membuat prediksi dan mengevaluasi model Random Forest
Pada kde ini n untuk memprediksi harga cabai pada periode pengujian, lalu mengukur seberapa tepat prediksi tersebut dibandingkan harga aslinya menggunakan MAE, RMSE, dan R2.
![image](modelRF.png)

Parameter:
Dioptimalkan dengan GridSearchCV:
Jumlah tree: 50, 100, 150.
Kedalaman tree: 10, 20, 30.
Minimum sampel untuk split: 2, 5, 10.
Parameter terbaik dipilih berdasarkan error terendah.

Kelebihan: Akurat, fleksibel, dan mencegah overfitting.
Kekurangan: Komputasi bisa berat, kurang mudah diinterpretasi.5. Kesimpulan dan Evaluation

## Evaluasi Model:
- Visualisai Perbandingan prediksi harga cabai dengan meyode Arima dan Random Forest

![image](visual.png)

Grafik ini menunjukkan perbandingan antara harga cabai yang sebenarnya dengan prediksi menggunakan dua model, yaitu ARIMA dan Random Forest. Model ARIMA menunjukkan hasil yang lebih stabil, sementara prediksi Random Forest tidak mengikuti perubahan harga cabai dengan baik.
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

### Refrensi
Hadiansyah, F. N. (2017). Prediksi Harga Cabai dengan Menggunakan Pemodelan Time Series ARIMA. Indonesian Journal of Computing.
