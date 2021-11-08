# Laporan Proyek Machine Learning - Raphael Discky

## IBM Stock Prediction - LSTM vs GRU

Saham merupakan salah satu pilihan investasi yang menarik karena dapat diperoleh untung yang besar dibandingkan dengan usaha lainnya. Untuk meminimalkan resiko kerugian, diperlukan perhatian yang jeli terhadap pergerakan saham dan perkembangan pasar modal merupakan salah satu indikator yang perlu dipantau. Dengan teknologi pemrosesan prediksi dan pembelajaran mesin saat ini, identifikasi prediksi harga saham dapat dilakukan secara otomatis. Deep Learning merupakan salah satu bagian dari pembelajaran mesin, dan memiliki akurasi pengenalan yang tinggi dengan data yang sangat banyak. Proyek ini menggunakan history harga saham dari IBM dan membandingkan metode RNN antara LSTM dan GRU untuk melakukan prediksi terhadap nilai saham dari IBM. 

  Format Referensi: [Predicting Stock Prices Using LSTM](https://www.researchgate.net/profile/Murtaza-Roondiwala/publication/327967988_Predicting_Stock_Prices_Using_LSTM/links/5bafbe6692851ca9ed30ceb9/Predicting-Stock-Prices-Using-LSTM.pdf) 

## Business Understanding
### Problem Statements

Berdasarkan latar belakang di atas, masalah dalam proyek dapat dirumuskan menjadi:
- Bagaimana merancang dan membangun model deep learning untuk prediksi harga saham menggunakan algoritma LSTM dan GRU 
- Bagaimana perbandingan hasil dengan metode LSTM dan GRU dalam memprediksi harga saham?

### Goals

Berdasarkan problem statement di atas, maka tujuan proyek ini adalah:
- Mengetahui cara untuk merancang dan membangun model deep learning untuk prediksi harga saham menggunakan algoritma LSTM dan GRU 
- Mengetahui metode yang lebih baik antara LSTM dan GRU dalam memprediksi harga saham

#### Solution statements
Pada tahap ini kita akan membandingkan dua metode LSTM dengan GRU untuk mendapatkan hasil prediksi harga saham yang terbaik. LSTM (Long Short Term Memory) adalah jenis modul pemrosesan lain untuk RNN. LSTM diciptakan oleh Hochreiter & Schmidhuber (1997) dan kemudian dikembangkan dan dipopulerkan oleh banyak periset. Seperti RNN, jaringan LSTM (LSTM network) juga terdiri dari modul-modul dengan pemrosesan berulang. mempunyai banyak sekali varian, misalnya LSTM dengan koneksi lubang intip (peephole connection), LSTM yang menggabungkan gerbang input dengan gerbang lupa, dan sebagainya. Salah satu varian yang populer adalah Gated Recurrent Unit atau disingkat GRU. GRU dimunculkan dalam makalah oleh Cho dkk (2014) dan Chung dkk (2014).
Kelebihan dari LSTM adalah ia lebih akurat ketika menggunakan dataset dengan sequences yang lebih panjang. Sedangkan kekurangannya adalah LSTM lebih kompleks dan lambat dibandingkan dengan GRU karena memiliki 3 gates. Untuk GRU kelebihan dari metode ini adalah lebih cepat dibandingkan LSTM karena hanya memiliki 2 gates saja tetapi GRU kurang akurat dibandingkan dengan jika menggunakan LSTM.

## Data Understanding

Data yang digunakan untuk proyek ini diambil dari Yahoo Finance: [IBM - Stock Price ](https://finance.yahoo.com/quote/IBM/history?period1=1370390400&period2=1636070400&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true). 

Berikut ini adalah dataframe dari IMB Stock Price:
![Image of Yaktocat](https://i.ibb.co/jh3fc0Z/Screenshot-2021-11-07-142304.png)

Pada dataset tersebut berisi sejarah harga stock pada perusahaan IBM dengan jumlah data sebanyak 10440 dengan 7 kolom yaitu Date, Open, High, Low, Close, Adj Close, dan Volume. Tetapi pada proyek ini kita akan hanya menggunakan kolom Date dan harga saham IBM pada waktu Close atau harga penutupan pada tanggal tertentu untuk membuat prediksi harga saham perusahaan IBM. 

### Variabel-variabel pada IBM Stock Price dataset adalah sebagai berikut:
- date : merupakan tanggal perubahan harga saham setiap satu hari
- open : merupakan harga pembukaan pada satu hari
- high : merupakan pencapaian harga tertinggi dalam satu hari
- low : merupakan pencapaian harga terendah dalam satu hari
- close : merupakan harga penutupan dalam satu hari
- dll

#### Visualisasi Harga Saham IBM dari tahun 1980 sampai 2021 berdasarkan Close Price
![Image of Saham](https://i.ibb.co/vsgxqgR/download.png)

## Data Preparation
Teknik data preparation yang dilakukan adalah membagi dataset menjadi data latih (train) dan data  uji (test). Selanjutnya melakukan scaling untuk fitur-fitur menggunakan normalisai dengan menggunakan fungsi MinMaxScaler. Normalisasi data dilakukan karena dapat membantu algoritma dalam konvergen, misalnya mencari local/global minimum secara efisien.

Membagi data kita menjadi data latih dan uji:
```
num_shape = 10000

train = df.iloc[:num_shape, 1:2].values
test = df.iloc[num_shape:, 1:2].values
```

Melakukan scaling untuk fitur menggunakan fungsi MinMaxScaler dengan range antara 0 sampai 1 untuk data latih:
```
sc = MinMaxScaler(feature_range = (0, 1))
train_scaled = sc.fit_transform(train)
```
## Modeling
#### Menggunakan model LSTM 
- Kelebihannya adalah LSTM lebih akurat ketika menggunakan dataset dengan sequences yang lebih panjang
- Kekurangannya adalah LSTM lebih complex dan lambat dibandingkan dengan GRU karena memiliki 3 gates
- Hyperparameter tuning dilakukan dengan mendiagnosis jumlah epoch dan membanding hasilnya dengan jumlah epoch yang lain

Pada proyek summary dari model LSTM yang digunakan adalah sebagai berikut:
![LSTM model](https://i.ibb.co/rpTzc97/Screenshot-2021-11-07-144306.png)

Model LSTM dilatih dengan menggunakan optimizer adam dan loss yang digunakan adalah MSE, epoch sebanyak 20 dan batch size 32.
Hasil pelatihan model adalah sebagai berikut:
![LSTM model](https://i.ibb.co/3TFXL14/Screenshot-2021-11-07-144731.png)

#### Menggunakan GRU
- Kelebihannya adalah GRU lebih cepat dibandingkan LSTM karena hanya memiliki 2 gates saja
- Kekurangannya adalah GRU kurang akurat dibandingkan dengan LSTM
- Hyperparameter tuning dilakukan dengan mendiagnosis jumlah epoch dan membanding hasilnya dengan jumlah epoch yang lain

Pada proyek summary dari model GRU yang digunakan adalah sebagai berikut:
![GRU model](https://i.ibb.co/BwR6bxm/Screenshot-2021-11-07-144325.png)

Model GRU dilatih dengan menggunakan optimizer sgd dan loss yang digunakan adalah MSE, epoch sebanyak 20 dan batch size 32.
Hasil pelatihan model adalah sebagai berikut:

![GRU model](https://i.ibb.co/ZVfsr4L/Screenshot-2021-11-07-144745.png)

#### Membandingkan Hasil MSE dari Kedua Model
##### Model LSTM
![LSTM model](https://i.ibb.co/qggF12X/Screenshot-2021-11-07-145230.png)
##### Model GRU
![LSTM model](https://i.ibb.co/r79vVGc/Screenshot-2021-11-07-145259.png)

Dari hasil kedua model tersebut dapat disimpulkan bahwa solusi yang terbaik adalah menggunakan model LSTM karena memiliki nilai MSE yang lebih rendah dibandingkan jika menggunakan model GRU.

## Evaluation
Metriks evaluasi menggunakan MSE (Mean Squared Error). Metode Mean Squared Error secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada peramalan. Nilai Mean Squared Error yang rendah atau nilai mean squared error mendekati nol menunjukkan bahwa hasil peramalan sesuai dengan data aktual dan bisa dijadikan untuk perhitungan peramalan di periode mendatang.

Cara kerja  Mean Squared Error (MSE) adalah dengan melakukan pengurangan nilai data aktual dengan data peramalan dan hasilnya dikuadratkan (squared) kemudian dijumlahkan secara keseluruhan dan membaginya dengan banyaknya data yang ada. 

![MSE](https://i.ibb.co/NtYhxLr/rumus-MSE.jpg)
Dimana :
At = nilai Aktual permintaan
Ft = nilai hasil peramalan
n = banyaknya data

Berikut ini hasil dari pelatihan model antara LSTM dan GRU menggunakan metriks evaluasi MSE:
| LSTM      | GRU |
| ----------- | ----------- |
| loss: 8.3698e-04     | loss: 0.0018     |

Hasil proyek menggunakan model LSTM mendapatkan loss sebesar 8.3698e-04, sedangkan jika menggunakan model GRU mendapatkan loss sebesar 0.0018. Karena loss dari LSTM lebih kecil dibandingkan dengan GRU maka bisa disimpulkan jika model LSTM memiliki akurasi yang lebih tinggi untuk prediksi harga saham IBM.

