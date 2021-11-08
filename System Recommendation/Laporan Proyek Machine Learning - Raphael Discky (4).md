# Laporan Proyek Machine Learning - Raphael

## Project Overview

Sistem rekomendasi merupakan sebuah metode yang digunakan untuk memberikan rekomendasi pada sebuah produk seperti buku, musik dan film dengan memberikan nilai prediksi tertinggi pada penggunanya. Sistem rekomendasi saat ini masih bisa ditingkatkan lagi kualitasnya, sehingga dengan menggunakan metode baru diharapkan dapat lebih meningkatkan lagi nilai relevansi dari hasil rekomendasi yang diberikan daripada sistem-sistem sebelumnya. Selain itu perkembangan film didunia juga setiap harinya  semakin meningkat dengan berbagai jenis genre yang dimiliki, sehingga membuat para penonton film merasa kesulitan untuk memilih film apa yang akan ditonton. Maka dari itu, diperlukan adanya sebuah sistem yang dapat memberikan rekomendasi dari penonton film lainnya. Pada proyek ini membuat sistem rekomendasi film menggunakan collaborative filtering dengan  algoritma K-Nearest Neighbors. 

Sistem rekomendasi menjadi semakin penting di dunia yang sibuk saat ini. Orang selalu mencari produk/layanan yang paling cocok untuk mereka. Oleh karena itu, sistem rekomendasi penting karena membantu mereka membuat pilihan yang tepat, tanpa harus mengeluarkan sumber daya lain yang besar. 
  
Referensi: [An improved collaborative movie recommendation system using computational intelligence](https://www.sciencedirect.com/science/article/abs/pii/S1045926X14000901) 

## Business Understanding
### Problem Statements

Rumusan masalah mengenai proyek ini adalah sebagai berikut:
- Bagaimana membuat sistem rekomendasi film menggunakan metode Collaborative Filtering dengan K-NN?
- Bagaimana implementasi untuk menghasilkan rekomendasi film dengan Collaborative Filtering dengan kemiripan genre?

### Goals

Tujuan proyek ini adalah sebagai berikut:
- Membuat sistem rekomendasi film menggunakan metode Collaborative Filtering dengan K-NN
- menghasilkan rekomendasi film dengan Collaborative Filtering dengan kemiripan genre

    ### Solution statements
    - Solusi pertama yang diberikan adalah sistem dibuat dengan menggunakan algoritma K-Nearest Neighbors
    - Solusi kedua adalah sistem bisa memprediksi rating film berdasarkan tetangganya dan membandingkannya dengan rating sebenarnya

## Data Understanding
Dataset yang digunakan adalah [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata). TMDB adalah database film dan TV yang dibangun komunitas yang memiliki data ekstensif tentang film dan Acara TV. Dataset ini terdiri dari dua file yaitu movies.csv dan credits.csv.

Variabel-variabel pada TMDB 5000 Movie Dataset adalah sebagai berikut:
##### Variabel untuk movies.csv
- genres : merupakan jenis film tertentu
- homepage: merupakan website film tertentu
- keywords: merupakan kata-kata yang berhubungan dengan film tertentu
- original_language: merupakan bahasa yang digunakan oleh film tertentu
- original_title: merupakan judul film tertentu
- overview: merupakan deskripsi singkat tentang film tertentu
- popularity: berapa tingkat kepopuleran film tertentu
- production_countries: dimana tempat produksi film tertentu
- release_date: tanggal rilis film tertentu
- revenue: penghasilan yang didapatkan dari film tertentu
- dll

##### Variabel untuk credits.csv
- title: merupakan judul dari film tertentu
- cast: merupakan pemeran dari film tertentu
- crew: merupakan pekerja dari film tertentu

Melakukan visualisasi data untuk genre film yang paling banyak pada dataset.
![Top Genres](https://i.ibb.co/Jyr8PjB/Screenshot-2021-11-07-171512.png)
Terlihat bahwa top genre pada dataset TMDB adalah drama sebanyak 2297 film.

## Data Preparation
Memeriksa dataset, kita dapat melihat bahwa genres, keywords, production_companies, production_countries, spoken_languages ditulis dalam format JSON. Demikian pula di file CSV lainnya, cast dan crew dalam format JSON. Kita akan mengubah kolom-kolom ini menjadi format yang dapat dengan mudah dibaca dan diinterpretasikan. Kita mengubahnya menjadi string dan kemudian mengubahnya menjadi list supaya data dapat diinterpretasikan dengan lebih mudah.

Terlihat jika kolom genres masih dalam bentuk JSON seperti pada gambar dibawah:
![JSON](https://i.ibb.co/VMQryHG/Screenshot-2021-11-07-154854.png)

##### Mengonversi JSON ke string
Untuk mengonversi JSON ke string kita bisa menulis kode seperti berikut:
```
movies['genres'] = movies['genres'].apply(json.loads)
for index,i in zip(movies.index,movies['genres']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name'])) 
    movies.loc[index,'genres'] = str(list1)

movies['keywords'] = movies['keywords'].apply(json.loads)
for index,i in zip(movies.index,movies['keywords']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies.loc[index,'keywords'] = str(list1)

movies['production_companies'] = movies['production_companies'].apply(json.loads)
for index,i in zip(movies.index,movies['production_companies']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies.loc[index,'production_companies'] = str(list1)

credits['cast'] = credits['cast'].apply(json.loads)
for index,i in zip(credits.index,credits['cast']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    credits.loc[index,'cast'] = str(list1)
    
credits['crew'] = credits['crew'].apply(json.loads)
def director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
credits['crew'] = credits['crew'].apply(director)
credits.rename(columns={'crew':'director'},inplace=True)
```


Hasil konversi ke string untuk movies.csv
![Movies Dataframe](https://i.ibb.co/Pt5k6sX/Screenshot-2021-11-07-134042.png)

Hasil konversi ke string untuk credits.csv
![Credits Dataframe](https://i.ibb.co/fp6wBnL/Screenshot-2021-11-07-134033.png)



##### One Hot Encoding untuk Beberapa Kolom
Kemudian kita akan melakukan One Hot Encoding untuk beberapa kolom.

One Hot Encoding untuk kolom Genres:
```
def binary(genre_list):
    binaryList = []
    
    for genre in genreList:
        if genre in genre_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList

movies['genres_bin'] = movies['genres'].apply(lambda x: binary(x))
movies['genres_bin'].head()
```

Hasil One Hot Encoding untuk kolom Genres:
```
0    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
1    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
2    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
3    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...
4    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
Name: genres_bin, dtype: object
```

One Hot Encoding untuk kolom Cast:
```
castList = []
for index, row in movies.iterrows():
    cast = row["cast"]
    
    for i in cast:
        if i not in castList:
            castList.append(i)

def binary(cast_list):
    binaryList = []
    
    for genre in castList:
        if genre in cast_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList

movies['cast_bin'] = movies['cast'].apply(lambda x: binary(x))
movies['cast_bin'].head()
```

Hasil One Hot Encoding untuk kolom Cast:
```
0    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
1    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, ...
2    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
3    [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, ...
4    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, ...
Name: cast_bin, dtype: object
```

One Hot Encoding untuk kolom Director:
```
def xstr(s):
    if s is None:
        return ''
    return str(s)
movies['director'] = movies['director'].apply(xstr)

directorList=[]
for i in movies['director']:
    if i not in directorList:
        directorList.append(i)

def binary(director_list):
    binaryList = []  
    for direct in directorList:
        if direct in director_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    return binaryList

movies['director_bin'] = movies['director'].apply(lambda x: binary(x))
movies.head()
```

Hasil One Hot Encoding untuk kolom Director:
![Director](https://i.ibb.co/Xyq5P2V/Screenshot-2021-11-07-134514.png)

## Modeling
Kemudian kita menggunakan Cosine Similarity untuk metric distance pada algoritma K-NN. Fungsi untuk merekomendasikan film dan memprediksi rating adalah sebagai berikut: 

```
def recommend_movies(name):
    new_movie = movies[movies['original_title'].str.contains(name)].iloc[0].to_frame().T
    print('Selected Movie: ',new_movie.original_title.values[0])
    def getNeighbors(baseMovie, K):
        distances = []
    
        for index, movie in movies.iterrows():
            if movie['new_id'] != baseMovie['new_id'].values[0]:
                dist = Similarity(baseMovie['new_id'].values[0], movie['new_id'])
                distances.append((movie['new_id'], dist))
    
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
    
        for x in range(K):
            neighbors.append(distances[x])
        return neighbors

    K = 10
    avgRating = 0
    neighbors = getNeighbors(new_movie, K)
    
    print('\nRecommended Movies: \n')
    for neighbor in neighbors:
        avgRating = avgRating+movies.iloc[neighbor[0]][2]  
        print( movies.iloc[neighbor[0]][0]+" | Genres: "+str(movies.iloc[neighbor[0]][1]).strip('[]').replace(' ','')+" | Rating: "+str(movies.iloc[neighbor[0]][2]))
    
    print('\n')
    avgRating = avgRating/K
    print('The predicted rating for %s is: %f' %(new_movie['original_title'].values[0],avgRating))
    print('The actual rating for %s is %f' %(new_movie['original_title'].values[0],new_movie['vote_average']))
```

Kemudian kita panggil fungsi recommend_movies() sebagai berikut:
![Rec Movies](https://i.ibb.co/my8wN1J/Screenshot-2021-11-07-140006.png)

Bisa dilihat bahwa film 'How to Train Your Dragon' memiliki kesamaan dengan film 'The Croods', 'Lilo & Stitch', 'Epic', 'Legend of the Guardians', dan lain-lain. Selain itu juga bisa dilihat jika prediksi rating 'How to Train Your Dragon' adalah 6.54 sedangkan rating aslinya adalah 7.5.

## Evaluation
Kita akan menggunakan metric distance dengan cosine similarity pada algoritma K-NN. Pada tahap ini kita mencari kesamaan antara dua film dengan menggunakan Cosine Similarity. Cara kerja dari Cosine Similarity adalah sebagai berikut:

Katakanlah kita memiliki 2 vektor. Jika vektor-vektor tersebut berdekatan sejajar, yaitu sudut antara vektor-vektor tersebut adalah 0, maka kita dapat mengatakan bahwa keduanya “serupa”, dengan nilai cos(0)=1. Sedangkan jika vektor-vektornya ortogonal, maka dapat dikatakan bebas atau TIDAK “serupa”, karena cos(90)=0.

Kita membuat fungsi sebagai berikut:
```
from scipy import spatial

def Similarity(movieId1, movieId2):
    a = movies.iloc[movieId1]
    b = movies.iloc[movieId2]
    
    genresA = a['genres_bin']
    genresB = b['genres_bin']
    
    genreDistance = spatial.distance.cosine(genresA, genresB)
    
    scoreA = a['cast_bin']
    scoreB = b['cast_bin']
    scoreDistance = spatial.distance.cosine(scoreA, scoreB)
    
    directA = a['director_bin']
    directB = b['director_bin']
    directDistance = spatial.distance.cosine(directA, directB)
    
    wordsA = a['words_bin']
    wordsB = b['words_bin']
    wordsDistance = spatial.distance.cosine(directA, directB)
    return genreDistance + directDistance + scoreDistance + wordsDistance
```
Kita bisa memanggil fungsi Similarity() untuk mencari distance antara dua film seperti berikut:
![Similarity](https://i.ibb.co/8X2yT37/Screenshot-2021-11-07-154416.png)

Dari fungsi tersebut kita bisa mencari kemiripan dari dua film dengan memanggil fungsi Similarity(). Similarity(2,4) akan menghasilkan 1.6381516362402322 sedangkan Similarity(9,12) 1.3472087901661332. Karena hasil distance lebih kecil film 9 dengan 12 memiliki kemiripan yang lebih dibandingkan antara film 2 dan 4.