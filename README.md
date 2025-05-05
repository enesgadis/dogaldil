
##  Projede Uygulanan Adımlar

### 1. Veri Kümesi Tanıtımı
- Kullanılan veri kümesi: `abcnews-date-text.csv` (https://www.kaggle.com/datasets/therohk/million-headlines)
- Toplam başlık: 1.1 milyon
- Amaç: Aynı anlamı taşıyan veya birbirine çok benzeyen başlıkları tespit etmek

### 2. Zipf Analizi
- Zipf yasası doğrultusunda başlıklarda geçen kelimelerin frekans dağılımları incelendi.
- [Zipf analizini görüntüle](https://github.com/enesgadis/dogaldil/blob/master/analysis/zipf_analysis.py)

### 3. Ön İşleme Adımları
- Noktalama ve stopword temizliği
- Lemmatization ve stemming uygulanarak iki ayrı veri seti oluşturuldu.
  - [Lemmatized CSV](https://github.com/enesgadis/dogaldil/blob/master/output/lemmatized_sentences.csv)  
  - [Stemmed CSV](https://github.com/enesgadis/dogaldil/blob/master/output/stemmed_sentences.csv)

### 4. Vektörleştirme: TF-IDF ve Word2Vec
- Her iki ön işlenmiş metin üzerinde TF-IDF uygulanarak büyük boyutlu vektör matrisleri oluşturuldu.
- 16 farklı parametre kombinasyonu ile Word2Vec modelleri eğitildi (window, min_count, vector_size).
- [Vektörleştirme kodlarına göz at](https://github.com/enesgadis/dogaldil/tree/master/vectorization)

### 5. Benzerlik Sınıflandırması
- Rastgele seçilen başlıklarla diğer başlıklar arasındaki cosine similarity hesaplandı.
- En benzer cümleler ekrana yazdırıldı.

##  Kullanılan Teknolojiler
- Python 3.11
- scikit-learn
- Gensim
- Matplotlib
- Pandas / Numpy
- Git LFS

##  Kurulum ve Kullanım

### Gerekli Kütüphaneler

Aşağıdaki komut ile tüm bağımlılıkları yükleyebilirsiniz:


pip install -r requirements.txt


### Adım Adım Uygulama

1. **Ön İşleme**  
   - Noktalama işaretleri ve stopword'ler temizlenir.
   - Lemmatizasyon ve stemming işlemleri uygulanır.
   
   python preprocessing.py
   

2. **Vektörleştirme (TF-IDF & Word2Vec)**  
   - Temizlenmiş verilerden TF-IDF matrisleri oluşturulur.
   - Word2Vec modelleri 16 farklı parametre kombinasyonu ile eğitilir.
   
   python vectorization.py
   

3. **Benzerlik Hesaplama**  
   - Cosine similarity ile en benzer başlıklar sıralanır.
   
   python similarity.py
   

## 🔬 Uygulanan Analizler ve Kodlar

| Adım | Açıklama | Bağlantı |
|------|----------|----------|
|  Zipf Analizi | Kelime frekans dağılımı | [zipf_analysis.py](analysis/zipf_analysis.py) |
|  Temizleme | Stopword, noktalama temizliği | [preprocessing.py](preprocessing/preprocess.py) |
|  Vektörleştirme | TF-IDF, Word2Vec uygulamaları | [vectorization](vectorization) |
|  Benzerlik Tespiti | Cosine similarity | [similarity.py](similarity.py) |

##  Çıktı Dosyaları

| Dosya | Açıklama | Durum |
|-------|----------|--------|
| `output/lemmatized_sentences.csv` | Lemmatize edilmiş cümleler | ✔ |
| `output/stemmed_sentences.csv` | Stemmed cümleler | ✔ |
| `output/tfidf_lemmatized.csv` | TF-IDF matrisi (lemmatize) |  GitHub limiti nedeniyle yüklenemedi |
| `output/tfidf_stemmed.csv` | TF-IDF matrisi (stemmed) |  GitHub limiti nedeniyle yüklenemedi |

> Not: 100 MB üzerindeki dosyalar **Git LFS** ile yüklenmek istenmiş ancak GitHub sınırları nedeniyle başarısız olmuştur.
##  Büyük Dosyalar
GitHub'ın sınırları nedeniyle bazı dosyalar LFS ile yüklenmeyi denenmiştir şuanlık başarılı olamamıştır.

- [TF-IDF (lemmatized)](https://github.com/enesgadis/dogaldil/blob/master/output/tfidf_lemmatized.csv)
- [TF-IDF (stemmed)](https://github.com/enesgadis/dogaldil/blob/master/output/tfidf_stemmed.csv)

##  Yazar
- **Enes Gadiş**  
  – [GitHub Profilim](https://github.com/enesgadis)

---

##  Rapor
Tüm süreç PDF rapor olarak da sunulmuştur:  
 [PDF Raporu Görüntüle](https://github.com/enesgadis/dogaldil/blob/master/2107231053%20Enes%20Gadiş.pdf)

