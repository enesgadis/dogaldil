# ABCNews NLP Project 
Bu proje, ABC News Headlines dataset'i üzerinde doğal dil işleme teknikleri kullanılarak metin ön işleme, vektörleştirme ve benzerlik analizi gerçekleştirmektedir.

## 📊 Proje Özeti

- **Veri Seti**: ABC News Headlines (1M haber başlığı, 2003-2021)
- **Boyut**: 62 MB, CSV formatı
- **Modeller**: 16 Word2Vec + 2 TF-IDF modeli
- **Analiz**: Zipf yasası, benzerlik hesaplama, değerlendirme

## 🚀 Kurulum

```bash
# Gerekli kütüphaneleri kur
pip install -r requirements.txt

# Veri setini data/ klasörüne yerleştir
# abcnews-date-text.csv dosyası gerekli
```

## 📁 Proje Yapısı

```
├── data/                          # Ham veri
│   └── abcnews-date-text.csv
├── preprocessing/                 # Ön işleme kodları
│   └── preprocess.py
├── vectorization/                 # Vektörleştirme
│   ├── create_tfidf.py
│   ├── train_word2vec.py
│   └── test_similarity.py
├── output/                        # İşlenmiş veriler
│   ├── lemmatized_sentences.csv
│   ├── stemmed_sentences.csv
│   ├── tfidf_lemmatized.csv
│   └── tfidf_stemmed.csv
├── models/                        # Eğitilmiş modeller
│   ├── lemmatized_model_*.model   # 8 adet
│   └── stemmed_model_*.model      # 8 adet
└── analysis/                      # Analiz sonuçları
    └── plots/
```

## 🔧 Kullanım

### Ödev 1: Veri Ön İşleme ve Model Eğitimi


# 1. Veri ön işleme
python preprocessing/preprocess.py

# 2. TF-IDF vektörleştirme
python vectorization/create_tfidf.py

# 3. Word2Vec model eğitimi
python vectorization/train_word2vec.py

# 4. Zipf analizi
python zipf_analysis.py




## 📈 Ödev 2 - Benzerlik Analizi Sonuçları

### Query Metni
- **Lemmatized**: "australia contribute million aid iraq"
- **Stemmed**: "australia contribut million aid iraq"

### Model Performansları

#### 🏆 En İyi 5 Model (Anlamsal Değerlendirme)
1. **lemmatized_model_cbow_window2_dim100** (Ortalama: 4.80)
2. **lemmatized_model_cbow_window2_dim300** (Ortalama: 4.80)
3. **lemmatized_model_cbow_window4_dim100** (Ortalama: 4.80)
4. **lemmatized_model_cbow_window4_dim300** (Ortalama: 4.80)
5. **lemmatized_model_skipgram_window2_dim100** (Ortalama: 4.80)

#### 📊 Model Türü Karşılaştırması
- **Word2Vec Modelleri**: Ortalama 4.40-4.80 puan
- **TF-IDF Modelleri**: Ortalama 3.20 puan

### Sıralama Tutarlılığı (Jaccard Benzerlik)

En yüksek tutarlılık gösteren model çiftleri:
1. **CBOW modelleri** arasında %100 tutarlılık
2. **SkipGram modelleri** arasında %67-100 tutarlılık
3. **Lemmatized vs Stemmed** modeller arası orta düzey tutarlılık

## 📋 Değerlendirme Kriterleri

### Anlamsal Değerlendirme (1-5 puan)
- **5 puan**: Neredeyse aynı temada, çok güçlü benzerlik
- **4 puan**: Anlamlı, açık benzerlik içeriyor
- **3 puan**: Ortalama düzeyde benzer
- **2 puan**: Kısmen ilgili ama bağlamı tutmuyor
- **1 puan**: Çok alakasız, anlamca zayıf benzerlik

### Jaccard Benzerlik Matrisi
- Model çiftleri arasında sıralama tutarlılığını ölçer
- 0-1 arası değer (1 = tam tutarlılık)
- 18x18 matris (16 Word2Vec + 2 TF-IDF)

## 📊 Çıktı Dosyaları



### Model Dosyaları
- **TF-IDF**: `output/tfidf_lemmatized.csv`, `output/tfidf_stemmed.csv`
- **Word2Vec**: `models/` klasöründe 16 adet .model dosyası

## 🔍 Temel Bulgular

### Word2Vec vs TF-IDF
- **Word2Vec**: Semantik benzerlik için daha başarılı
- **TF-IDF**: Kelime bazlı benzerlik, daha genel sonuçlar

### CBOW vs SkipGram
- **CBOW**: Daha tutarlı sonuçlar, hızlı eğitim
- **SkipGram**: Nadir kelimeler için daha iyi performans

### Lemmatization vs Stemming
- **Lemmatization**: Daha anlamlı sonuçlar
- **Stemming**: Daha agresif normalizasyon

### Pencere Boyutu ve Vektör Boyutu
- **Window=2**: Daha yerel bağlam
- **Window=4**: Daha geniş bağlam
- **Dim=300**: Daha zengin temsil, daha iyi performans

## 🎯 Sonuç ve Öneriler

1. **En İyi Model**: Lemmatized CBOW modelleri (window=2/4, dim=100/300)
2. **Önerilen Kullanım**: Iraq konulu haberler için Word2Vec modelleri
3. **Gelecek Çalışmalar**: Daha büyük vektör boyutları, farklı algoritmalar

## 📚 Referanslar

- **Veri Seti**: [ABC News Headlines - Kaggle](https://www.kaggle.com/therohk/million-headlines)
- **Word2Vec**: Mikolov et al. (2013)
- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **Jaccard Similarity**: Set intersection/union ratio

## 👨‍💻 Geliştirici

**Enes Gadiş**  
Öğrenci No: 2107231053  
Gümüşhane Üniversitesi - Yazılım Mühendisliği
