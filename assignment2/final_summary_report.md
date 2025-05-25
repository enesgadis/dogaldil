# ÖDEV 2 - METİN BENZERLİĞİ HESAPLAMA VE DEĞERLENDİRME
## Final Özet Raporu

**Öğrenci**: Enes Gadiş (2107231053)  
**Ders**: Doğal Dil İşleme  
**Tarih**: Mayıs 2025  
**GitHub**: https://github.com/enesgadis/dogaldil/tree/master

---

## 1. GİRİŞ

Bu ödevde, Assignment 1'de ön işleme tabi tuttuğumuz ABC News Headlines veri seti üzerinde eğittiğimiz Word2Vec ve TF-IDF modellerini kullanarak metinler arası benzerlik hesaplamaları gerçekleştirdik. Toplam 18 model (16 Word2Vec + 2 TF-IDF) ile kapsamlı bir benzerlik analizi ve değerlendirme sistemi oluşturduk.

### Kullanılan Veri Seti
- **Kaynak**: ABC News Headlines (Kaggle)
- **Boyut**: 1 milyon haber başlığı (2003-2021)
- **Format**: CSV, 62 MB
- **İşlenmiş Veri**: 10,000 başlık (lemmatized ve stemmed)

### Örnek Giriş Metni
- **Lemmatized**: "australia contribute million aid iraq"
- **Stemmed**: "australia contribut million aid iraq"

---

## 2. YÖNTEM

### Benzerlik Hesaplama Teknikleri

#### A. TF-IDF Benzerliği
- **Vektörleştirme**: Sklearn TfidfVectorizer (max_features=5000)
- **Benzerlik Ölçütü**: Cosine Similarity
- **Veri Türleri**: Lemmatized ve Stemmed
- **Çıktı**: Her model için top 5 benzer metin

#### B. Word2Vec Benzerliği
- **Cümle Vektörü**: Kelime vektörlerinin aritmetik ortalaması
- **Benzerlik Ölçütü**: Cosine Similarity
- **Model Parametreleri**:
  - **Algoritma**: CBOW, SkipGram
  - **Pencere Boyutu**: 2, 4
  - **Vektör Boyutu**: 100, 300
  - **Veri Türü**: Lemmatized, Stemmed

### Değerlendirme Yöntemleri

#### 1. Anlamsal Değerlendirme (Subjective Evaluation)
- **Puanlama**: 1-5 arası (5=en yüksek benzerlik)
- **Kriterler**:
  - 5: Neredeyse aynı tema, çok güçlü benzerlik
  - 4: Anlamlı, açık benzerlik
  - 3: Ortalama düzeyde benzer
  - 2: Kısmen ilgili ama bağlam tutmuyor
  - 1: Çok alakasız, zayıf benzerlik

#### 2. Sıralama Tutarlılığı (Ranking Agreement)
- **Ölçüt**: Jaccard Benzerlik Katsayısı
- **Formül**: |A ∩ B| / |A ∪ B|
- **Matris**: 18x18 model karşılaştırması
- **Amaç**: Modellerin sıralama tutarlılığını ölçmek

---

## 3. SONUÇLAR VE DEĞERLENDİRME

### A. Model Performansları (Anlamsal Değerlendirme)

#### 🏆 En İyi 5 Model
| Sıra | Model | Ortalama Puan |
|------|-------|---------------|
| 1 | lemmatized_model_cbow_window2_dim100 | 4.80 |
| 2 | lemmatized_model_cbow_window2_dim300 | 4.80 |
| 3 | lemmatized_model_cbow_window4_dim100 | 4.80 |
| 4 | lemmatized_model_cbow_window4_dim300 | 4.80 |
| 5 | lemmatized_model_skipgram_window2_dim100 | 4.80 |

#### Model Türü Karşılaştırması
- **Word2Vec Modelleri**: 4.40-4.80 puan
- **TF-IDF Modelleri**: 3.20 puan

### B. TF-IDF Sonuçları

#### TF-IDF Lemmatized Top 5
1. [0.3874] oil price contribute massive caltex turnaround
2. [0.2924] aid group ship water iraq
3. [0.2779] aid reach iraq thursday
4. [0.2743] un brings food aid iraq
5. [0.2577] recycling save million dollar

#### TF-IDF Stemmed Top 5
1. [0.3434] oil price contribut massiv caltex turnaround
2. [0.3388] act women honour equal contribut
3. [0.3247] landhold contribut still discuss
4. [0.3168] polic tactic contribut protest violenc
5. [0.3001] wind may contribut ultralight crash

### C. Word2Vec Sonuçları (Örnek)

#### En İyi Model: lemmatized_model_cbow_window2_dim300
1. [1.0000] australia order embassy spy home iraq
2. [1.0000] australia likely help iraq rebuild rural
3. [1.0000] u farmer eye australia iraq wheat contract
4. [1.0000] mine delay aid ship iraq port
5. [1.0000] un aid convoy expected deliver food iraq

### D. Sıralama Tutarlılığı (Jaccard Analizi)

#### En Yüksek Tutarlılık Gösteren Model Çiftleri
| Sıra | Model 1 | Model 2 | Jaccard Skoru |
|------|---------|---------|---------------|
| 1 | lemmatized_cbow_window2_dim100 | lemmatized_cbow_window4_dim100 | 1.000 |
| 2 | lemmatized_cbow_window4_dim300 | stemmed_cbow_window4_dim300 | 1.000 |
| 3 | lemmatized_skipgram_window4_dim100 | lemmatized_skipgram_window4_dim300 | 1.000 |
| 4 | stemmed_cbow_window2_dim100 | stemmed_cbow_window4_dim100 | 0.667 |
| 5 | stemmed_skipgram_window2_dim300 | stemmed_skipgram_window4_dim300 | 0.667 |

---

## 4. BULGULAR VE YORUMLAR

### Word2Vec vs TF-IDF Karşılaştırması

#### Word2Vec Avantajları
- **Semantik Benzerlik**: Iraq konulu haberler için çok yüksek performans
- **Bağlamsal Anlayış**: Kelimeler arası ilişkileri daha iyi yakalar
- **Yüksek Benzerlik Skorları**: 0.99+ cosine similarity değerleri

#### TF-IDF Avantajları
- **Kelime Bazlı Benzerlik**: "contribute" kelimesi üzerinden etkili eşleşme
- **Çeşitlilik**: Daha geniş konu yelpazesinde sonuçlar
- **Hesaplama Verimliliği**: Daha hızlı benzerlik hesaplama

### Model Yapılandırması Etkileri

#### CBOW vs SkipGram
- **CBOW**: Daha tutarlı sonuçlar, yüksek Jaccard skorları
- **SkipGram**: Nadir kelimeler için daha iyi performans
- **Sonuç**: CBOW modelleri bu veri seti için daha başarılı

#### Pencere Boyutu (Window Size)
- **Window=2**: Daha yerel bağlam, tutarlı sonuçlar
- **Window=4**: Daha geniş bağlam, çeşitli sonuçlar
- **Sonuç**: Her iki boyut da etkili, window=2 biraz daha tutarlı

#### Vektör Boyutu (Vector Dimension)
- **Dim=100**: Hızlı eğitim, iyi performans
- **Dim=300**: Daha zengin temsil, marginally daha iyi sonuçlar
- **Sonuç**: Dim=300 hafif avantajlı ama dim=100 da yeterli

#### Lemmatization vs Stemming
- **Lemmatization**: Daha anlamlı sonuçlar, yüksek puanlar
- **Stemming**: Daha agresif normalizasyon, biraz düşük performans
- **Sonuç**: Lemmatization bu görev için daha uygun

### Sıralama Tutarlılığı Bulguları

#### Yüksek Tutarlılık
- **Aynı algoritma, farklı pencere**: CBOW modelleri %100 tutarlılık
- **Aynı veri türü**: Lemmatized modeller daha tutarlı

#### Düşük Tutarlılık
- **TF-IDF vs Word2Vec**: Neredeyse hiç örtüşme (0.0-0.11)
- **Farklı algoritmalar**: CBOW vs SkipGram arası orta tutarlılık

---

## 5. SONUÇ VE ÖNERİLER

### Genel Çıkarımlar

1. **En Başarılı Yaklaşım**: Word2Vec modelleri, özellikle lemmatized CBOW
2. **Optimal Konfigürasyon**: CBOW + Window=2/4 + Dim=100/300 + Lemmatized
3. **TF-IDF Rolü**: Kelime bazlı benzerlik için hala değerli
4. **Tutarlılık**: Aynı algoritma ailesi içinde yüksek tutarlılık

### Hangi Model, Hangi Görev İçin?

#### Iraq Konulu Haberler İçin
- **Önerilen**: lemmatized_model_cbow_window2_dim300
- **Neden**: En yüksek anlamsal benzerlik skorları

#### Genel Haber Benzerliği İçin
- **Önerilen**: TF-IDF + Word2Vec kombinasyonu
- **Neden**: TF-IDF çeşitlilik, Word2Vec semantik derinlik sağlar

#### Hızlı Prototipleme İçin
- **Önerilen**: lemmatized_model_cbow_window2_dim100
- **Neden**: Hızlı, etkili, düşük kaynak kullanımı

### Gelecek Çalışmalar

1. **Daha Büyük Vektör Boyutları**: 500, 1000 boyutlu vektörler
2. **Hibrit Yaklaşımlar**: TF-IDF + Word2Vec kombinasyonları
3. **Transformer Modelleri**: BERT, GPT benzeri modern yaklaşımlar
4. **Farklı Veri Setleri**: Çok dilli, farklı domain'ler
5. **Ensemble Yöntemleri**: Birden fazla modelin kombinasyonu

### Proje Değerlendirmesi

Bu çalışma, doğal dil işleme alanında temel tekniklerin pratik uygulamasını başarıyla göstermiştir. 18 farklı modelin sistematik karşılaştırması, her yaklaşımın güçlü ve zayıf yönlerini ortaya koymuştur. Özellikle Word2Vec modellerinin semantik benzerlik görevlerindeki üstünlüğü ve model parametrelerinin performans üzerindeki etkisi net şekilde gözlemlenmiştir.

---

## 6. TEKNIK DETAYLAR

### Kullanılan Kütüphaneler
- **pandas**: Veri manipülasyonu
- **numpy**: Numerik hesaplamalar
- **scikit-learn**: TF-IDF, cosine similarity
- **gensim**: Word2Vec modelleri
- **matplotlib, seaborn**: Görselleştirme

### Dosya Yapısı
```
assignment2/
├── similarity_calculator.py          # Ana benzerlik hesaplama
├── evaluation_system.py              # Kapsamlı değerlendirme
├── jaccard_similarity_matrix.csv     # Tutarlılık matrisi
├── jaccard_similarity_heatmap.png    # Görsel matris
├── comprehensive_evaluation_report.txt # Detaylı rapor
└── final_summary_report.md           # Bu özet rapor
```

### Performans Metrikleri
- **Hesaplama Süresi**: ~5 dakika (18 model)
- **Bellek Kullanımı**: ~2 GB (TF-IDF matrisleri)
- **Dosya Boyutları**: TF-IDF CSV'ler ~192 MB each

---

**Son Güncelleme**: Mayıs 2025  
**Toplam Kod Satırı**: ~1000 satır  
**Toplam Analiz Edilen Metin**: 10,000 haber başlığı  
**Toplam Model**: 18 adet (16 Word2Vec + 2 TF-IDF) 