import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import os
import pickle
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SimilarityCalculator:
    def __init__(self):
        self.models = {}
        self.tfidf_data = {}
        self.sentences_data = {}
        self.results = {}
        
    def load_data(self):
        """Load all necessary data files"""
        print("📂 Veri dosyaları yükleniyor...")
        
        # TF-IDF dosyalarını yükle
        try:
            self.tfidf_data['lemmatized'] = pd.read_csv("output/tfidf_lemmatized.csv")
            self.tfidf_data['stemmed'] = pd.read_csv("output/tfidf_stemmed.csv")
            print(f"✅ TF-IDF dosyaları yüklendi")
        except Exception as e:
            print(f"❌ TF-IDF yükleme hatası: {e}")
            return False
            
        # Cümle dosyalarını yükle
        try:
            self.sentences_data['lemmatized'] = pd.read_csv("output/lemmatized_sentences.csv", header=None)
            self.sentences_data['stemmed'] = pd.read_csv("output/stemmed_sentences.csv", header=None)
            print(f"✅ Cümle dosyaları yüklendi")
        except Exception as e:
            print(f"❌ Cümle dosyası yükleme hatası: {e}")
            return False
            
        # Word2Vec modellerini yükle
        self.load_word2vec_models()
        
        return True
        
    def load_word2vec_models(self):
        """Load all Word2Vec models"""
        models_dir = "models"
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.model')]
        
        print(f"🔄 {len(model_files)} Word2Vec modeli yükleniyor...")
        
        for model_file in model_files:
            try:
                model_path = os.path.join(models_dir, model_file)
                model = Word2Vec.load(model_path)
                self.models[model_file.replace('.model', '')] = model
                print(f"✅ {model_file}")
            except Exception as e:
                print(f"❌ {model_file}: {e}")
                
        print(f"✅ Toplam {len(self.models)} model yüklendi")
    
    def get_sentence_vector_word2vec(self, sentence: str, model: Word2Vec) -> np.ndarray:
        """Calculate average vector representation for a sentence using Word2Vec"""
        words = sentence.split()
        vectors = []
        
        for word in words:
            if word in model.wv:
                vectors.append(model.wv[word])
                
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.wv.vector_size)
    
    def calculate_tfidf_similarity(self, query_text: str, data_type: str = 'lemmatized') -> List[Tuple[int, float]]:
        """Calculate TF-IDF similarity for query text"""
        print(f"🔄 TF-IDF {data_type} benzerlik hesaplanıyor...")
        
        # Query metninin vektörünü bul (veri setindeki index'i)
        sentences = self.sentences_data[data_type].iloc[:, 0].tolist()
        
        try:
            query_idx = sentences.index(query_text)
        except ValueError:
            print(f"❌ Query metni veri setinde bulunamadı: {query_text}")
            return []
            
        # TF-IDF verisini al (document_id sütununu atla)
        tfidf_matrix = self.tfidf_data[data_type].iloc[:, 1:].values
        query_vector = tfidf_matrix[query_idx].reshape(1, -1)
        
        # Tüm dokümanlarla benzerlik hesapla
        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
        
        # Sırala ve ilk 6'yı al (query kendisi de dahil olacak, onu çıkaracağız)
        similarity_scores = [(i, sim) for i, sim in enumerate(similarities)]
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Query'nin kendisini çıkar
        filtered_scores = [(idx, score) for idx, score in similarity_scores if idx != query_idx]
        
        return filtered_scores[:5]
    
    def calculate_word2vec_similarity(self, query_text: str, model_name: str) -> List[Tuple[int, float]]:
        """Calculate Word2Vec similarity for query text"""
        print(f"🔄 Word2Vec {model_name} benzerlik hesaplanıyor...")
        
        if model_name not in self.models:
            print(f"❌ Model bulunamadı: {model_name}")
            return []
            
        model = self.models[model_name]
        
        # Model tipine göre veri setini seç
        data_type = 'lemmatized' if 'lemmatized' in model_name else 'stemmed'
        sentences = self.sentences_data[data_type].iloc[:, 0].tolist()
        
        # Query metninin vektörünü hesapla
        query_vector = self.get_sentence_vector_word2vec(query_text, model)
        
        # Tüm cümlelerle benzerlik hesapla
        similarities = []
        query_idx = -1
        
        try:
            query_idx = sentences.index(query_text)
        except ValueError:
            print(f"❌ Query metni veri setinde bulunamadı: {query_text}")
            return []
            
        for i, sentence in enumerate(sentences):
            if i != query_idx:  # Query'nin kendisini atla
                sentence_vector = self.get_sentence_vector_word2vec(sentence, model)
                
                # Cosine similarity hesapla
                if np.linalg.norm(query_vector) > 0 and np.linalg.norm(sentence_vector) > 0:
                    similarity = np.dot(query_vector, sentence_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(sentence_vector)
                    )
                    similarities.append((i, similarity))
                else:
                    similarities.append((i, 0.0))
        
        # Sırala ve ilk 5'i al
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:5]
    
    def run_similarity_analysis(self, query_text: str):
        """Run complete similarity analysis for all models"""
        print(f"\n🎯 Query metni: '{query_text}'")
        print("=" * 80)
        
        self.results = {'query': query_text, 'models': {}}
        
        # TF-IDF hesaplamaları
        print("\n📊 TF-IDF Benzerlik Analizi")
        print("-" * 40)
        
        for data_type in ['lemmatized', 'stemmed']:
            model_name = f'tfidf_{data_type}'
            results = self.calculate_tfidf_similarity(query_text, data_type)
            self.results['models'][model_name] = {
                'type': 'tfidf',
                'data_type': data_type,
                'results': results
            }
            self.print_results(model_name, results, data_type)
        
        # Word2Vec hesaplamaları
        print("\n🧠 Word2Vec Benzerlik Analizi")
        print("-" * 40)
        
        for model_name in sorted(self.models.keys()):
            results = self.calculate_word2vec_similarity(query_text, model_name)
            data_type = 'lemmatized' if 'lemmatized' in model_name else 'stemmed'
            self.results['models'][model_name] = {
                'type': 'word2vec',
                'data_type': data_type,
                'results': results
            }
            self.print_results(model_name, results, data_type)
    
    def print_results(self, model_name: str, results: List[Tuple[int, float]], data_type: str):
        """Print similarity results in a formatted way"""
        sentences = self.sentences_data[data_type].iloc[:, 0].tolist()
        
        print(f"\n🔍 Model: {model_name}")
        print("Top 5 benzer metinler:")
        
        for rank, (idx, score) in enumerate(results, 1):
            sentence = sentences[idx] if idx < len(sentences) else "Hata"
            print(f"  {rank}. [{score:.4f}] {sentence}")
    
    def save_results(self, filename: str = "assignment2/similarity_results.pkl"):
        """Save results to pickle file"""
        os.makedirs("assignment2", exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"\n💾 Sonuçlar kaydedildi: {filename}")
        
    def generate_summary_report(self):
        """Generate a summary report of all results"""
        report_file = "assignment2/similarity_summary.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ÖDEV 2 - BENZERLİK ANALİZİ ÖZET RAPORU\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Query metni: {self.results['query']}\n\n")
            
            for model_name, data in self.results['models'].items():
                f.write(f"\n🔍 Model: {model_name}\n")
                f.write("-" * 40 + "\n")
                
                data_type = data['data_type']
                sentences = self.sentences_data[data_type].iloc[:, 0].tolist()
                
                for rank, (idx, score) in enumerate(data['results'], 1):
                    sentence = sentences[idx] if idx < len(sentences) else "Hata"
                    f.write(f"{rank}. [{score:.4f}] {sentence}\n")
        
        print(f"📄 Özet rapor oluşturuldu: {report_file}")

def main():
    # Örnek query metni (veri setinden seçilmiş)
    query_text = "australia contribute million aid iraq"
    
    calculator = SimilarityCalculator()
    
    if calculator.load_data():
        calculator.run_similarity_analysis(query_text)
        calculator.save_results()
        calculator.generate_summary_report()
    else:
        print("❌ Veri yükleme başarısız!")

if __name__ == "__main__":
    main() 