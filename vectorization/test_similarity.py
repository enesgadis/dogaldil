from gensim.models import Word2Vec
import os

def test_word_similarity(model_dir="models", test_word="economy"):
    if not os.path.exists(model_dir):
        print(f" Model klasörü bulunamadı: {model_dir}")
        return

    models = [f for f in os.listdir(model_dir) if f.endswith(".model")]
    if not models:
        print(" Hiçbir model bulunamadı.")
        return

    for model_name in sorted(models):
        model_path = os.path.join(model_dir, model_name)
        print(f"\n Model: {model_name}")
        try:
            model = Word2Vec.load(model_path)
            if test_word in model.wv:
                print(f" '{test_word}' için en benzer 5 kelime:")
                for word, score in model.wv.most_similar(test_word, topn=5):
                    print(f"   {word:<15} -> {score:.4f}")
            else:
                print(f" '{test_word}' kelimesi modelde bulunamadı.")
        except Exception as e:
            print(f" Hata oluştu: {e}")

if __name__ == "__main__":
    test_word_similarity()
