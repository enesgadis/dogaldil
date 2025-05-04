import pandas as pd
from gensim.models import Word2Vec
import os


lemmatized_file = "output/lemmatized_sentences.csv"
stemmed_file = "output/stemmed_sentences.csv"
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)


def load_sentences(filepath):
    df = pd.read_csv(filepath)
    return df.iloc[:, 0].dropna().apply(lambda x: x.split()).tolist()

lemmatized_data = load_sentences(lemmatized_file)
stemmed_data = load_sentences(stemmed_file)


params = [
    {"sg": 0, "window": 2, "vector_size": 100},
    {"sg": 1, "window": 2, "vector_size": 100},
    {"sg": 0, "window": 4, "vector_size": 100},
    {"sg": 1, "window": 4, "vector_size": 100},
    {"sg": 0, "window": 2, "vector_size": 300},
    {"sg": 1, "window": 2, "vector_size": 300},
    {"sg": 0, "window": 4, "vector_size": 300},
    {"sg": 1, "window": 4, "vector_size": 300},
]


def train_models(data, data_name):
    for p in params:
        model = Word2Vec(
            sentences=data,
            vector_size=p["vector_size"],
            window=p["window"],
            sg=p["sg"],
            min_count=2,
            epochs=10
        )
        mtype = "cbow" if p["sg"] == 0 else "skipgram"
        fname = f"{data_name}_model_{mtype}_window{p['window']}_dim{p['vector_size']}.model"
        model.save(os.path.join(output_dir, fname))
        print(f" Saved: {fname}")


print(" Lemmatized modeller eğitiliyor...")
train_models(lemmatized_data, "lemmatized")

print(" Stemmed modeller eğitiliyor...")
train_models(stemmed_data, "stemmed")

print(" Tüm modeller eğitildi ve kaydedildi.")
