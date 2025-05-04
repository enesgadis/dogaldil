import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

print(" Başlık benzerlik analizi başlatılıyor...")


df = pd.read_csv("output/lemmatized_sentences.csv").head(500)


tfidf = pd.read_csv("output/tfidf_lemmatized.csv").head(500)


similarity_matrix = cosine_similarity(tfidf)


threshold = 0.85
duplicates = []

for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        score = similarity_matrix[i][j]
        if score > threshold:
            duplicates.append({
                "index_1": i,
                "index_2": j,
                "score": round(score, 4),
                "title_1": df.iloc[i, 0],
                "title_2": df.iloc[j, 0]
            })

out_df = pd.DataFrame(duplicates)
os.makedirs("output", exist_ok=True)
out_df.to_csv("output/duplicate_titles.csv", index=False)

print(f" {len(duplicates)} eşleşme bulundu. Sonuçlar 'output/duplicate_titles.csv' dosyasına yazıldı.")
