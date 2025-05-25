import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os


os.makedirs("output", exist_ok=True)


df_lem = pd.read_csv("output/lemmatized_sentences.csv")
df_stem = pd.read_csv("output/stemmed_sentences.csv")


vectorizer = TfidfVectorizer()


tfidf_lem = vectorizer.fit_transform(df_lem.iloc[:, 0])
pd.DataFrame(tfidf_lem.toarray(), columns=vectorizer.get_feature_names_out()).to_csv("output/tfidf_lemmatized.csv", index=False)


tfidf_stem = vectorizer.fit_transform(df_stem.iloc[:, 0])
pd.DataFrame(tfidf_stem.toarray(), columns=vectorizer.get_feature_names_out()).to_csv("output/tfidf_stemmed.csv", index=False)

print("TF-IDF dosyaları oluşturuldu.")
