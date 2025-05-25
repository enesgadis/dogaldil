import os
os.makedirs("output", exist_ok=True)

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re



nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


df = pd.read_csv("data/abcnews-date-text.csv")
df = df.head(10000)


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return tokens


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


print("Preprocessing başlıyor...")
df["tokens"] = df["headline_text"].apply(preprocess)
df["stemmed"] = df["tokens"].apply(lambda tokens: [stemmer.stem(t) for t in tokens])
df["lemmatized"] = df["tokens"].apply(lambda tokens: [lemmatizer.lemmatize(t) for t in tokens])


os.makedirs("output", exist_ok=True)


print("CSV dosyaları yazılıyor...")
df["stemmed"].apply(lambda x: " ".join(x)).to_csv("output/stemmed_sentences.csv", index=False, header=False)
df["lemmatized"].apply(lambda x: " ".join(x)).to_csv("output/lemmatized_sentences.csv", index=False, header=False)

print("İşlem tamamlandı.")
