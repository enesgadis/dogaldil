import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
import ast

def plot_zipf(file_path, title, save_path):
    df = pd.read_csv(file_path)


    last_column = df.iloc[:, -1]


    if isinstance(last_column.iloc[0], str) and last_column.iloc[0].startswith('['):
        words = []
        for row in last_column:
            try:
                words.extend(ast.literal_eval(row))
            except Exception as e:
                continue
    else:
        text = " ".join(last_column.astype(str))
        words = text.split()

    freq = Counter(words)
    freq_sorted = sorted(freq.values(), reverse=True)

    ranks = range(1, len(freq_sorted) + 1)
    freqs = freq_sorted

  
    plt.figure(figsize=(8, 6))
    plt.loglog(ranks, freqs)
    plt.title(f"Zipf Plot - {title}")
    plt.xlabel("Log(Rank)")
    plt.ylabel("Log(Frequency)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    os.makedirs("output/plots", exist_ok=True)

    plot_zipf("../data/abcnews-date-text.csv", "Original Headlines", "../output/plots/zipf_original.png")
    plot_zipf("../output/lemmatized_sentences.csv", "Lemmatized", "../output/plots/zipf_lemmatized.png")
    plot_zipf("../output/stemmed_sentences.csv", "Stemmed", "../output/plots/zipf_stemmed.png")


    print(" Zipf grafikleri başarıyla kaydedildi.")
