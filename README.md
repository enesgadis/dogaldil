# ABCNews NLP Project

This project performs preprocessing and vectorization (TF-IDF, Word2Vec) on the ABC News Headlines dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Preprocessing

```bash
python preprocessing/preprocess.py
```

## TF-IDF

```bash
python vectorization/tfidf_vectorizer.py
```

## Word2Vec

```bash
python vectorization/train_word2vec.py
```

## Folder Structure

- `data/`: Raw and cleaned datasets
- `output/`: Vector files (TF-IDF, Word2Vec outputs)
- `models/`: Trained Word2Vec models