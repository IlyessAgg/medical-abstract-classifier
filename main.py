from src.data import load_pubmedqa, preprocess, get_splits
from src.model import get_embeddings, train_classifier
from src.evaluate import evaluate_classifier
import numpy as np

if __name__ == "__main__":
    # Load data
    dataset = load_pubmedqa()
    texts, labels, encoder = preprocess(dataset["train"])
    X_train, X_test, y_train, y_test = get_splits(texts, labels)
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Label classes: {encoder.classes_}")

    # Embed data
    # X_train_emb = get_embeddings(X_train)
    # np.save('data/embeddings_train.npy', X_train_emb)
    # X_test_emb = get_embeddings(X_test)
    # np.save('data/embeddings_test.npy', X_test_emb)

    X_train_emb = np.load('data/embeddings_train.npy')
    X_test_emb = np.load('data/embeddings_test.npy')

    # Classifier
    logistic = train_classifier(X_train_emb, y_train)

    # Evaluation
    report = evaluate_classifier(logistic, X_test_emb, y_test, encoder)