from src.data import load_pubmedqa, preprocess, get_splits

if __name__ == "__main__":
    dataset = load_pubmedqa()
    texts, labels, encoder = preprocess(dataset["train"])
    X_train, X_test, y_train, y_test = get_splits(texts, labels)
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Label classes: {encoder.classes_}")
    print(f"Example input:\n{X_train[0][:300]}...")