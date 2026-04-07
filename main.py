from src.data import load_pubmedqa, preprocess, get_splits
from src.model import get_embeddings, train_classifier
from src.evaluate import evaluate_classifier
import numpy as np

import mlflow
import mlflow.sklearn

def run_experiment(experiment_name, params, X_train, y_train, X_test, y_test, encoder):
    mlflow.set_experiment(experiment_name)
    
    run_name = params.pop("name")
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)
        
        # Train
        clf = train_classifier(X_train, y_train, **params)
        
        # Evaluate
        report = evaluate_classifier(clf, X_test, y_test, encoder)
        
        # Log metrics
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])
        mlflow.log_metric("f1_maybe", report["maybe"]["f1-score"])
        
        # Log model
        mlflow.sklearn.log_model(clf, "model")

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
    # logistic = train_classifier(X_train_emb, y_train)

    # Evaluation
    # report = evaluate_classifier(logistic, X_test_emb, y_test, encoder)

    # MLflow experiment

    experiments = [
        {"name":"dummy", "classifier": "dummy"},
        {"name":"logistic-no_weights", "classifier": "logistic", "C": 1.0, "max_iter": 1000},
        {"name":"logistic-base", "classifier": "logistic", "C": 1.0, "class_weight": "balanced", "max_iter": 1000},
        {"name":"logistic-C10", "classifier": "logistic", "C": 10.0, "class_weight": "balanced", "max_iter": 1000},
        {"name":"logistic-C01", "classifier": "logistic", "C": 0.1, "class_weight": "balanced", "max_iter": 1000},
        {"name":"svm-base", "classifier": "svm", "C": 1.0, "class_weight": "balanced", "kernel": "rbf", "gamma": "scale"},
        {"name":"svm-g01", "classifier": "svm", "C": 1.0, "class_weight": "balanced", "kernel": "rbf", "gamma": 0.1},
        {"name":"RF-base", "classifier": "random_forest", "n_estimators": 100, "class_weight": "balanced"},
        {"name":"RF-n200", "classifier": "random_forest", "n_estimators": 200, "class_weight": "balanced"},
        {"name":"RF-max_depth10", "classifier": "random_forest", "n_estimators": 100, "class_weight": "balanced", "max_depth": 10},
        {"name":"RF-max_depth20", "classifier": "random_forest", "n_estimators": 100, "class_weight": "balanced", "max_depth": 20},
        {"name":"RF-no_weights", "classifier": "random_forest", "n_estimators": 100},   
    ]

    for exp in experiments:
        run_experiment('text classifier more', exp, X_train_emb, y_train, X_test_emb, y_test, encoder)