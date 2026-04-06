from sklearn.metrics import classification_report

def evaluate_classifier(clf, X_test, y_test, encoder):
    """
    Return a dict of metrics: accuracy, f1 (weighted).
    Also print a classification report.
    hint: sklearn.metrics has classification_report
    """
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    return report