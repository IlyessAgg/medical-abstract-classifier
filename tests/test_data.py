from src.data import preprocess, get_splits
from collections import Counter

def test_preprocess_output_lengths():
    dataset = [
        {
            "question": "Is X effective ?",
            "final_decision": "yes",
            "context": {
                "contexts": ["doc1", "doc2", "doc3"]
            }
        },
        {
            "question": "Does Y work ?",
            "final_decision": "no",
            "context": {
                "contexts": ["doc1", "doc2"]
            }
        }
    ]
    texts, labels, encoder = preprocess(dataset)
    assert len(texts) == len(labels)


def test_get_splits_sizes():
    texts = ["text1", "text2", "text3", "text4", "text5", "text6", "text7", "text8"]
    labels = ["yes", "yes", "yes", "no", "no", "no", "maybe", "maybe"]
    X_train, X_test, y_train, y_test = get_splits(texts, labels, test_size=0.25)

    assert len(X_train) + len(X_test) == len(texts)
    assert len(y_train) + len(y_test) == len(labels)

    # Stratification check
    orig_counts = Counter(labels)
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)

    # Check all labels are present
    assert set(y_train) == set(labels)
    assert set(y_test) == set(labels)

    # Check approximate distribution
    for label in orig_counts:
        orig_ratio = orig_counts[label] / len(labels)
        train_ratio = train_counts[label] / len(y_train)
        test_ratio = test_counts[label] / len(y_test)

        assert abs(train_ratio - orig_ratio) < 0.3
        assert abs(test_ratio - orig_ratio) < 0.3