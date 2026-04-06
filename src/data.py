from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_pubmedqa():
    """Load the PubMedQA dataset and return the train split."""
    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    return dataset

def preprocess(dataset):
    """
    Extract inputs and labels from raw dataset.
    
    Input text: combine question + all context strings into one string.
    Label: final_decision (encode as integer).
    
    Returns: texts (list of str), labels (list of int), encoder (LabelEncoder)
    """
    texts = []
    labels = []

    for example in dataset:
        text = example['question'] + ' ' + ' '.join(example['context']['contexts'])
        texts.append(text)

    # encode string labels to integers using LabelEncoder
    encoder = LabelEncoder()
    labels = encoder.fit_transform(dataset['final_decision']).tolist()

    return texts, labels, encoder


def get_splits(texts, labels, test_size=0.2, random_state=42):
    """Split into train/test sets."""
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size,
                                                        random_state=random_state, stratify=labels)
    return X_train, X_test, y_train, y_test