from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

def meanpooling(output, mask):
    embeddings = output[0] # First element of model_output contains all token embeddings
    mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def get_embeddings(texts, batch_size=16):
    """
    Encode a list of texts into embeddings using PubMedBERT.
    Returns a numpy array of shape (n_texts, hidden_size).
    
    Use mean pooling over the last hidden state as the sentence embedding.
    Process in batches to avoid memory issues.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # tokenize with padding and truncation
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512)
        
        # forward pass
        with torch.no_grad():
            output = model(**inputs)

        # mean pool the last_hidden_state over the sequence dimension
        # last_hidden_state shape is (batch_size, seq_len, hidden_size)
        embeddings = meanpooling(output, inputs['attention_mask'])

        all_embeddings.append(embeddings.detach().cpu().numpy())

    return np.vstack(all_embeddings)


# Current get_embeddings just truncates, but since medical documents can be very long
# it'd be good to chunk texts, embed each chunk then aggregate.
def get_embeddings_chunked(texts, batch_size=16, max_length=512, stride=128):
    """
    Encode texts into embeddings using chunking for long inputs.

    Each text:
        → tokenized
        → split into overlapping chunks
        → each chunk embedded
        → chunk embeddings aggregated into one vector

    Returns:
        numpy array of shape (n_texts, hidden_size)
    """
    pass


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

def train_classifier(X_train, y_train, classifier="logistic", **params):
    """
    Train a classifier based on the specified model type.

    Parameters:
    - X_train: The training data features.
    - y_train: The training labels.
    - classifier: The type of classifier to train ('logistic', 'svm', 'random_forest').
    - params: Additional parameters for the classifier.

    Returns:
    - The trained classifier.
    """
    if classifier == "dummy":
        clf = DummyClassifier(strategy="most_frequent")
    elif classifier == "logistic":
        clf = LogisticRegression(random_state=23, **params)
    elif classifier == "svm":
        clf = SVC(random_state=23, **params)
    elif classifier == "random_forest":
        clf = RandomForestClassifier(random_state=23, **params)
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")
    
    clf.fit(X_train, y_train)
    return clf