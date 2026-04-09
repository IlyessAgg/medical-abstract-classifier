# Medical abstract classifier

> Building a medical abstract classifier using [PubMed](https://pubmedqa.github.io/index.html) data.

<div align="center">

<p align="center">
  <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2JobDF0NWJuMnl2cXYyOG5pY2FuaTZvNHp0eGl1YnFieWl2ODY5bCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/6xjato9VF2mWSzStXN/giphy.gif" alt="Doctor's here" height=350/>
</p>

![CI](https://github.com/IlyessAgg/medical-abstract-classifier/workflows/CI/badge.svg)

</div>

The main goal of this project is to build something cool whilst learning new tools. Right now my environment is `WSL2` and I'm trying to use regular `.py` scripts instead of `.ipynb` notebooks.  
Some tools I'd like to fiddle with are : *HuggingFace, MLflow, Docker, FastAPI, GitHub Actions.*

## Data

This project utilizes the [labeled PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) dataset, which poses a challenging class imbalance problem. The dataset consists of 552 `yes` samples, 338 `no` samples, and 110 `maybe` samples. To handle it properly, I will stratify the splitting of the data, and will probably tinker with the `class_weight` argument of classifiers, making sure it's set to `balanced`.  
Another important preprocessing step involves *tokenization*. The current tokenizer has a **512** `max_length` limit, enforced via truncation, which means we might throw away meaningful information. A more robust approach would involve chunking longer texts, embedding each segment, and then aggregating these embeddings for the final representation.

## Model

A basic classifier achieves an accuracy of around **50%**, which is better than random but not better than a *dummy* classifier. The `maybe` class is essentially not being learned *(F1 of 0.25)*, likely due to its inherent ambiguity and low prevalence in the data.  

<p align="center">
  <img src="assets/MLflow_experiment.png" alt="MLflow run"/>
</p>

> MLflow UI displaying multiple runs for the text classification task with different classifiers and hyperparameters, including baselines.

An interesting finding is that `logistic-no_weights` *(no class_weight balancing)* gets the best accuracy at **0.59** but `f1_maybe` of **0**. This reflects a common precision-recall tradeoff where the model optimizes for overall accuracy by focusing on majority classes while neglecting minority classes.


## API Usage

To interact with the model, you can use the `/predict` endpoint. Here's an example of how to use `curl` to make a POST request with a sample input:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "Do blood cells transport oxygen ?",
  "context": "Red blood cells (RBCs), referred to as erythrocytes in academia and medical publishing, also known as red cells, erythroid cells, and rarely haematids, are the most common type of blood cell and the vertebrates principal means of delivering oxygen (O2) to the body tissues—via blood flow through the circulatory system."
}'
```
This will return a predicted label (`yes`, `no`, or `maybe`) with the associated confidence score.

> Be cautious of **unsupported** characters, such as line breaks in the context field, which will throw an error 422.

## Project Structure

Here's an overview of the project structure:

```
├── src/                      # Source code directory            
│   ├── __init__.py
│   ├── api.py                # FastAPI app and endpoint definitions
│   ├── data.py               # Data loading and preprocessing
│   ├── evaluate.py           # Evaluation metrics and reports
│   ├── model.py              # Embeddings encoding and model training
├── models/                   # Saved models and encoders
├── data/                     # Data files (e.g., embeddings_train.npy)
├── .github/                  # GitHub configuration
│   └── workflows/            # CI/CD workflows (e.g., ci.yml)
├── tests/                    # Unit tests for data processing and pipeline
│   └── test_data.py
├── requirements.txt          # Python dependencies for the project
├── Dockerfile                # Docker configuration for containerizing the app
├── main.py                   # Entry point: runs the full training pipeline
├── assets/                   # Images and other assets for the README
└── README.md
```

## Continuous Integration

This project uses **GitHub Actions** to automatically run linting (with `ruff`) and unit tests on every push and pull request.  
Passing tests and style checks are indicated with the badge above.

## Tools Used

- **HuggingFace**: For using pre-trained models and datasets (e.g., PubMedQA).
- **MLflow**: For tracking and logging model training experiments.
- **FastAPI**: To create the API for serving the model and making predictions.
- **Docker**: For containerizing the application to simplify deployment.
- **GitHub Actions**: Continuous Integration for tests and code quality.