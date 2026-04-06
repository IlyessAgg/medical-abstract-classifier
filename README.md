# Medical abstract classifier
> Building a medical abstract classifier using [PubMed](https://pubmedqa.github.io/index.html) data.

<p align="center">
  <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2JobDF0NWJuMnl2cXYyOG5pY2FuaTZvNHp0eGl1YnFieWl2ODY5bCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/6xjato9VF2mWSzStXN/giphy.gif" alt="Doctor's here" height=350/>
</p>

The main goal of this project is to build something cool whilst learning new tools. Right now my environment is `WSL2` and I'm trying to use regular `.py` scripts instead of `.ipynb` notebooks.  
Some tools I'd like to fiddle with are : *HuggingFace, MLflow, Docker, FastAPI, GitHub Actions.*

## Thoughts
Using [labeled PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA):
- Classe imbalance to be wary of : 552 `yes` ; 338 `no` ; 110 `maybe`  
- The current tokenizer has a `512` max_length limit. For now the limit is enforced via truncation, which means we might throw away meaningful information. A better approach would be to chunk texts, embed each chunk then aggregate.