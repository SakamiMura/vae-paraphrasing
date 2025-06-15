# Variational Autoencoder for Sentence Paraphrasing

## Project Description

This project implements a Variational Autoencoder (VAE) to perform sentence paraphrasing.  
The model is trained to encode input sentences into a latent vector and then decode them back into paraphrased versions.

Key components:
- Encoder & Decoder built with LSTM layers (PyTorch)
- GloVe word embeddings (100 dimensions)
- Custom loss function with KL divergence and annealing
- Teacher Forcing technique during training

This project was built as part of my exploration of Deep Learning and Natural Language Processing.

---

## Dataset

The dataset consists of paraphrased sentence pairs.  
**NOTE:** Dataset is not included due to size limitations.  
You can use any paraphrase dataset (e.g. Quora Question Pairs, PAWS, or custom data).

## Pre-trained embeddings

The project uses pre-trained GloVe embeddings (100d).  
You can download them from [GloVe official website](https://nlp.stanford.edu/projects/glove/).

Place the embedding file in:  
`data/glove.6B.100d.txt`

---

## How to run

1️⃣ Install dependencies:
```bash
pip install -r requirements.txt
2️⃣ Prepare data files and embeddings as described above.

3️⃣ Run training:

bash
Kopiuj
Edytuj
python vae_paraphrasing.py
Results
The model trains using both reconstruction loss and KL divergence.

KL annealing was implemented to stabilize the learning process.

Sample paraphrased sentences can be generated after training.

Technologies
Python 3

PyTorch

NumPy

Matplotlib

GloVe embeddings

Author
Created by Sakami (2025)

yaml
Kopiuj
Edytuj

---

### requirements.txt (prosty):

```txt
torch
numpy
matplotlib
tqdm