# VAE-Based Paraphrase Generation

Variational Autoencoder (VAE) implementation for generating high-quality paraphrases from input text. This project addresses the common posterior collapse problem in text VAEs through advanced techniques including Free Bits, KL annealing, and word dropout.

## 🚀 Features

- **Anti-Collapse VAE Architecture**: Implements Free Bits technique and staged training to prevent posterior collapse
- **GloVe Embeddings Integration**: Leverages pre-trained GloVe embeddings for better semantic representations
- **Advanced Training Techniques**: 
  - KL divergence annealing
  - Word dropout regularization
  - Staged embedding fine-tuning
  - Gradient clipping
- **Flexible Generation**: Generate multiple diverse paraphrases from the same input using latent space sampling
- **Comprehensive Evaluation**: Built-in tools for analyzing model performance and latent space quality

## 📋 Prerequisites

### Required Files
Before running the project, ensure you have the following files in the project root:

1. **`train.csv`** - Training dataset containing question pairs with `is_duplicate` labels
   - Expected columns: `question1`, `question2`, `is_duplicate`
   - Download from Quora Question Pairs dataset or similar paraphrase datasets

2. **`glove.6B.100d.txt`** - Pre-trained GloVe embeddings (100-dimensional)
   - Download from: https://nlp.stanford.edu/projects/glove/
   - File size: ~347MB

3. **`anti_collapse_vae_model.pth`** - Pre-trained model (created after training)
   - Generated automatically during training
   - Required only for generation/evaluation modes

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- ~2GB disk space for embeddings and data

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/vae-paraphrasing.git
cd vae-paraphrasing
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download required NLTK data** (will be done automatically on first run):
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

## 📁 Project Structure

````markdown
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
````
