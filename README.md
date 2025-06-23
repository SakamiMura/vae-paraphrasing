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

```
vae-paraphrasing/
├── src/
│   ├── data/
│   │   └── preprocessing.py      # Data loading and preprocessing
│   ├── models/
│   │   ├── components.py         # VAE encoder and decoder components
│   │   └── vae.py               # Main VAE model class
│   ├── training/
│   │   ├── loss.py              # VAE loss with Free Bits
│   │   └── trainer.py           # Training loop and utilities
│   ├── utils/
│   │   └── generation.py        # Paraphrase generation utilities
│   └── evaluation/
│       └── evaluator.py         # Model evaluation and visualization
├── config.py                    # Configuration settings
├── main.py                      # Main entry point
├── quick_test.py               # Quick model verification
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Training a New Model

First, ensure you have the required files (`train.csv` and `glove.6B.100d.txt`) in the project root, then:

```bash
python main.py --mode train
```

This will:
- Load and preprocess the training data
- Build vocabulary from paraphrase pairs
- Initialize the VAE model with GloVe embeddings
- Train for 20 epochs with staged learning and KL annealing
- Save the trained model as `anti_collapse_vae_model.pth`

### 2. Quick Model Test

After training, verify your model works correctly:

```bash
python quick_test.py
```

This script tests the model with sample sentences and confirms it's generating paraphrases properly.

### 3. Interactive Paraphrase Generation

Launch the interactive mode to generate paraphrases for any input:

```bash
python main.py --mode interactive
```

Example session:
```
Enter sentence: How can I learn programming?

Original: How can I learn programming?
Generated paraphrases:
  1: what is the best way to learn coding
  2: how do i start learning to program
  3: where can i learn programming skills
  4: what are good resources for learning programming
  5: how to begin studying programming
```

## 🔧 Usage Modes

### Training Mode
```bash
python main.py --mode train
```
- Trains a new VAE model from scratch
- Requires `train.csv` and `glove.6B.100d.txt`
- Saves model to `anti_collapse_vae_model.pth`

### Generation Mode
```bash
python main.py --mode generate
```
- Generates paraphrases for predefined test sentences
- Requires trained model file

### Interactive Mode
```bash
python main.py --mode interactive
```
- Allows real-time paraphrase generation
- Enter any sentence and get multiple paraphrases
- Type 'quit' to exit

### Evaluation Mode
```bash
python main.py --mode evaluate
```
- Plots training loss curves
- Analyzes latent space statistics
- Requires matplotlib for visualization

## ⚙️ Configuration

Modify `config.py` to adjust model parameters:

```python
# Model architecture
EMBEDDING_DIM = 100      # GloVe embedding dimension
HIDDEN_DIM = 256         # RNN hidden size
LATENT_DIM = 32          # VAE latent space dimension

# Training parameters
BATCH_SIZE = 32          # Training batch size
NUM_EPOCHS = 20          # Number of training epochs
LEARNING_RATE = 0.001    # Initial learning rate

# Data parameters
MIN_WORD_FREQ = 5        # Minimum word frequency for vocabulary
MAX_LEN = 90             # Maximum sequence length
```

## 🧠 Model Architecture

### VAE Components
- **Encoder**: GRU-based encoder that maps input sequences to latent distributions
- **Decoder**: GRU-based decoder that generates sequences from latent codes
- **Embedding Layer**: Pre-trained GloVe embeddings (frozen initially, fine-tuned later)

### Anti-Collapse Techniques
1. **Free Bits**: Prevents KL divergence from collapsing to zero
2. **KL Annealing**: Gradually increases KL weight during training
3. **Word Dropout**: Randomly replaces input words with `<UNK>` tokens
4. **Staged Training**: Freezes embeddings initially, then fine-tunes
