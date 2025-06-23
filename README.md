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


