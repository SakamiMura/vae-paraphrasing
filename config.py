import torch

class Config:
    # Data paths
    TRAIN_CSV_PATH = "train.csv"
    GLOVE_PATH = "glove.6B.100d.txt"
    MODEL_SAVE_PATH = "anti_collapse_vae_model.pth"
    
    # Model parameters
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    LATENT_DIM = 32
    MIN_WORD_FREQ = 5
    MAX_LEN = 90
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.1
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generation parameters
    NUM_PARAPHRASE_SAMPLES = 5
