import torch
import argparse
from config import Config
from src.data.preprocessing import DataPreprocessor
from src.models.vae import VAE
from src.training.trainer import VAETrainer
from src.utils.generation import ParaphraseGenerator
from src.evaluation.evaluator import ModelEvaluator

def train_model():
    """Train a new VAE model"""
    print("Starting training...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(Config.MIN_WORD_FREQ, Config.MAX_LEN)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    tokenized_pairs = preprocessor.load_and_filter_data(Config.TRAIN_CSV_PATH)
    vocab = preprocessor.build_vocabulary(tokenized_pairs)
    encoded_pairs = preprocessor.encode_pairs(tokenized_pairs)
    
    # Pad sequences and split data
    inputs, targets = preprocessor.pad_sequences(encoded_pairs)
    train_inputs, test_inputs, train_targets, test_targets = preprocessor.split_data(inputs, targets)
    
    # Load embeddings
    embedding_matrix = preprocessor.load_glove_embeddings(Config.GLOVE_PATH, Config.EMBEDDING_DIM)
    
    # Initialize model
    model = VAE(preprocessor.vocab_size, Config.EMBEDDING_DIM, Config.HIDDEN_DIM, Config.LATENT_DIM)
    model.to(Config.DEVICE)
    
    # Set embeddings
    model.encoder.embedding.weight.data = embedding_matrix.to(Config.DEVICE)
    model.decoder.embedding.weight.data = embedding_matrix.to(Config.DEVICE)
    model.encoder.embedding.weight.requires_grad = False
    model.decoder.embedding.weight.requires_grad = False
    
    # Train model
    trainer = VAETrainer(model, preprocessor.word2idx, Config.DEVICE, Config.LEARNING_RATE)
    trainer.train(train_inputs.to(Config.DEVICE), train_targets.to(Config.DEVICE), 
                  Config.NUM_EPOCHS, Config.BATCH_SIZE)
    
    # Save model
    trainer.save_model(Config.MODEL_SAVE_PATH, preprocessor.vocab_size, Config.EMBEDDING_DIM,
                      Config.HIDDEN_DIM, Config.LATENT_DIM, preprocessor.idx2word, Config.MAX_LEN)
    
    print("Training completed!")

def load_and_generate():
    """Load trained model and generate paraphrases"""
    print("Loading pretrained model...")
    
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    
    # Load model
    model = VAE(checkpoint['vocab_size'], checkpoint['embedding_dim'], 
                checkpoint['hidden_dim'], checkpoint['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    
    # Initialize generator
    generator = ParaphraseGenerator(model, checkpoint['word2idx'], checkpoint['idx2word'],
                                  checkpoint['max_len'], Config.DEVICE)
    
    # Test sentences
    test_sentences = [
        "How can I learn physics easily?",
        "Where is the biggest library in the world?",
        "Where can I find good resources for machine learning?",
    ]
    
    print("\n=== GENERATING PARAPHRASES ===")
    for sentence in test_sentences:
        print(f"\nOriginal: {sentence}")
        paraphrases = generator.generate_paraphrases(sentence, Config.NUM_PARAPHRASE_SAMPLES)
        for i, paraphrase in enumerate(paraphrases, 1):
            print(f"  {i}: {paraphrase}")

def evaluate_model():
    """Evaluate trained model"""
    print("Evaluating model...")
    
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    
    # Load model
    model = VAE(checkpoint['vocab_size'], checkpoint['embedding_dim'], 
                checkpoint['hidden_dim'], checkpoint['latent_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    
    evaluator = ModelEvaluator(model, Config.DEVICE)
    
    # Plot training losses
    evaluator.plot_training_losses(checkpoint['training_losses'], 
                                 checkpoint['reconstruction_losses'],
                                 checkpoint['kl_losses'])

def interactive_generate():
    """Interactive paraphrase generation with user input"""
    print("Loading pretrained model...")
    
    try:
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        print(f"Model loaded successfully from {Config.MODEL_SAVE_PATH}")
        
        # Load model
        model = VAE(checkpoint['vocab_size'], checkpoint['embedding_dim'], 
                    checkpoint['hidden_dim'], checkpoint['latent_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(Config.DEVICE)
        model.eval()
        
        print(f"Model architecture: vocab_size={checkpoint['vocab_size']}, "
              f"embedding_dim={checkpoint['embedding_dim']}, "
              f"hidden_dim={checkpoint['hidden_dim']}, "
              f"latent_dim={checkpoint['latent_dim']}")
        
        # Initialize generator
        generator = ParaphraseGenerator(model, checkpoint['word2idx'], checkpoint['idx2word'],
                                      checkpoint['max_len'], Config.DEVICE)
        
        print("\n=== INTERACTIVE PARAPHRASE GENERATION ===")
        print("Enter sentences to generate paraphrases (type 'quit' to exit)")
        print("Example: 'How can I learn programming?'")
        
        while True:
            user_input = input("\nEnter sentence: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not user_input:
                print("Please enter a valid sentence.")
                continue
                
            try:
                print(f"\nOriginal: {user_input}")
                print("Generated paraphrases:")
                paraphrases = generator.generate_paraphrases(user_input, Config.NUM_PARAPHRASE_SAMPLES)
                
                for i, paraphrase in enumerate(paraphrases, 1):
                    print(f"  {i}: {paraphrase}")
                    
            except Exception as e:
                print(f"Error generating paraphrases: {e}")
                print("Try a simpler sentence or check if the model was trained properly.")
                
    except FileNotFoundError:
        print(f"Model file not found: {Config.MODEL_SAVE_PATH}")
        print("Please ensure you have trained the model first or the model file exists.")
    except Exception as e:
        print(f"Error loading model: {e}")

def main():
    parser = argparse.ArgumentParser(description='VAE Paraphrasing Tool')
    parser.add_argument('--mode', choices=['train', 'generate', 'evaluate', 'interactive'], required=True,
                       help='Mode to run: train, generate, evaluate, or interactive')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model()
    elif args.mode == 'generate':
        load_and_generate()
    elif args.mode == 'evaluate':
        evaluate_model()
    elif args.mode == 'interactive':
        interactive_generate()

if __name__ == "__main__":
    main()
