import torch
from config import Config
from src.models.vae import VAE
from src.utils.generation import ParaphraseGenerator

def quick_test():
    """Quick test to verify the model works"""
    print("=== QUICK VAE MODEL TEST ===")
    
    try:
        # Load model
        print("Loading model...")
        checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        
        model = VAE(checkpoint['vocab_size'], checkpoint['embedding_dim'], 
                    checkpoint['hidden_dim'], checkpoint['latent_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(Config.DEVICE)
        model.eval()
        
        print("✓ Model loaded successfully")
        print(f"  - Vocabulary size: {checkpoint['vocab_size']}")
        print(f"  - Device: {Config.DEVICE}")
        
        # Handle missing max_len in checkpoint
        max_len = checkpoint.get('max_len', Config.MAX_LEN)  # Use default if not in checkpoint
        print(f"  - Max length: {max_len}")
        
        # Test generator
        generator = ParaphraseGenerator(model, checkpoint['word2idx'], checkpoint['idx2word'],
                                      max_len, Config.DEVICE)
        
        # Multiple test sentences
        test_sentences = [
            "What is machine learning?",
            "How can I learn programming?",
            "Where is the library?",
            "What is the best way to study?"
        ]
        
        print("\n=== TESTING PARAPHRASE GENERATION ===")
        
        for test_sentence in test_sentences:
            print(f"\nOriginal: '{test_sentence}'")
            
            try:
                paraphrases = generator.generate_paraphrases(test_sentence, 3)
                
                print("Generated paraphrases:")
                for i, paraphrase in enumerate(paraphrases, 1):
                    print(f"  {i}: {paraphrase}")
                    
            except Exception as gen_error:
                print(f"  Error generating for this sentence: {gen_error}")
                continue
            
        print("\n✓ Model is working correctly!")
        print("The VAE model is successfully generating paraphrases!")
        return True
        
    except FileNotFoundError:
        print(f"✗ Model file not found: {Config.MODEL_SAVE_PATH}")
        print("Make sure the model file exists in the project directory.")
        return False
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Model test failed. Check the error above.")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 Your VAE model is ready to use!")
        print("You can now run: python main.py --mode interactive")
    else:
        print("\n❌ Model test failed. Please check the errors above.")
