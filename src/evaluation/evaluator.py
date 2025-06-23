import torch
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def plot_training_losses(self, training_losses, reconstruction_losses, kl_losses):
        """Plot training losses"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(training_losses)
        plt.title('Total Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 2)
        plt.plot(reconstruction_losses)
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 3, 3)
        plt.plot(kl_losses)
        plt.title('KL Divergence Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_latent_space(self, test_inputs, latent_dim):
        """Analyze latent space statistics"""
        self.model.eval()
        with torch.no_grad():
            z, mu, logvar = self.model.encoder(test_inputs[:100])
            
            print(f"Latent dimension: {latent_dim}")
            print(f"Mean μ statistics:")
            print(f"  Mean: {mu.mean().item():.4f}")
            print(f"  Std:  {mu.std().item():.4f}")
            print(f"Log-variance statistics:")
            print(f"  Mean: {logvar.mean().item():.4f}")
            print(f"  Std:  {logvar.std().item():.4f}")
