import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .loss import vae_loss_with_free_bits

class VAETrainer:
    def __init__(self, model, word2idx, device, lr=0.001):
        self.model = model
        self.word2idx = word2idx
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        self.training_losses = []
        self.reconstruction_losses = []
        self.kl_losses = []
    
    def train_epoch(self, data_loader, epoch, num_epochs, kl_weight, word_dropout):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
            input_batch = input_batch.to(self.device)
            target_batch = target_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, mu, logvar = self.model(input_batch, target_batch, 
                                          word_dropout_rate=word_dropout, word2idx=self.word2idx)
            
            total_loss, recon_loss, kl_loss = vae_loss_with_free_bits(
                logits, target_batch, mu, logvar, self.word2idx, kl_weight, free_bits=1.0
            )
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            if batch_idx % 1000 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, '
                      f'Loss: {total_loss.item():.4f}, '
                      f'Recon: {recon_loss.item():.4f}, '
                      f'KL: {kl_loss.item():.4f}')
        
        return epoch_loss / len(data_loader), epoch_recon_loss / len(data_loader), epoch_kl_loss / len(data_loader)
    
    def train(self, train_inputs, train_targets, num_epochs=20, batch_size=32):
        """Full training loop"""
        train_dataset = TensorDataset(train_inputs, train_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            # Staged training and KL annealing
            if epoch == 10:
                self.model.encoder.embedding.weight.requires_grad = True
                self.model.decoder.embedding.weight.requires_grad = True
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.0005
                print("Embeddings UNFROZEN, LR reduced")
            
            # KL weight scheduling
            if epoch < 8:
                kl_weight = 0.0
            elif epoch < 12:
                kl_weight = 0.01
            elif epoch < 16:
                kl_weight = 0.05
            else:
                kl_weight = 0.1
            
            word_dropout = 0.4 if epoch < 5 else 0.3 if epoch < 10 else 0.2
            
            avg_loss, avg_recon, avg_kl = self.train_epoch(
                train_loader, epoch, num_epochs, kl_weight, word_dropout
            )
            
            self.training_losses.append(avg_loss)
            self.reconstruction_losses.append(avg_recon)
            self.kl_losses.append(avg_kl)
            
            print(f'Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, '
                  f'Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}')
    
    def save_model(self, filepath, vocab_size, embedding_dim, hidden_dim, latent_dim, 
                   idx2word, max_len):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
            'word2idx': self.word2idx,
            'idx2word': idx2word,
            'max_len': max_len,
            'training_losses': self.training_losses,
            'reconstruction_losses': self.reconstruction_losses,
            'kl_losses': self.kl_losses,
        }, filepath)
