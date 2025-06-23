import torch.nn as nn
from .components import VAE_Encoder, VAE_Decoder

class VAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(vocab_size, embedding_dim, hidden_dim, latent_dim)
        self.decoder = VAE_Decoder(vocab_size, embedding_dim, hidden_dim, latent_dim)

    def forward(self, input_sequence, target_sequence=None, word_dropout_rate=0.3, word2idx=None):
        z, mu, logvar = self.encoder(input_sequence)

        if target_sequence is not None:  # Training
            logits = self.decoder(z, target_sequence, word_dropout_rate=word_dropout_rate, 
                                training=self.training, word2idx=word2idx)
            return logits, mu, logvar
        else:  # Generation
            generated = self.decoder(z, training=False, word2idx=word2idx)
            return generated, mu, logvar
