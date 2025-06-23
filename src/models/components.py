import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim):
        super(VAE_Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.hidden_to_mean = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, input_sequence):
        embedded = self.embedding(input_sequence)
        _, hidden = self.gru(embedded)
        hidden = hidden.squeeze(0)

        mu = self.hidden_to_mean(hidden)
        logvar = self.hidden_to_logvar(hidden)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar

class VAE_Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim):
        super(VAE_Decoder, self).__init__()
        decoder_hidden_dim = hidden_dim // 2
        
        self.latent_to_hidden = nn.Linear(latent_dim, decoder_hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim + latent_dim, decoder_hidden_dim, batch_first=True)
        self.fc = nn.Linear(decoder_hidden_dim, vocab_size)

        self.latent_dim = latent_dim
        self.decoder_hidden_dim = decoder_hidden_dim

    def forward(self, z, target_sequence=None, max_length=90, word_dropout_rate=0.3, training=True, word2idx=None):
        batch_size = z.size(0)
        device = z.device

        hidden = self.latent_to_hidden(z).unsqueeze(0)

        if target_sequence is not None:  # Training mode
            seq_len = target_sequence.size(1) - 1
            embedded = self.embedding(target_sequence[:, :-1])

            if training and word_dropout_rate > 0 and word2idx is not None:
                dropout_mask = torch.rand(batch_size, seq_len, device=device) < word_dropout_rate
                unk_token = word2idx['<UNK>']
                target_dropped = target_sequence[:, :-1].clone()
                target_dropped[dropout_mask] = unk_token
                embedded = self.embedding(target_dropped)

            z_expanded = z.unsqueeze(1).expand(batch_size, seq_len, self.latent_dim)
            decoder_input = torch.cat([embedded, z_expanded], dim=2)

            output, _ = self.gru(decoder_input, hidden)
            logits = self.fc(output)
            return logits

        else:  # Generation mode
            if word2idx is None:
                raise ValueError("word2idx is required for generation mode")
                
            outputs = []
            input_token = torch.full((batch_size, 1), word2idx['< SOS >'], dtype=torch.long, device=device)

            for _ in range(max_length):
                embedded = self.embedding(input_token)
                z_step = z.unsqueeze(1)
                decoder_input = torch.cat([embedded, z_step], dim=2)

                output, hidden = self.gru(decoder_input, hidden)
                logits = self.fc(output.squeeze(1))
                predicted = torch.argmax(logits, dim=-1, keepdim=True)
                outputs.append(predicted)
                input_token = predicted

                if torch.all(predicted.squeeze() == word2idx['<EOS>']):
                    break

            return torch.cat(outputs, dim=1)
