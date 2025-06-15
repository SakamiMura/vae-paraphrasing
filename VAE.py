# ============================================================================
# imports   
# ============================================================================
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# ============================================================================


# Wczytanie danych
df = pd.read_csv("train.csv")
# Filtrowanie tylko tych wierszy, które są parafrazami
df_paraphrase = df[df['is_duplicate'] == 1].copy()

# Stworzymy listę par tokenów [(tokeny_zdanie1, tokeny_zdanie2), ...]
tokenized_pairs = []

for q1, q2 in zip(df_paraphrase['question1'], df_paraphrase['question2']):
    if pd.isna(q1) or pd.isna(q2):
        continue  # pomiń jeśli brakuje któregokolwiek pytania
    tokens1 = word_tokenize(q1.lower())
    tokens2 = word_tokenize(q2.lower())
    tokenized_pairs.append((tokens1, tokens2))


# Zliczamy wszystkie słowa w tokenach (z obu pytań)
word_counts = Counter(
    token           # to konkretne słowo
    for pair in tokenized_pairs      # dla każdej pary (tokens1, tokens2)
    for sentence in pair             # w obu pytaniach z pary
    for token in sentence            # dla każdego słowa w pytaniu
)

# print("Liczba unikalnych słów:", len(word_counts))
# print("10 najczęstszych słów:", word_counts.most_common(10))



# 1. Lista słów z częstotliwością >= 5 (reszta wyleci)
filtered_words = [word for word, freq in word_counts.items() if freq >= 5]

# 2. Specjalne tokeny na początek
special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']

# 3. Łączymy specjalne tokeny + nasze najczęstsze słowa
vocab = special_tokens + filtered_words

# 4. Mapa: słowo → indeks
word2idx = {}
for idx, word in enumerate(vocab):
    word2idx[word] = idx

# 5. Mapa: indeks → słowo
idx2word = {}
for word, idx in word2idx.items():
    idx2word[idx] = word

# 6. Zamień słowa na indeksy
encoded_pairs = []  # tu będziemy zapisywać zakodowane pary

for tokens1, tokens2 in tokenized_pairs:
    encoded1 = []  # tu zapiszemy indeksy dla pierwszego pytania
    encoded2 = []  # i tu dla drugiego

    # Dodaj początek zdania
    encoded1.append(word2idx['<SOS>'])
    encoded2.append(word2idx['<SOS>'])

    # Zamień słowa z tokens1 na indeksy
    for word in tokens1:
        if word in word2idx:
            encoded1.append(word2idx[word])
        else:
            encoded1.append(word2idx['<UNK>'])

    # Zamień słowa z tokens2 na indeksy
    for word in tokens2:
        if word in word2idx:
            encoded2.append(word2idx[word])
        else:
            encoded2.append(word2idx['<UNK>'])

    # Dodaj koniec zdania
    encoded1.append(word2idx['<EOS>'])
    encoded2.append(word2idx['<EOS>'])

    # Zamień na tensory i zapisz do listy
    encoded_pairs.append((torch.tensor(encoded1), torch.tensor(encoded2)))


# 7. Wczytanie pre-trained GloVe embeddings
glove_path = "glove.6B.100d.txt"  # lub ścieżka względna
embedding_dim = 100
glove_embeddings = {}

with open(glove_path, encoding="utf8") as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vector = torch.tensor([float(val) for val in parts[1:]], dtype=torch.float)
        glove_embeddings[word] = vector

# === Stworzenie macierzy embeddingów ===
vocab_size = len(word2idx)
embedding_matrix = torch.zeros((vocab_size, embedding_dim))

for word, idx in word2idx.items():
    if word in glove_embeddings:
        embedding_matrix[idx] = glove_embeddings[word]
    else:
        embedding_matrix[idx] = torch.randn(embedding_dim)  # nie ma w GloVe = losowy

# 8. Padnijemy sekwencje do tej samej długości

# Rozdziel zakodowane pary na dwie osobne listy
inputs = []   # zdanie wejściowe (question1)
targets = []  # zdanie alternatywne (question2)

for pair in encoded_pairs:
    input_tensor = pair[0]
    target_tensor = pair[1]

    inputs.append(input_tensor)
    targets.append(target_tensor)

# Sprawdźmy max długości sekwencji zeby dopasowac pad
max_len = max(pad.shape[0] for pad in inputs + targets)


# Inputy i cele są różnej długości, więc musimy je wyrównać
def pad_to_fixed_length(tensor_list, max_len, pad_value):
    padded_list = []
    for t in tensor_list:
        if len(t) < max_len:
            pad_len = max_len - len(t)
            padded = torch.cat([t, torch.full((pad_len,), pad_value)])
        else:
            padded = t[:max_len]  # przytnij, jeśli dłuższe
        padded_list.append(padded)
    return torch.stack(padded_list)

padded_inputs = pad_to_fixed_length(inputs, max_len, word2idx['<PAD>'])
padded_targets = pad_to_fixed_length(targets, max_len, word2idx['<PAD>'])


# Spreparowane dane czyste i pachnące
# padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=word2idx['<PAD>'])
# padded_targets = pad_sequence(targets, batch_first=True, padding_value=word2idx['<PAD>'])
# # Podgląd
# print("Pojedyncze zdanie (tensor):", padded_inputs[0])
# print("Długość sekwencji po paddingu:", padded_inputs.shape[1])
# print("Liczba zdań (batch size):", padded_inputs.shape[0])


# Podział na zestaw treningowy i testowy (np. 90% train, 10% test)
train_inputs, test_inputs, train_targets, test_targets = train_test_split(
    padded_inputs,
    padded_targets,
    test_size=0.1,
    random_state=42  # żeby podział był zawsze taki sam
)

# # Podgląd kształtów (czy wszystko OK)
# print("Train input shape:", train_inputs.shape)
# print("Train target shape:", train_targets.shape)
# print("Test input shape:", test_inputs.shape)
# print("Test target shape:", test_targets.shape)



#PODSUMOWANIE:
# Wczytanie danych	✅
# Filtrowanie parafraz	✅
# Tokenizacja	✅
# Zliczenie słów + słownik	✅
# Zamiana słów na indeksy	✅
# Padding	✅
# Train/test split	✅

# TODO: Zaimplementuj encoder i decoder - FIXED VERSION FOR POSTERIOR COLLAPSE

import torch.nn.functional as F

# Tworzymy klasę Encoder
class VAE_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim):
        super(VAE_Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        # Zamiast tylko jednego ukrytego stanu mamy teraz dwa: mean i logvar
        self.hidden_to_mean = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, input_sequence):
        embedded = self.embedding(input_sequence)                      # [batch, seq, emb]
        _, hidden = self.gru(embedded)                                 # [1, batch, hidden]
        hidden = hidden.squeeze(0)                                     # [batch, hidden]

        mu = self.hidden_to_mean(hidden)                               # [batch, latent]
        logvar = self.hidden_to_logvar(hidden)                         # [batch, latent]

        std = torch.exp(0.5 * logvar)                                  # [batch, latent]
        eps = torch.randn_like(std)                                    # [batch, latent]
        z = mu + eps * std                                             # reparametrization trick

        return z, mu, logvar

class VAE_Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim):
        super(VAE_Decoder, self).__init__()

        # KLUCZ: Decoder ma MNIEJSZY hidden_dim żeby był bardziej zależny od z
        decoder_hidden_dim = hidden_dim // 2  # 128 zamiast 256

        self.latent_to_hidden = nn.Linear(latent_dim, decoder_hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # KLUCZ: z jest podawane na KAŻDYM kroku jako dodatkowy input
        self.gru = nn.GRU(embedding_dim + latent_dim, decoder_hidden_dim, batch_first=True)
        self.fc = nn.Linear(decoder_hidden_dim, vocab_size)

        self.latent_dim = latent_dim
        self.decoder_hidden_dim = decoder_hidden_dim

    def forward(self, z, target_sequence=None, max_length=90, word_dropout_rate=0.3, training=True):
        batch_size = z.size(0)
        device = z.device

        # Initial hidden state z latent
        hidden = self.latent_to_hidden(z).unsqueeze(0)  # [1, batch, hidden]

        if target_sequence is not None:  # Training mode - Teacher forcing z WORD DROPOUT
            seq_len = target_sequence.size(1) - 1  # Remove last token
            embedded = self.embedding(target_sequence[:, :-1])  # [batch, seq-1, emb]

            # WORD DROPOUT: Randomly replace tokens with <UNK> during training
            if training and word_dropout_rate > 0:
                dropout_mask = torch.rand(batch_size, seq_len, device=device) < word_dropout_rate
                unk_token = word2idx['<UNK>']
                target_dropped = target_sequence[:, :-1].clone()
                target_dropped[dropout_mask] = unk_token
                embedded = self.embedding(target_dropped)

            # KLUCZ: Concat z na każdym kroku
            z_expanded = z.unsqueeze(1).expand(batch_size, seq_len, self.latent_dim)  # [batch, seq, latent]
            decoder_input = torch.cat([embedded, z_expanded], dim=2)  # [batch, seq, emb+latent]

            output, _ = self.gru(decoder_input, hidden)
            logits = self.fc(output)
            return logits

        else:  # Generation mode
            outputs = []
            input_token = torch.full((batch_size, 1), word2idx['<SOS>'], dtype=torch.long, device=device)

            for _ in range(max_length):
                embedded = self.embedding(input_token)  # [batch, 1, emb]
                z_step = z.unsqueeze(1)  # [batch, 1, latent]
                decoder_input = torch.cat([embedded, z_step], dim=2)  # [batch, 1, emb+latent]

                output, hidden = self.gru(decoder_input, hidden)
                logits = self.fc(output.squeeze(1))
                predicted = torch.argmax(logits, dim=-1, keepdim=True)
                outputs.append(predicted)
                input_token = predicted

                if torch.all(predicted.squeeze() == word2idx['<EOS>']):
                    break

            return torch.cat(outputs, dim=1)

class VAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(vocab_size, embedding_dim, hidden_dim, latent_dim)
        self.decoder = VAE_Decoder(vocab_size, embedding_dim, hidden_dim, latent_dim)

    def forward(self, input_sequence, target_sequence=None, word_dropout_rate=0.3):
        z, mu, logvar = self.encoder(input_sequence)

        if target_sequence is not None:  # Training
            logits = self.decoder(z, target_sequence, word_dropout_rate=word_dropout_rate, training=self.training)
            return logits, mu, logvar
        else:  # Generation
            generated = self.decoder(z, training=False)
            return generated, mu, logvar

def vae_loss_with_free_bits(logits, targets, mu, logvar, kl_weight=1.0, free_bits=2.0):
    """
    VAE Loss with Free Bits to prevent posterior collapse

    Free Bits: Pozwala na pewną minimalną wartość KL bez kary
    Zapobiega całkowitemu collapse do prior
    """
    # Reconstruction loss (cross entropy)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets[:, 1:].reshape(-1)  # Skip <SOS> token

    # Ignore padding tokens in loss calculation
    mask = (targets_flat != word2idx['<PAD>'])
    recon_loss = nn.functional.cross_entropy(logits_flat[mask], targets_flat[mask])

    # KL divergence loss PER DIMENSION z Free Bits
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [batch, latent_dim]
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits / kl_per_dim.size(1))  # Free bits per dim
    kl_loss = torch.sum(kl_per_dim) / batch_size

    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss

# Device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# PRELOAD SEKCJA
# ============================================================================

# Load the pretrained model
print("Loading pretrained VAE model...")
checkpoint = torch.load('anti_collapse_vae_model.pth', map_location=device)

# CRITICAL: Extract ALL saved parameters, especially vocab dictionaries
vocab_size = checkpoint['vocab_size']
embedding_dim = checkpoint['embedding_dim']
hidden_dim = checkpoint['hidden_dim']
latent_dim = checkpoint['latent_dim']

# CRITICAL: Use the EXACT same vocabularies that were used during training
word2idx = checkpoint['word2idx']
idx2word = checkpoint['idx2word']

# Load training history
training_losses = checkpoint['training_losses']
reconstruction_losses = checkpoint['reconstruction_losses']
kl_losses = checkpoint['kl_losses']

print(f"Loaded vocab size: {vocab_size}")
print(f"Current embedding_matrix size: {embedding_matrix.shape}")

# Verify vocabulary consistency
if vocab_size != len(word2idx):
    print(f"WARNING: Vocab size mismatch! Checkpoint: {vocab_size}, Current: {len(word2idx)}")

if embedding_matrix.shape[0] != vocab_size:
    print(f"WARNING: Embedding matrix size mismatch! Expected: {vocab_size}, Got: {embedding_matrix.shape[0]}")

# Initialize model with loaded parameters
vae_model = VAE(vocab_size, embedding_dim, hidden_dim, latent_dim).to(device)

# Load the trained weights (this includes the correct embeddings!)
vae_model.load_state_dict(checkpoint['model_state_dict'])

# DO NOT overwrite embeddings! They're already loaded correctly from state_dict
# REMOVED: vae_model.encoder.embedding.weight.data = embedding_matrix.to(device)
# REMOVED: vae_model.decoder.embedding.weight.data = embedding_matrix.to(device)

print("Pretrained VAE model loaded successfully!")
print(f"Model architecture: vocab_size={vocab_size}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, latent_dim={latent_dim}")
print(f"Final training loss from checkpoint: {training_losses[-1]:.4f}")

# CRITICAL: Recalculate max_len from the loaded vocabulary
# Find the maximum length that was used during training by checking saved data shapes
# We need to ensure we use the same max_len that was used during training

# For now, let's extract it from the checkpoint or set it based on typical values
# Check if max_len was saved in checkpoint
if 'max_len' in checkpoint:
    max_len = checkpoint['max_len']
    print(f"Using saved max_len: {max_len}")
else:
    # If not saved, we need to be careful - let's use a reasonable default
    # and verify with the user that this matches training
    max_len = 90  # This was likely the value used during training
    print(f"Using default max_len: {max_len} (verify this matches training!)")

# Validation: Test the model with a simple sentence to verify it works
print("\n=== MODEL VALIDATION ===")
test_sentence = "what is python"
tokens = ['<SOS>'] + word_tokenize(test_sentence.lower()) + ['<EOS>']
indices = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]

# Pad to max_len
if len(indices) < max_len:
    indices.extend([word2idx['<PAD>']] * (max_len - len(indices)))
else:
    indices = indices[:max_len]

test_input = torch.tensor(indices, device=device).unsqueeze(0)

vae_model.eval()
with torch.no_grad():
    try:
        z, mu, logvar = vae_model.encoder(test_input)
        print(f"Encoder test successful - latent shape: {z.shape}")
        print(f"μ stats: mean={mu.mean().item():.4f}, std={mu.std().item():.4f}")
        print(f"σ stats: mean={torch.exp(0.5 * logvar).mean().item():.4f}")

        # Test decoder
        generated = vae_model.decoder(z, target_sequence=None, training=False)
        print(f"Decoder test successful - output shape: {generated.shape}")

        # Convert back to words to verify
        words = []
        for idx in generated[0][:10]:  # First 10 tokens
            word = idx2word[idx.item()]
            words.append(word)
            if word == '<EOS>':
                break
        print(f"Sample generation: {' '.join(words)}")

    except Exception as e:
        print(f"Model validation FAILED: {e}")
        print("This indicates the model was not loaded correctly!")

# Move data to device for evaluation (only if they exist)
if 'train_inputs' in globals():
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)
    print("Data moved to device successfully")
else:
    print("WARNING: Training data not found in current session")
"""
============================================================================
TRAINING SEKCJA
============================================================================
"""
# # Initialize complete VAE model with ANTI-COLLAPSE settings
# vocab_size = len(word2idx)
# embedding_dim = 100
# hidden_dim = 256  # Encoder będzie miał 256, decoder 128
# latent_dim = 32   # Mniejszy latent na początku

# vae_model = VAE(vocab_size, embedding_dim, hidden_dim, latent_dim).to(device)

# # FREEZE embeddings na początku, potem unfreeze
# vae_model.encoder.embedding.weight.data = embedding_matrix.to(device)
# vae_model.encoder.embedding.weight.requires_grad = False  # FREEZE na początku
# vae_model.decoder.embedding.weight.data = embedding_matrix.to(device)
# vae_model.decoder.embedding.weight.requires_grad = False  # FREEZE na początku

# print("Embeddings FROZEN for initial training")

# # Training setup
# optimizer = optim.Adam(vae_model.parameters(), lr=0.001)  # Wyższy LR na początku
# num_epochs = 20
# batch_size = 32

# # Create data loaders
# train_dataset = TensorDataset(train_inputs, train_targets)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Training loop with AGGRESSIVE KL annealing i STAGED training
# training_losses = []
# reconstruction_losses = []
# kl_losses = []

# vae_model.train()
# for epoch in range(num_epochs):
#     epoch_loss = 0
#     epoch_recon_loss = 0
#     epoch_kl_loss = 0

#     # STAGED TRAINING z embedding unfreezing
#     if epoch == 10:
#         # Unfreeze embeddings po 10 epokach
#         vae_model.encoder.embedding.weight.requires_grad = True
#         vae_model.decoder.embedding.weight.requires_grad = True
#         # Zmniejsz learning rate dla fine-tuningu
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 0.0005
#         print("Embeddings UNFROZEN, LR reduced to 0.0005")

#     # BARDZO AGRESYWNE KL annealing
#     if epoch < 8:
#         kl_weight = 0.0  # Pierwsze 8 epok: tylko rekonstrukcja
#     elif epoch < 12:
#         kl_weight = 0.01  # Epoki 8-12: bardzo mały KL weight
#     elif epoch < 16:
#         kl_weight = 0.05  # Epoki 12-16: powolny wzrost
#     else:
#         kl_weight = 0.1   # Ostatnie epoki: maksymalnie 0.1 (nie 0.5!)

#     # Word dropout rate - większy na początku
#     word_dropout = 0.4 if epoch < 5 else 0.3 if epoch < 10 else 0.2

#     for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
#         optimizer.zero_grad()

#         # Forward pass z word dropout
#         logits, mu, logvar = vae_model(input_batch, target_batch, word_dropout_rate=word_dropout)

#         # Calculate loss z Free Bits
#         total_loss, recon_loss, kl_loss = vae_loss_with_free_bits(
#             logits, target_batch, mu, logvar, kl_weight, free_bits=1.0
#         )

#         # Backward pass
#         total_loss.backward()

#         # Gradient clipping to prevent explosion
#         torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=1.0)

#         optimizer.step()

#         epoch_loss += total_loss.item()
#         epoch_recon_loss += recon_loss.item()
#         epoch_kl_loss += kl_loss.item()

#         if batch_idx % 1000 == 0:
#             print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, '
#                   f'Loss: {total_loss.item():.4f}, '
#                   f'Recon: {recon_loss.item():.4f}, '
#                   f'KL: {kl_loss.item():.4f}, '
#                   f'KL Weight: {kl_weight:.3f}, '
#                   f'Word Dropout: {word_dropout:.2f}')

#     avg_loss = epoch_loss / len(train_loader)
#     avg_recon_loss = epoch_recon_loss / len(train_loader)
#     avg_kl_loss = epoch_kl_loss / len(train_loader)

#     training_losses.append(avg_loss)
#     reconstruction_losses.append(avg_recon_loss)
#     kl_losses.append(avg_kl_loss)

#     print(f'Epoch {epoch+1}/{num_epochs} completed - '
#           f'Avg Loss: {avg_loss:.4f}, '
#           f'Avg Recon: {avg_recon_loss:.4f}, '
#           f'Avg KL: {avg_kl_loss:.4f}, '
#           f'KL Weight: {kl_weight:.3f}')

# # Save the trained model
# torch.save({
#     'model_state_dict': vae_model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'vocab_size': vocab_size,
#     'embedding_dim': embedding_dim,
#     'hidden_dim': hidden_dim,
#     'latent_dim': latent_dim,
#     'word2idx': word2idx,
#     'idx2word': idx2word,
#     'training_losses': training_losses,
#     'reconstruction_losses': reconstruction_losses,
#     'kl_losses': kl_losses,
#     'device': device.type
# }, 'anti_collapse_vae_model.pth')

# print("Anti-collapse VAE model saved successfully!")
# print(f"Final training loss: {training_losses[-1]:.4f}")
# print(f"Training completed on {device}")
# # """

# TODO: Wygeneruj nowe parafrazy z różnych próbek z

def generate_paraphrases(model, input_sentence, num_samples=5):
    """
    Generate paraphrases using different samples from the latent space

    FIXED: Properly handle decoder output for the new anti-collapse architecture
    Reparametrization trick w akcji: z = μ + ε ⋅ σ
    Każda próbka ε daje inną parafrazę przy tych samych μ i σ
    """
    model.eval()

    # Get device from model
    device = next(model.parameters()).device

    # Tokenize input sentence
    tokens = ['<SOS>'] + word_tokenize(input_sentence.lower()) + ['<EOS>']
    indices = []
    for token in tokens:
        if token in word2idx:
            indices.append(word2idx[token])
        else:
            indices.append(word2idx['<UNK>'])

    # Pad to max_len
    if len(indices) < max_len:
        indices.extend([word2idx['<PAD>']] * (max_len - len(indices)))
    else:
        indices = indices[:max_len]

    input_tensor = torch.tensor(indices, device=device).unsqueeze(0)  # [1, seq_len]

    paraphrases = []
    with torch.no_grad():
        # Get latent representation
        z, mu, logvar = model.encoder(input_tensor)

        # Generate multiple samples from the same distribution
        for i in range(num_samples):
            # Sample new z using reparametrization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std, device=device)
            z_sample = mu + eps * std

            # FIXED: Call decoder directly in generation mode (no target_sequence)
            generated_indices = model.decoder(z_sample, target_sequence=None, training=False)

            # Convert indices back to words
            generated_words = []
            for idx in generated_indices[0]:  # First sample in batch
                token_id = idx.item()
                word = idx2word[token_id]
                if word == '<EOS>':
                    break
                if word not in ['<SOS>', '<PAD>']:
                    generated_words.append(word)

            paraphrase = ' '.join(generated_words)
            paraphrases.append(f"Sample {i+1}: {paraphrase}")

    return paraphrases

# Alternative generation function that uses the full VAE model
def generate_paraphrases_alt(model, input_sentence, num_samples=5):
    """
    Alternative generation using the full VAE model forward pass
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenize input sentence
    tokens = ['<SOS>'] + word_tokenize(input_sentence.lower()) + ['<EOS>']
    indices = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]

    # Pad to max_len
    if len(indices) < max_len:
        indices.extend([word2idx['<PAD>']] * (max_len - len(indices)))
    else:
        indices = indices[:max_len]

    input_tensor = torch.tensor(indices, device=device).unsqueeze(0)

    paraphrases = []
    with torch.no_grad():
        for i in range(num_samples):
            # Generate without target (generation mode)
            generated_indices, _, _ = model(input_tensor, target_sequence=None)

            # Convert to words
            generated_words = []
            for idx in generated_indices[0]:
                token_id = idx.item()
                word = idx2word[token_id]
                if word == '<EOS>':
                    break
                if word not in ['<SOS>', '<PAD>']:
                    generated_words.append(word)

            paraphrase = ' '.join(generated_words)
            paraphrases.append(f"Sample {i+1}: {paraphrase}")

    return paraphrases

# Test generation with improved model
test_sentences = [
    "How can I learn physics easily?",
    "Where is the biggest library in the world?",
    "Where can I find good resources for machine learning?",
]

print("=== TESTING IMPROVED VAE - METHOD 1 ===")
for sentence in test_sentences:
    print(f"\nOriginal: {sentence}")
    print("Generated paraphrases:")
    try:
        paraphrases = generate_paraphrases(vae_model, sentence, num_samples=3)
        for paraphrase in paraphrases:
            print(f"  {paraphrase}")
    except Exception as e:
        print(f"  Error with method 1: {e}")
        print("  Trying alternative method...")
        paraphrases = generate_paraphrases_alt(vae_model, sentence, num_samples=3)
        for paraphrase in paraphrases:
            print(f"  {paraphrase}")

# Additional test: Show latent space variability
def test_latent_variability(model, input_sentence, num_samples=5):
    """Test if latent space shows proper variability"""
    model.eval()
    device = next(model.parameters()).device

    tokens = ['<SOS>'] + word_tokenize(input_sentence.lower()) + ['<EOS>']
    indices = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]

    if len(indices) < max_len:
        indices.extend([word2idx['<PAD>']] * (max_len - len(indices)))
    else:
        indices = indices[:max_len]

    input_tensor = torch.tensor(indices, device=device).unsqueeze(0)

    print(f"\n=== LATENT SPACE VARIABILITY TEST ===")
    print(f"Input: {input_sentence}")

    with torch.no_grad():
        z, mu, logvar = model.encoder(input_tensor)
        std = torch.exp(0.5 * logvar)

        print(f"Latent statistics:")
        print(f"  μ mean: {mu.mean().item():.4f}, std: {mu.std().item():.4f}")
        print(f"  σ mean: {std.mean().item():.4f}, std: {std.std().item():.4f}")
        print(f"  KL divergence: {(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item():.4f}")

test_latent_variability(vae_model, "How can I learn physics easily?")

# TODO: Oceń model

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluate_model():
    """Evaluate the VAE model using various metrics"""

    # Get device from model
    device = next(vae_model.parameters()).device

    # 1. Plot training losses
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

    # 2. Latent space analysis
    print("\n=== LATENT SPACE ANALYSIS ===")
    with torch.no_grad():
        sample_inputs = test_inputs[:100]  # Already on correct device
        z, mu, logvar = vae_model.encoder(sample_inputs)

        print(f"Latent dimension: {latent_dim}")
        print(f"Mean μ statistics:")
        print(f"  Mean: {mu.mean().item():.4f}")
        print(f"  Std:  {mu.std().item():.4f}")
        print(f"Log-variance statistics:")
        print(f"  Mean: {logvar.mean().item():.4f}")
        print(f"  Std:  {logvar.std().item():.4f}")

    # 3. Generation diversity test
    print("\n=== GENERATION DIVERSITY TEST ===")
    test_sentences = [
        "What is artificial intelligence?",
        "How do I learn Python programming?",
        "What is the meaning of life?"
    ]

    for sentence in test_sentences:
        print(f"\nOriginal: {sentence}")
        paraphrases = generate_paraphrases(vae_model, sentence, num_samples=3)
        for paraphrase in paraphrases:
            print(f"  {paraphrase}")

# Function to load saved model on any device
def load_model(model_path, device=None):
    """Load saved model with device compatibility"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)

    # Create model with saved parameters
    model = VAE(
        checkpoint['vocab_size'],
        checkpoint['embedding_dim'],
        checkpoint['hidden_dim'],
        checkpoint['latent_dim']
    ).to(device)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded on {device}")
    return model, checkpoint

# Run evaluation
evaluate_model()
