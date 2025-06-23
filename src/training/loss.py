import torch
import torch.nn as nn

def vae_loss_with_free_bits(logits, targets, mu, logvar, word2idx, kl_weight=1.0, free_bits=2.0):
    """VAE Loss with Free Bits to prevent posterior collapse"""
    # Reconstruction loss
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets[:, 1:].reshape(-1)

    # Ignore padding tokens
    mask = (targets_flat != word2idx['<PAD>'])
    recon_loss = nn.functional.cross_entropy(logits_flat[mask], targets_flat[mask])

    # KL divergence with Free Bits
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits / kl_per_dim.size(1))
    kl_loss = torch.sum(kl_per_dim) / batch_size

    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss
