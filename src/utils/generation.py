import torch
from nltk.tokenize import word_tokenize

class ParaphraseGenerator:
    def __init__(self, model, word2idx, idx2word, max_len, device):
        self.model = model
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.max_len = max_len
        self.device = device
    
    def generate_paraphrases(self, input_sentence, num_samples=5):
        """Generate multiple paraphrases using latent space sampling"""
        self.model.eval()
        
        # Tokenize and encode input
        tokens = ['< SOS >'] + word_tokenize(input_sentence.lower()) + ['<EOS>']
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        # Pad to max_len
        if len(indices) < self.max_len:
            indices.extend([self.word2idx['<PAD>']] * (self.max_len - len(indices)))
        else:
            indices = indices[:self.max_len]
        
        input_tensor = torch.tensor(indices, device=self.device).unsqueeze(0)
        
        paraphrases = []
        with torch.no_grad():
            z, mu, logvar = self.model.encoder(input_tensor)
            
            for i in range(num_samples):
                # Sample from latent distribution
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std, device=self.device)
                z_sample = mu + eps * std
                
                # Generate sequence
                generated_indices = self.model.decoder(z_sample, target_sequence=None, 
                                                     training=False, word2idx=self.word2idx)
                
                # Convert to words
                generated_words = []
                for idx in generated_indices[0]:
                    token_id = idx.item()
                    word = self.idx2word[token_id]
                    if word == '<EOS>':
                        break
                    if word not in ['< SOS >', '<PAD>']:
                        generated_words.append(word)
                
                paraphrase = ' '.join(generated_words)
                paraphrases.append(paraphrase)
        
        return paraphrases
