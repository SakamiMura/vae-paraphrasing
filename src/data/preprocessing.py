import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch
from sklearn.model_selection import train_test_split

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class DataPreprocessor:
    def __init__(self, min_word_freq=5, max_len=90):
        self.min_word_freq = min_word_freq
        self.max_len = max_len
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def load_and_filter_data(self, csv_path):
        """Load CSV data and filter for paraphrases only"""
        df = pd.read_csv(csv_path)
        df_paraphrase = df[df['is_duplicate'] == 1].copy()
        
        tokenized_pairs = []
        for q1, q2 in zip(df_paraphrase['question1'], df_paraphrase['question2']):
            if pd.isna(q1) or pd.isna(q2):
                continue
            tokens1 = word_tokenize(q1.lower())
            tokens2 = word_tokenize(q2.lower())
            tokenized_pairs.append((tokens1, tokens2))
            
        return tokenized_pairs
    
    def build_vocabulary(self, tokenized_pairs):
        """Build vocabulary from tokenized pairs"""
        word_counts = Counter(
            token
            for pair in tokenized_pairs
            for sentence in pair
            for token in sentence
        )
        
        filtered_words = [word for word, freq in word_counts.items() if freq >= self.min_word_freq]
        special_tokens = ['<PAD>', '<UNK>', '< SOS >', '<EOS>']
        vocab = special_tokens + filtered_words
        
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(vocab)
        
        return vocab
    
    def encode_pairs(self, tokenized_pairs):
        """Encode tokenized pairs to indices"""
        encoded_pairs = []
        
        for tokens1, tokens2 in tokenized_pairs:
            encoded1 = [self.word2idx['< SOS >']]
            encoded2 = [self.word2idx['< SOS >']]
            
            for word in tokens1:
                encoded1.append(self.word2idx.get(word, self.word2idx['<UNK>']))
            for word in tokens2:
                encoded2.append(self.word2idx.get(word, self.word2idx['<UNK>']))
                
            encoded1.append(self.word2idx['<EOS>'])
            encoded2.append(self.word2idx['<EOS>'])
            
            encoded_pairs.append((torch.tensor(encoded1), torch.tensor(encoded2)))
            
        return encoded_pairs
    
    def pad_sequences(self, encoded_pairs):
        """Pad sequences to fixed length"""
        inputs = [pair[0] for pair in encoded_pairs]
        targets = [pair[1] for pair in encoded_pairs]
        
        def pad_to_fixed_length(tensor_list, max_len, pad_value):
            padded_list = []
            for t in tensor_list:
                if len(t) < max_len:
                    pad_len = max_len - len(t)
                    padded = torch.cat([t, torch.full((pad_len,), pad_value)])
                else:
                    padded = t[:max_len]
                padded_list.append(padded)
            return torch.stack(padded_list)
        
        padded_inputs = pad_to_fixed_length(inputs, self.max_len, self.word2idx['<PAD>'])
        padded_targets = pad_to_fixed_length(targets, self.max_len, self.word2idx['<PAD>'])
        
        return padded_inputs, padded_targets
    
    def split_data(self, inputs, targets, test_size=0.1):
        """Split data into train and test sets"""
        return train_test_split(inputs, targets, test_size=test_size, random_state=42)
    
    def load_glove_embeddings(self, glove_path, embedding_dim=100):
        """Load pre-trained GloVe embeddings"""
        glove_embeddings = {}
        
        with open(glove_path, encoding="utf8") as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                vector = torch.tensor([float(val) for val in parts[1:]], dtype=torch.float)
                glove_embeddings[word] = vector
        
        # Create embedding matrix
        embedding_matrix = torch.zeros((self.vocab_size, embedding_dim))
        for word, idx in self.word2idx.items():
            if word in glove_embeddings:
                embedding_matrix[idx] = glove_embeddings[word]
            else:
                embedding_matrix[idx] = torch.randn(embedding_dim)
                
        return embedding_matrix
