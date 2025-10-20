import re
import numpy as np
from collections import defaultdict

class Tokenizer:
    def __init__(self, num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
        self.num_words = num_words
        self.filters = filters
        self.word_index = {}
        self.word_counts = defaultdict(int)
        self.document_count = 0
        
    def fit_on_texts(self, texts):
        """Fit the tokenizer on texts"""
        for text in texts:
            self.document_count += 1
            # Clean text
            text = self._clean_text(text)
            # Split into words
            words = text.split()
            for word in words:
                self.word_counts[word] += 1
        
        # Create word index
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add special tokens
        self.word_index['<pad>'] = 0
        self.word_index['<unk>'] = 1
        
        # Add words to index
        for word, count in sorted_words:
            if self.num_words and len(self.word_index) >= self.num_words:
                break
            if word not in self.word_index:
                self.word_index[word] = len(self.word_index)
    
    def texts_to_sequences(self, texts):
        """Convert texts to sequences of integers"""
        sequences = []
        for text in texts:
            text = self._clean_text(text)
            words = text.split()
            sequence = []
            for word in words:
                if word in self.word_index:
                    sequence.append(self.word_index[word])
                else:
                    sequence.append(self.word_index['<unk>'])
            sequences.append(sequence)
        return sequences
    
    def _clean_text(self, text):
        """Clean text by removing filters"""
        for char in self.filters:
            text = text.replace(char, ' ')
        # Remove extra spaces
        text = ' '.join(text.split())
        return text

def pad_sequences(sequences, maxlen=None, padding='pre', truncating='pre', value=0):
    """Pad sequences to the same length"""
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq = seq[-maxlen:]
            else:
                seq = seq[:maxlen]
        
        if padding == 'pre':
            padded_seq = [value] * (maxlen - len(seq)) + seq
        else:
            padded_seq = seq + [value] * (maxlen - len(seq))
        
        padded_sequences.append(padded_seq)
    
    return np.array(padded_sequences)

class WordsEmbeddings:
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding_matrix = None
    
    def load_embeddings(self, word_index):
        """Load embeddings - to be implemented by subclasses"""
        pass

class GloveEmbeddings(WordsEmbeddings):
    def __init__(self, vocab_size, embed_dim, glove_path):
        super().__init__(vocab_size, embed_dim)
        self.glove_path = glove_path
    
    def load_embeddings(self, word_index):
        """Load GloVe embeddings"""
        embeddings_index = {}
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        self.embedding_matrix = np.zeros((self.vocab_size, self.embed_dim))
        for word, i in word_index.items():
            if i < self.vocab_size:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    self.embedding_matrix[i] = embedding_vector
                else:
                    self.embedding_matrix[i] = np.random.normal(0, 0.1, self.embed_dim)
        
        return self.embedding_matrix

class FasttextEmbeddings(WordsEmbeddings):
    def __init__(self, vocab_size, embed_dim, fasttext_path):
        super().__init__(vocab_size, embed_dim)
        self.fasttext_path = fasttext_path
    
    def load_embeddings(self, word_index):
        """Load FastText embeddings"""
        # Placeholder for FastText loading
        # In practice, you would use the fasttext library
        self.embedding_matrix = np.random.normal(0, 0.1, (self.vocab_size, self.embed_dim))
        return self.embedding_matrix
