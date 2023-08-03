from collections import defaultdict
from typing import List, Dict
from tools import *


# Base Tokenizer
class Tokenizer:
    def __init__(self, corpus: List[str]):
        # Input
        self.corpus = corpus
        
        # Target
        self.merge_rules = {}
        self.vocab = []

        # Some Dicts
        self.word_freqs = {}
        self.splits  ={}

    @property
    def vocab_size(self):
        return len(self.vocab)
    
    # Count Word Frequencies -> Dict['word': freq]
    def count_word_freqs(self, word_list: List[str])-> Dict[str, int]:
        word_freqs = {}
        
        for word in word_list:
            if word not in word_freqs.keys():
                word_freqs[word] = 1
            else:
                word_freqs[word] += 1
        return word_freqs

    # get the pair frequencies based on word_freqs -> Dict[('w', 'o'): 1, ...]
    def compute_pair_freqs(self):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    # tokenize
    def _tokenize(self, p_text):
        splits = [[l for l in word] for word in p_text]
        for pair, merge in self.merge_rules.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split

        return sum(splits, [])


# Tokenizer with BPE Algorithm
class BPETokenizer(Tokenizer):
    '''
    Without Deal with Unknown Tokens
    '''
    def __init__(self, corpus: List[str], _vocab_size: int):
        super(BPETokenizer, self).__init__(corpus)

        self._vocab_size = _vocab_size
    
    # Get the vocabulary -> List['a', 'b', ...]
    def get_base_vocab(self, special_tokens: List[str] = ['PAD', 'BOS', 'EOS'])-> List[str]:
        alphabet = []
        for key in self.word_freqs.keys():
            for char in key:
                if char not in alphabet:
                    alphabet.append(char)
        alphabet.sort()
        base_vocab = special_tokens + alphabet.copy()
        return base_vocab
    
    # Get the splits -> Dict['word': ['w', 'o', 'r', 'd']]
    def get_splits(self)-> Dict[str, List[str]]:
        self.splits = {word: [c for c in word] for word in self.word_freqs.keys()}

    # merge the pair<a, b> in_place the splits
    def merge_pair(self, a, b):
        for word in self.word_freqs:
            split = self.splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            self.splits[word] = split

    # Train the tokenizer
    def train(self, special_tokens: List[str] = ['PAD', 'BOS', 'EOS']):
        # Preprocess
        word_list = bpe_pre_process(self.corpus)
        self.word_freqs = self.count_word_freqs(word_list)  # Include Normalization & Pre-Tokenization 
        self.vocab = self.get_base_vocab(special_tokens)  # Could add more special tokens
        self.get_splits()

        # Training
        while len(self.vocab) < self._vocab_size:
            pair_freqs = self.compute_pair_freqs()
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            self.merge_pair(*best_pair)
            self.merge_rules[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])
    
    def tokenize(self, text):
        p_text = bpe_pre_process([text])
        
        return self._tokenize(p_text)


