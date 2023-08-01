from collections import defaultdict
from typing import List, Dict
from tools import *


# Tokenizer with BPE Algorithm
class BPETokenizer:
    def __init__(self, corpus: List[str], vocab_size: int):
        # Input
        self.corpus = corpus
        self.vocab_size = vocab_size
        
        # Some dicts
        self.merge_rules = {}
        self.vocab = []
        self.word_freqs = {}
        self.splits = {}

    # Count Word Frequencies -> Dict['word': freq]
    def count_word_freqs(self)-> Dict[str, int]:
        word_freqs = {}
        word_list = pre_process(self.corpus)
        for word in word_list:
            if word not in word_freqs.keys():
                word_freqs[word] = 1
            else:
                word_freqs[word] += 1
        return word_freqs
    
    # Get the vocabulary -> List['a', 'b', ...]
    def get_vocab(self, special_tokens: List[str])-> List[str]:
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
    def train(self, special_tokens: List[str]):
        # Preprocess
        self.word_freqs = self.count_word_freqs()  # Include Normalization & Pre-Tokenization 
        self.vocab = self.get_vocab(special_tokens)  # Could add more special tokens
        self.get_splits()

        # Training
        while len(self.vocab) < self.vocab_size:
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

    # tokenize
    def tokenize(self, text):
        p_text = pre_process([text])
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

