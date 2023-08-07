import pickle

from tools import *
from typing import List, Dict, Any
from collections import defaultdict



# Base Tokenizer
class Tokenizer:
    def __init__(self, corpus: List[str] = None):
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

        opt = []
        for item in p_text:

            splits = [[l for l in word] for word in item]
            for pair, merge in self.merge_rules.items():
                for idx, split in enumerate(splits):
                    i = 0
                    while i < len(split) - 1:
                        if split[i] == pair[0] and split[i + 1] == pair[1]:
                            split = split[:i] + [merge] + split[i + 2 :]
                        else:
                            i += 1
                    splits[idx] = split
            
            opt.append(sum(splits, []))

        return opt

    # Save the trained tokenizer
    def save(self, name: str, dir_pth: str):
        file_pth = dir_pth + f'{name}.pkl'
        with open(file_pth, 'wb') as f:
            pickle.dump(self.merge_rules, f)
            pickle.dump(self.vocab, f)

    # Load the trained tokenizer
    def _load(self, file_path: str):
        with open(file_path, 'rb') as f:
            self.merge_rules = pickle.load(f)
            self.vocab = pickle.load(f)


# Tokenizer with BPE Algorithm
class BPETokenizer(Tokenizer):
    '''
    Without Deal with Unknown Tokens
    '''
    def __init__(self, corpus: List[str] = None, _vocab_size: int = None):
        super(BPETokenizer, self).__init__(corpus)

        self._vocab_size = _vocab_size

        # Special Token
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 16000
    
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
    def train(self, special_tokens: List[str] = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']):
        if self.corpus is None:
            print('Please input the corpus!')
            return

        # Preprocess
        word_list = bpe_pre_process(self.corpus)  # [[...], [...], ...]
        flatten_word_list = [item for sublist in word_list for item in sublist]  # [...]
        self.word_freqs = self.count_word_freqs(flatten_word_list)  # Include Normalization & Pre-Tokenization 
        
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
    
    def tokenize(self, text: Any):
        if isinstance(text, str):
            p_text = bpe_pre_process([text])
        else:
            p_text = bpe_pre_process(text)
        
        return self._tokenize(p_text)

    def encode(self, text: Any)-> List[List[int]]:
        tokens_batch = self.tokenize(text)
        opt = []
        for item in tokens_batch:
            tmp = []
            for token in item:
                if token in self.vocab:
                    tmp.append(self.vocab.index(token))
                else:
                    tmp.append(self.vocab.index('<UNK>'))
            tmp.append(self.vocab.index('<EOS>'))
            opt.append(tmp)
        return opt
    
    def decode(self, tokens_batch: List[List[int]]):
        opt = []
        for item in tokens_batch:
            tmp = []
            for token in item:
                tmp.append(self.vocab[token])
            opt.append(tmp)
        return opt
    
    def load(self, file_path: str):
        self._load(file_path)
        
        # Special Token
        self.pad_id = self.vocab.index('<PAD>')
        self.bos_id = self.vocab.index('<BOS>')
        self.eos_id = self.vocab.index('<EOS>')
        self.unk_id = self.vocab.index('<UNK>')


class WordPieceTokenizer(Tokenizer):
    def __init__(self, corpus: List[str] = None, _vocab_size: int = None):
        super(WordPieceTokenizer, self).__init__(corpus)

        self._vocab_size = _vocab_size

        # Special Token
        