import os
import torch
import random
import itertools

from torch.utils.data import Dataset


def txt_clean(corpus_path):
    file_names = os.listdir(corpus_path)
    for file_name in file_names:
        with open(os.path.join(corpus_path, file_name), 'r') as f:
            lines = f.readlines()
        with open(os.path.join(corpus_path, file_name), 'w') as f:
            for line in lines:
                if len(line) > 50:
                    f.write(line)
    

def produce_data_pair_corpus(corpus_path, tokenizer, max_seq_len=256):
    file_names = os.listdir(corpus_path)
    data_pair_corpus = []
    for file_name in file_names:
        print('Processing file: ', file_name)
        with open(os.path.join(corpus_path, file_name), 'r') as f:
            lines = f.readlines()
        for line in lines:
            tokens = tokenizer.tokenize(line)[0]
            num_token = len(tokens)
            num_pair = num_token // max_seq_len + 1
            for i in range(num_pair):
                if i == num_pair - 1:
                    percent = random.uniform(0.4, 0.6)
                    index = int(percent * (num_token - i * max_seq_len))
                    data_pair_corpus.append([tokens[i*max_seq_len:i*max_seq_len+index], tokens[i*max_seq_len+index:]])
                else:
                    percent = random.uniform(0.4, 0.6)
                    index = int(percent * max_seq_len)
                    data_pair_corpus.append((tokens[i*max_seq_len:i*max_seq_len+index], tokens[i*max_seq_len+index:(i+1)*max_seq_len]))

    return data_pair_corpus


class BERTDataset(Dataset):
    def __init__(self, data_pair_corpus, tokenizer, 
                 mask_token: str, 
                 cls_token: str,
                 pad_token: str,
                 seq_token: str,
                 max_seq_len = 256):
        # Config
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_token = mask_token
        self.cls_token = cls_token
        self.pad_token = pad_token
        self.seq_token = seq_token

        self.corpus_lines = len(data_pair_corpus)
        self.lines = data_pair_corpus

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):

        # Step 1: get random sentence pair, either negative or positive (saved as is_next_label)
        t1, t2, is_next_label = self.get_sent(item)

        # Step 2: replace random words in sentence with mask / random words
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
         # Adding PAD token for labels
        t1 = [self.cls_token] + t1_random + [self.seq_token]
        t2 = t2_random + [self.seq_token]
        t1_label = [self.pad_token] + t1_label + [self.pad_token]
        t2_label = t2_label + [self.pad_token]

        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as max_seq_len
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.max_seq_len]
        bert_input = (t1 + t2)[:self.max_seq_len]
        bert_label = (t1_label + t2_label)[:self.max_seq_len]
        padding = [self.pad_token for _ in range(self.max_seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)
        
        # To input_ids
        # bert_input = self.tokenizer.encode(bert_input)
        # bert_label = self.tokenizer.encode(bert_label)
        
        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        # return {key: torch.tensor(value) for key, value in output.items()}
        return {key: torch.tensor([1,3,2]) for key, value in output.items()}

    def random_word(self, tokens):
        label = tokens.copy()
        # 15% of the tokens would be replaced
        for i in range(len(tokens)):
            prob = random.random()

            if prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                    tokens[i] = self.mask_token

                # 10% chance change token to random token
                elif prob < 0.9:
                    tokens[i] = self.tokenizer.vocab[random.randrange(self.tokenizer.vocab_size)]
                # 10% chance change token to current token

        return tokens, label

    def get_sent(self, index):
        '''return random sentence pair'''
        t1, t2 = self.get_corpus_line(index)

        # negative or positive pair, for next sentence prediction
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        '''return sentence pair'''
        return self.lines[item][0], self.lines[item][1]

    def get_random_line(self):
        '''return random single sentence'''
        return self.lines[random.randrange(len(self.lines))][0]