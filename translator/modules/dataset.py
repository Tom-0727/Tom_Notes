import os
import html
import requests

from typing import List, Dict
from torch.utils.data import Dataset


# Get IWSLT En-Vi Data
def get_iwsltenvi_data(store_dir: str = './'):
    train_en_url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en'
    train_vi_url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi'
    test_en_url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en'
    test_vi_url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi'

    store_pth = f'{store_dir}data/iwsltenvi/'
    os.makedirs(store_pth, exist_ok=True)
    print('The Data would be stored in: ', store_pth)

    if not os.path.exists(f'{store_pth}train_en.txt'):
        train_en = requests.get(train_en_url).text
        with open(f'{store_pth}train_en.txt', 'w') as f:
            f.write(train_en)
    
    if not os.path.exists(f'{store_pth}train_vi.txt'):
        train_vi = requests.get(train_vi_url).text
        with open(f'{store_pth}train_vi.txt', 'w') as f:
            f.write(train_vi)
    
    if not os.path.exists(f'{store_pth}test_en.txt'):
        test_en = requests.get(test_en_url).text
        with open(f'{store_pth}test_en.txt', 'w') as f:
            f.write(test_en)
    
    if not os.path.exists(f'{store_pth}test_vi.txt'):
        test_vi = requests.get(test_vi_url).text
        with open(f'{store_pth}test_vi.txt', 'w') as f:
            f.write(test_vi)
    
    print('Done!')


# Load & Clean the data (Convert HTML-encoded characters to normal)
def load_iwsltenvi_data(train: bool = True,
                        test: bool = True,
                        data_dir: str = './data/iwsltenvi/'):
    
    train_data = {}
    test_data = {}
    # Load the data
    if train:
        with open(f'{data_dir}train_en.txt', 'r') as f:
            en_text = html.unescape(f.read()).split('\n')
        with open(f'{data_dir}train_vi.txt', 'r') as f:
            vi_text = html.unescape(f.read()).split('\n')
        train_data['en'] = en_text
        train_data['vi'] = vi_text
        
    if test:
        with open(f'{data_dir}test_en.txt', 'r') as f:
            en_text += html.unescape(f.read()).split('\n')
        with open(f'{data_dir}test_vi.txt', 'r') as f:
            vi_text += html.unescape(f.read()).split('\n')

        test_data['en'] = en_text
        test_data['vi'] = vi_text
    
    return train_data, test_data


# IWSLTDataset
class IWSLTDataset(Dataset):
    def __init__(self, 
                 data: Dict[str, List[str]]):
        
        self.data = data
        
    def __getitem__(self, index):
        # The output should be the string pair: (src: str, trg: str)
        src = '<BOS>' + self.data['en'][index] + '<EOS>'
        trg = '<BOS>' + self.data['vi'][index] + '<EOS>'

        return src, trg
        
    def __len__(self):
        return len(self.data['en'])