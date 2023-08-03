import unicodedata

from typing import List


# lowercase & NFD
def normalization(text: str)-> str:
    text = unicodedata.normalize("NFD", text)
    text = text.lower()
    return text


# Pre-Tokenization: split by space & turn space to Ġ
def bpe_pre_tokenization(text: str)-> List[str]:
    tmp = ''
    opt = []
    for i in text:
        if i != " ":
            tmp += i
        else:
            opt.append(tmp)
            tmp = 'Ġ'
    opt.append(tmp)
    return opt


# Synthesis
def bpe_pre_process(texts: List[str])-> List[str]:
    opt = []
    for text in texts:
        text = normalization(text)
        text_lst = bpe_pre_tokenization(text)
        opt.extend(text_lst)
    return opt

    