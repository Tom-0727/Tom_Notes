
import sys
import json
import torch
import argparse

import torch.nn as nn

# Append Package Path
sys.path.append('../')
sys.path.append('../tokenization')

from modules import *
from transformer import *
from tokenization.tools import *
from tokenization.tokenizer import *


def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='0', type=str, required=False,
                        help='GPU Setting')
    parser.add_argument('--ptm_pth', default='./models/', type=str, required=False,
                        help='Path to pretrained model')
    parser.add_argument('--model_config', default='./config/standard_transformer.json', type=str, required=False,
                        help='Path to model config')
    parser.add_argument('--tokenizer_pth', default='./iwslt2013_tokenizer.pkl', type=str, required=False,
                        help='Path to the dataset for training')
    
    args = parser.parse_args()
    return args



# json to dict (For reading model config)
def json2dict(file_path):
    # Step 1: Read the .js file content and preprocess it
    with open(file_path, 'r') as js_file:
        js_content = js_file.read()
    
    # Step 2: Convert the preprocessed JSON-like content to a Python dictionary
    try:
        python_dict = json.loads(js_content)
        return python_dict
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")


def seq2seq_inference(input: str, 
                      tokenizer,
                      model,
                      args)-> str:

    # Tokenization
    input_ids = tokenizer.encode(input)
    if len(input_ids[0]) > args.max_len:
        input_ids[0] = input_ids[0][:args.max_len-1]
        input_ids[0].append(tokenizer.eos_id)
    
    # Inputs
    src = torch.tensor(input_ids, dtype=torch.long).to(args.device)
    trg = torch.tensor([[tokenizer.bos_id]]).to(args.device)  # 1 x seq_len

    # Prediction
    with torch.no_grad():
        opt = model(src, trg)  # 1 x seq_len x vocab_size
        predict = torch.argmax(opt, dim=-1)  # 1 x seq_len

        while predict.size(1) < args.max_len:
            # concat predict to trg
            
            trg = torch.cat([torch.tensor([[tokenizer.bos_id]]).to(args.device) , predict], dim=-1)
            opt = model(src, trg)
            predict = torch.argmax(opt, dim=-1)

            # whether end
            if predict[0][-1] == tokenizer.eos_id:
                break

    predict = predict.cpu().numpy().tolist()
    predict = tokenizer.decode(predict)
    return ''.join(predict)



def main():
    args = set_args()

    # Cuda Env Setting
    args.cuda = torch.cuda.is_available()
    device = 'cuda' if args.cuda else 'cpu'
    print('Using :', device, 'Device: ', args.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.device = device  # This is for the functions defined above, the args would transfer to the functions later

    # Load Tokenizer
    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.load(args.tokenizer_pth)

    # Read Modle Config
    config = json2dict(args.model_config)

    args.max_len = config['max_seq_len']

    model = Transformer(embed_dim = config['embed_dim'], 
                    s_vocab_size = bpe_tokenizer.vocab_size, 
                    t_vocab_size = bpe_tokenizer.vocab_size, 
                    max_seq_len = args.max_len, 
                    num_layers = config['num_layers'], 
                    expansion_factor = config['expansion_factor'],
                    n_heads = config['num_heads'])
    
    # Load Model
    model.load_state_dict(torch.load(args.ptm_pth))
    model = model.to(device)
    model.eval()

    print('Load Pretrained Model from', args.ptm_pth)

    # Start Inference
    print('Start Inference')
    while True:
        usr_input = input('Input: ')
        output = seq2seq_inference(usr_input, bpe_tokenizer, model, args)
        print('Output: ', output)


if __name__ == '__main__':
    main()