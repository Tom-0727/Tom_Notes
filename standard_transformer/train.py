import os
import sys
import time
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
from torch.utils.data import DataLoader


# Set Arguments
def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='0', type=str, required=False,
                        help='GPU Setting')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Train without GPU')
    
    parser.add_argument('--load_ptm', action='store_true',
                        help='Load the Pretrained Model')
    parser.add_argument('--train', action='store_true', 
                        help='Train the model')
    parser.add_argument('--validate', action='store_true',
                        help='Validate the model')
    parser.add_argument('--validate_steps', default=1, type=int, required=False,
                        help='Steps of epoch to validate model')

    parser.add_argument('--ptm_pth', default='./models/', type=str, required=False,
                        help='Path to pretrained model')
    parser.add_argument('--model_config', default='./config/standard_transformer.json', type=str, required=False,
                        help='Path to model config')
    parser.add_argument('--save_steps', default=10, type=int, required=False,
                        help='Steps of epoch to save model')
    parser.add_argument('--save_model_path', default='./models/', type=str, required=False,
                        help='Path to save model')
    
    parser.add_argument('--tokenizer_pth', default='./iwslt2013_tokenizer.pkl', type=str, required=False,
                        help='Path to the dataset for training')
    parser.add_argument('--data_dir', default='./data/iwsltenvi/', type=str, required=False,
                        help='Path to the dataset for training')

    parser.add_argument('--epochs', default=20, type=int, required=False,
                        help='Epochs to train')
    parser.add_argument('--batch_size', default=64, type=int, required=False,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help="parameter of DataLoader, used to set multi-thread, 0 is going without multi-thread")

    parser.add_argument('--warmup_steps', type=int, default=5000,
                        help='Steps to warm up to max learning rate')
    parser.add_argument('--lr', default=2.6e-5, type=float, required=False,
                        help='Max Learning rate, it takes warmup steps to reach this rate')
    parser.add_argument('--eps', default=1.0e-09, type=float, required=False,
                        help='AdamW optimizer epsilon rate')
    parser.add_argument('--info', action = 'store_true',
                        help='Print the information of the model')

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


# Load Dataset
def load_dataset(args):
    train_data, test_data = load_iwsltenvi_data(data_dir=args.data_dir)
    train_dataset = IWSLTDataset(train_data)
    test_dataset = IWSLTDataset(test_data)

    return train_dataset, test_dataset


# Dynamic Padding with max_len
def dynamic_padding(batch, 
                    pad_id: int = 0, 
                    eos_id: int = 2,
                    max_seq_len: int = 256):
    # Find the longest sequence in the batch
    max_len = min(max_seq_len, max([len(x) for x in batch]))
    
    for i in range(len(batch)):
        # Truncate the sequence > max_len
        if len(batch[i]) > max_seq_len: 
            batch[i] = batch[i][:max_seq_len-1]

            # Turn the last token to eos_id
            batch[i].append(eos_id)

        # Fill the rest of the sequence with padding
        batch[i] = batch[i] + [pad_id] * (max_len - len(batch[i]))


    return batch


# Count the Accuracy
def acc_count(logits, trg, pad_id):
    # logits: batch_size, seq_len, vocab_size
    # trg: batch_size, seq_len
    prediction = torch.argmax(logits, dim=-1)  # batch_size, seq_len
    correct_num = torch.sum(prediction == trg).item()
    total_num = torch.sum(trg != pad_id).item()

    return correct_num, total_num


# Data preprocess
def data_preprocess(tokenizer, src, trg, args):
    data_pre_start = time.time()
    # List[str] -> List[int]
    tokenization_start = time.time()
    src = tokenizer.encode(src)
    trg = tokenizer.encode(trg)
    tokenization_end = time.time()

    # Dynamic Padding
    src = dynamic_padding(src, pad_id=tokenizer.pad_id, max_seq_len=args.max_len)
    trg = dynamic_padding(trg, pad_id=tokenizer.pad_id, max_seq_len=args.max_len)

    # List[int] -> Tensor with dtype=torch.long & to GPU
    src = torch.tensor(src, dtype=torch.long)
    trg = torch.tensor(trg, dtype=torch.long)

    # trg shifted right
    shifted_trg = torch.zeros_like(trg, dtype=torch.long)
    shifted_trg[:, 1:] = trg[:, 1:]
    shifted_trg[:, 0] = tokenizer.bos_id

    if args.info:
        print(f'Tokenization Time: {tokenization_end - tokenization_start:.2f}', end=' ', flush=True)

    return src, trg, shifted_trg


# The Training Process in one Epoch
def train_epoch(model, train_dataloader, tokenizer, criterion, optimizer, scheduler, epoch, args):
    model.train()

    epoch_start_time = time.time()
    total_loss = 0
    epoch_correct_num, epoch_total_num = 0, 0

    for batch_idx, (src, trg) in enumerate(train_dataloader):
        try:
            # ============== Prepare Data ============== #
            src, trg, shifted_trg = data_preprocess(tokenizer, src, trg, args)

            # To device
            src = src.to(args.device)
            trg = trg.to(args.device)
            shifted_trg = shifted_trg.to(args.device)
            
            # =============== Forward ================== #
            forward_start = time.time()
            opt = model(src, shifted_trg)  # batch_size, seq_len, vocab_size
            forward_end = time.time()

            # =============== Backward ================= #
            # Calculate Accuracy
            correct_num, total_num = acc_count(opt, trg, tokenizer.pad_id)

            # For epoch
            epoch_correct_num += correct_num
            epoch_total_num += total_num

            # Calculate Loss
            loss = criterion(opt.view(-1, opt.size(-1)), trg.view(-1))
            loss = loss.mean()
            total_loss += loss.item()

            loss.backward()
            

            # ================= Update =================== #
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # ============== Print Info ================= #
            if args.info:
                print(f'Forward Time: {forward_end - forward_start:.2f}')
            
            del src, trg, opt

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
        
        # print('/', end='', flush=True)
            
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_acc = epoch_correct_num / epoch_total_num
    epoch_time = time.time() - epoch_start_time

    broadcast_info = f'Training_epoch{epoch}: Loss {epoch_mean_loss:.4f} | Acc {epoch_acc:.4f} | Time {epoch_time:.2f}'
    print(broadcast_info)

    return epoch_mean_loss


def validate_epoch(model, test_dataloader, tokenizer, criterion, epoch, args):
    model.eval()

    total_loss = 0
    epoch_correct_num = 0
    epoch_total_num = 0
    epoch_start_time = time.time()

    try:
        with torch.no_grad():
            for batch_idx, (src, trg) in enumerate(test_dataloader):
                # ============== Prepare Data ============== #
                src, trg, shifted_trg = data_preprocess(tokenizer, src, trg, args)

                # To device
                src = src.to(args.device)
                trg = trg.to(args.device)
                shifted_trg = shifted_trg.to(args.device)
                
                opt = model(src, shifted_trg)  # batch_size, seq_len, vocab_size

                correct_num, total_num = acc_count(opt, trg, tokenizer.pad_id)
                epoch_correct_num += correct_num
                epoch_total_num += total_num

                # Calculate Loss
                loss = criterion(opt.view(-1, opt.size(-1)), trg.view(-1))
                loss = loss.mean()
                total_loss += loss.item()
            
            epoch_mean_loss = total_loss / len(test_dataloader)
            epoch_acc = epoch_correct_num / epoch_total_num
            epoch_time = time.time() - epoch_start_time

            broadcast_info = f'Validating_epoch{epoch}: Loss {epoch_mean_loss:.4f} | Acc {epoch_acc:.4f} | Time {epoch_time:.2f}'
            print(broadcast_info)

    except RuntimeError as exception:
        if "out of memory" in str(exception):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise exception
            



# The Train Process
def train(model, tokenizer, train_dataset, test_dataset, args):
    if args.train:
        train_loader = DataLoader(
            train_dataset,
            batch_size = args.batch_size,
            shuffle = True,
            num_workers = args.num_workers,
            pin_memory = True
        )

    if args.validate:
        test_loader = DataLoader(
            test_dataset,
            batch_size = args.batch_size,
            shuffle = False,
            num_workers = args.num_workers,
            pin_memory = True
        )

    # Training Modules
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: epoch/ args.warmup_steps if epoch < args.warmup_steps else (1.0 - epoch / args.epochs) if epoch < args.epochs else 0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # Side Information
    train_losses = []

    for epoch in range(args.epochs):
        # ======== Train ======= #
        if args.train:
            print(f'Epoch{epoch} start training...')
            train_loss = train_epoch(model, train_loader, tokenizer, criterion, optimizer, scheduler, epoch, args)
            train_losses.append(train_loss)

        # ======== Validate ====== #
        if args.validate and (epoch % args.validate_steps == 0):
            print(f'Epoch{epoch} start validating...')
            validate_epoch(model, test_loader, tokenizer, criterion, epoch, args)


        # ======= SAVE ======= #
        if epoch % args.save_steps == 0 and epoch != 0:
            os.makedirs(args.save_model_path, exist_ok=True)
            torch.save(model.state_dict(), f'{args.save_model_path}epoch{epoch}.pth')
            print(f'Save model at epoch{epoch}!')

    return train_losses


def main():
    args = set_args()

    args.cuda = torch.cuda.is_available() and (not args.no_cuda)
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
    
    if args.load_ptm:
        model.load_state_dict(torch.load(args.ptm_pth))
        print('Load Pretrained Model from', args.ptm_pth)
    
    # Move to GPU
    model = model.to(device)

    # Calculate the Number of parameters
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print(f'Number of parameters: {num_parameters}')

    train_dataset, test_dataset = load_dataset(args)

    train_losses = train(model, bpe_tokenizer, train_dataset, test_dataset, args)

    with open('./train_history.txt', 'a') as f:
        f.write('Train Losses variate: ' + str(train_losses) + '\n')


if __name__ == '__main__':
    main()