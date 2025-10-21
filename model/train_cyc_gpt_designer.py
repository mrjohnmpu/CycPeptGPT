import os
import sys
# 将当前文件的父目录（model）的父目录（CYC_GPT）添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import time
import csv
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.optim import AdamW, Adam
from transformers import get_scheduler, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader
# from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer
from torch.nn import CrossEntropyLoss
from early_stop.pytorchtools import EarlyStopping
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __getitem__(self, index):
        input_ids = self.data_list[index]
        return input_ids

    def __len__(self):
        return len(self.data_list)


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="", type=str, help='')
    parser.add_argument('--vocab_path', default="", type=str, help='')
    parser.add_argument('--best_model_dir', default="../output/best_model", type=str, help='')
    parser.add_argument('--best_ckpt_path', default="../output/best_checkpoint.pt", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.txt', type=str, help='')
    parser.add_argument('--eval_raw_path', default='test_raw_data.txt', type=str, help='')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='batch size')
    parser.add_argument('--epochs', default=1001, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=1000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=5e-5, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False, help='print log steps')
    return parser.parse_args()


def calculate_loss_and_accuracy(outputs, labels, device):
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(device)

    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    _, preds = shift_logits.max(dim=-1)
    not_ignore = shift_labels.ne(tokenizer.pad_token_id)
    num_targets = not_ignore.long().sum().item()

    correct = (shift_labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    return loss, accuracy


def collate_fn(batch):
    input_ids = []
    input_lens_list = [len(w) for w in batch]
    max_input_len = max(input_lens_list)
    for btc_idx in range(len(batch)):
        input_len = len(batch[btc_idx])
        input_ids.append(batch[btc_idx])
        input_ids[btc_idx].extend([tokenizer.pad_token_id] * (max_input_len - input_len))
    return torch.tensor(input_ids, dtype=torch.long)


def data_loader(args, train_data_path, tokenizer, shuffle):
    data_list = []
    eval_data_list = []

    df = pd.read_csv(train_data_path)
    print("数据总行数:{}".format(len(df)))

    train_df, eval_df = train_test_split(df, test_size=0.1, shuffle=shuffle, random_state=42)
    print(f"训练集大小: {len(train_df)}, 验证集大小: {len(eval_df)}")

    max_seq_length = tokenizer.model_max_length
    train_smiles_list = train_df['SMILES'].tolist()
    eval_smiles_list = eval_df['SMILES'].tolist()

    for smiles_string in tqdm(train_smiles_list, desc="Tokenizing training data"):
        tokenized_ids = tokenizer.encode(
            smiles_string, 
            truncation=True, 
            max_length=max_seq_length
        )
        data_list.append(tokenized_ids)
        # data_list.append(tokenizer.encode(data_i, padding="max_length", truncation=True, max_length=34,
        #                                   return_special_tokens_mask=True, ))

    dataset = MyDataset(data_list)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)

    for smiles_string in tqdm(eval_smiles_list, desc="Tokenizing eval data"):
        tokenized_ids = tokenizer.encode(
            smiles_string, 
            truncation=True, 
            max_length=max_seq_length
        )
        eval_data_list.append(tokenized_ids)

    eval_dataset = MyDataset(eval_data_list)
    eval_dataloader = DataLoader(dataset=eval_dataset,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)

    return dataloader,eval_dataloader

def train(args, model, dataloader,eval_dataloader):
    num_training_steps = args.epochs * len(dataloader)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("using device : {}".format(device))
    model.to(device)
    model.train()
    batch_steps = 0
    early_stopping = EarlyStopping(patience=5, verbose=False)

    for epoch in range(args.epochs):
        epoch_loss_list = []
        print("\n")
        print("***********")
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        print("***********")
        print("\n")
        for batch in dataloader:
            batch_steps += 1
            inputs = {"input_ids": batch.to(device)}
            outputs = model(**inputs, labels=batch.to(device))
            loss, acc = calculate_loss_and_accuracy(outputs, batch.to(device), device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if batch_steps % args.log_step == 0:
                print("train epoch {}/{}, batch {}/{}, loss {}, accuracy {}".format(
                    epoch, args.epochs,
                    batch_steps,
                    num_training_steps,
                    loss, acc,
                ))

            epoch_loss_list.append(loss.cpu().detach().numpy())
        epoch_loss = evaluate(model,eval_dataloader)
        early_stopping(epoch_loss, model, optimizer, lr_scheduler, epoch, args.best_model_dir, args.best_ckpt_path)

        # model_to_save = model.module if hasattr(model, 'module') else model
        # model_to_save.save_pretrained(args.final_model_path)


def evaluate(model, dataloader):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = GPT2LMHeadModel.from_pretrained(args.save_model_path)

    model.to(device)
    model.eval()
    loss_list, acc_list = [], []
    batch_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch_steps += 1
            inputs = {"input_ids": batch.to(device)}
            outputs = model(**inputs, labels=batch.to(device))
            loss, acc = calculate_loss_and_accuracy(outputs, batch.to(device), device)
            loss_list.append(float(loss))
            acc_list.append(float(acc))


            # print("eval batch {}/{}, loss {}, accuracy {}".format(
            #     batch_steps,
            #     len(dataloader),
            #     loss, acc))

    epoch_loss = np.mean(loss_list)



    print("loss: {},".format(np.mean(loss_list)),
          "accuracy: {}.".format(np.mean(acc_list)))
    return epoch_loss




def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    args = setup_args()
    # args.model_path, args.vocab_path = '', './my_token/vocab.txt'
    args.train_raw_path = '../data/restored_train.csv'


    # tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    CHEMFORMER_TOKENIZER_FILE = "model/bart_vocab.json"
    tokenizer = PreTrainedTokenizerFast.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")

    # tokenizer.pad_token = "<PAD>"
    # tokenizer.bos_token = "^"  
    # tokenizer.eos_token = "&"
    # tokenizer.mask_token = "<MASK>"
    # tokenizer.unk_token = "?"
    # tokenizer.sep_token = "<SEP>"
    tokenizer.model_max_length = 576
    tokenizer.bos_token = "<bos>"
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token = "<pad>"


    model_config = GPT2Config(
        architectures=["GPT2LMHeadModel"],  # pretrain的时候用来预加载模型
        model_type="GPT2LMHeadModel",  # 定义模型类型，导出给`AutoConfig`用，如果要上传到hub请必填
        vocab_size=tokenizer.vocab_size,
        n_positions=576,
        n_ctx=576,
        n_embd=768,
        n_layer=12,
        n_head=8,
        bos_token_id=tokenizer.bos_token_id,  # 前面构建的tokenizer的 PAD ID
        eos_token_id=tokenizer.eos_token_id,  # 前面构建的tokenizer的 PAD ID
        pad_token_id=tokenizer.pad_token_id,  # 前面构建的tokenizer的 PAD ID
        # mask_token_id=tokenizer.mask_token_id,  # 前面构建的tokenizer的 PAD ID

        task_specific_params={
            "text-generation": {
                "do_sample": True,
                "max_length": 576
            }
        }
    )
    model = GPT2LMHeadModel(config=model_config)

    train_dataloader,eval_dataloader = data_loader(args, args.train_raw_path, tokenizer=tokenizer, shuffle=True)
    train(args, model, train_dataloader, eval_dataloader)