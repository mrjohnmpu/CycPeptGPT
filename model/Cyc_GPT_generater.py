import os
import sys

# 将当前文件的父目录（model）的父目录（CYC_GPT）添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import time
import torch
import argparse
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2 import GPT2LMHeadModel
from transformers import BertTokenizer
import torch.nn.functional as F
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)


def seed_everything(seed: int):
    """
    固定所有随机种子以确保结果可复现。
    """
    # 1. 固定 Python 内置的 random 模块
    random.seed(seed)

    # 2. 固定 os.environ['PYTHONHASHSEED']
    #    注意: 这需要你在启动 Python 脚本 *之前* 就设置好
    #    e.g. export PYTHONHASHSEED=42
    #    在脚本内部设置可能不会生效
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 3. 固定 NumPy
    np.random.seed(seed)

    # 4. 固定 PyTorch (CPU)
    torch.manual_seed(seed)

    # 5. 固定 PyTorch (GPU, if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前 GPU 设置种子
        torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置种子 (在 DDP 中很重要)

    # 6. 固定 cuDNN 的行为
    #    这将强制 cuDNN 使用确定性的（但可能更慢的）算法
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # print(f"--- 所有随机种子已固定为: {seed} ---")


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
    # parser.add_argument('--best_model_dir', default="../output/best_model_with_mask/checkpoint-3960", type=str,
    #                     help='Trainer 将在此保存 checkpoint')
    parser.add_argument('--best_model_dir', default="../output/best_model", type=str,
                        help='Trainer 将在此保存 checkpoint')
    parser.add_argument('--train_raw_path', default='train_raw_data.txt', type=str, help='')
    parser.add_argument('--eval_raw_path', default='test_raw_data.txt', type=str, help='')
    parser.add_argument('--batch_size', default=32, type=int, required=False,
                        help='per_device batch size (每个 GPU 的 batch size)')
    parser.add_argument('--accumulation_steps', default=4, type=int, required=False)
    parser.add_argument('--epochs', default=1001, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=10000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=5e-5, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)  # Trainer 会自动使用
    parser.add_argument('--log_step', default=10, type=int, required=False, help='logging steps')
    parser.add_argument('--patience', default=10, type=int, required=False, help='early stopping patience')
    parser.add_argument('--max_len', default=576, type=int, required=False, help='The max length for each sequence')
    return parser.parse_args()


def decode(matrix):
    chars = []
    for i in matrix:
        if i == '<eos>': break
        chars.append(i.upper())
    seq = "".join(chars)
    return seq


def predict(model, tokenizer, batch_size, text=""):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    model.eval()
    time1 = time.time()
    max_length = 576
    input_ids = []
    input_ids.extend(tokenizer.encode(text))
    input_ids = input_ids[0]

    input_tensor = torch.zeros(batch_size, 1).long()

    input_tensor[:] = input_ids

    Seq_list = []

    finished = torch.zeros(batch_size, 1).byte().to(device)

    for i in range(max_length):
        # input_tensor = torch.tensor([input_ids])
        inputs = {"input_ids": input_tensor.to(device)}
        outputs = model(**inputs)
        logits = outputs.logits

        logits = F.softmax(logits[:, -1, :])

        last_token_id = torch.multinomial(logits, 1)
        # .detach().to('cpu').numpy()
        EOS_sampled = (last_token_id == tokenizer.eos_token_id)
        finished = torch.ge(finished + EOS_sampled, 1)
        if torch.prod(finished) == 1:
            print('End')
            break

        last_token = tokenizer.convert_ids_to_tokens(last_token_id)

        input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)

        Seq_list.append(last_token)
    # print(Seq_list)
    Seq_list = np.array(Seq_list).T

    print("time cost: {}".format(time.time() - time1))
    return Seq_list
    # print(Seq_list)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    seed_everything(42)
    args = setup_args()

    # tokenizer = PreTrainedTokenizerFast.from_pretrained("./MolGPT_pretrained-by-ZINC15")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")
    tokenizer.model_max_length = 576

    model = GPT2LMHeadModel.from_pretrained(args.best_model_dir)

    output = []
    Seq_all = []
    for i in range(10):
        Seq_list = predict(model, tokenizer, batch_size=32)

        Seq_all.extend(Seq_list)
    for j in Seq_all:
        output.append(decode(j))

    output = pd.DataFrame(output)

    output.to_csv('../output/generate_cyc_seq.csv', index=False, header=False, sep=' ')
