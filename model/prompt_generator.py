import os
import sys
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from soft_prompt_embedding import SoftEmbedding
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_scheduler
)
import random
from safetensors.torch import load_file

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="", type=str, help='')
    parser.add_argument('--vocab_path', default="/vocab.txt", type=str, help='')
    parser.add_argument('--batch_size', default=128, type=int, required=False, help='batch size')
    parser.add_argument('--top_k', default=10, type=int, required=False, help='print log steps')
    parser.add_argument('--top_p', default=1.0, type=float, required=False, help='print log steps')
    return parser.parse_args()


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


def top_k_top_p_filtering(
        logits: torch.FloatTensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_p = float(top_p)
    if top_k > 0:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if 0 < top_p <= 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits

def decode(matrix):
    chars = []
    for i in matrix:
        if i == '[SEP]':
            break
        chars.append(i.upper())
    seq = "".join(chars)
    return seq

def predict(args,model, tokenizer, batch_size, text=""):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model, _ = load_model(args.save_model_path, args.vocab_path)

    model.to(device)
    model.eval()
    time1 = time.time()
    max_length = 576

    input_ids = list(range(100, 100 + 10))

    input_ids.extend(tokenizer.encode(text))

    input_ids = input_ids[:11]

    input_tensor = torch.zeros(batch_size, 11).long()

    for index,i in enumerate(input_ids):
        input_tensor[:,index] = input_ids[index]

        # input_tensor[:,11] = input_ids

    Seq_list = []

    finished = torch.zeros(batch_size,1).byte().to(device)

    for i in range(max_length):
        # input_tensor = torch.tensor([input_ids])
        inputs = {"input_ids": input_tensor.to(device)}
        try:
            outputs = model(**inputs)
        except Exception as e:
            print(e)
        logits = outputs.logits

        # if topk
        logits = top_k_top_p_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        logits = F.softmax(logits[:,-1,:])
        last_token_id = torch.multinomial(logits, 1)
        # .detach().to('cpu').numpy()
        EOS_sampled = (last_token_id == tokenizer.sep_token_id)
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
    # args.model_path, args.vocab_path = '', '../voc/vocab.txt'
    # tokenizer = PreTrainedTokenizerFast.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./MolGPT_pretrained-by-ZINC15")
    prompt_model_load = load_file("../output/best_model_prompt/best_model_epoch_61/model.safetensors")
    model= GPT2LMHeadModel.from_pretrained('../output/best_model_prompt/best_model_epoch_61/')
    s_wte = SoftEmbedding(model.get_input_embeddings(),
                          n_tokens=10,
                          initialize_from_vocab=True)
    s_wte.learned_embedding.data = prompt_model_load['transformer.wte.learned_embedding']
    s_wte.wte.weight.data = prompt_model_load['transformer.wte.wte.weight']
    del prompt_model_load
    model.set_input_embeddings(s_wte)

    output = []
    Seq_all = []
    if not os.path.exists('../output/generate/prompt_generate/'):
        os.makedirs('../output/generate/prompt_generate/')
    for i in range(500):
        print(i)
        Seq_list = predict(args,model,tokenizer,batch_size=64)
        batch_decoded = []
        for seq_tokens in Seq_list:
            batch_decoded.append(decode(seq_tokens))
        df_batch = pd.DataFrame(batch_decoded)
        current_mode = 'w' if i == 0 else 'a'
        df_batch.to_csv('../output/generate/prompt_generate/cyc_prompt_topk_500_64.csv',
                        mode=current_mode,  # 关键：动态切换模式
                        index=False,
                        header=False,
                        sep=' ')
        Seq_all.extend(Seq_list)

