# **************************************************************
# Based on Pytorch Lightning
# **************************************************************
import os
import sys
import time
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

# 【修改 1】导入你在训练时定义的 LightningModule
# 假设你的 LightningModule 定义在 prompt_lightning_module.py 文件中
from prompt_lightning_module import ContrastivePrefixModule

import random


def setup_args():
    parser = argparse.ArgumentParser()
    # 【修改 2】将 model_path 参数改为 ckpt_path，指向 .ckpt 文件
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the PyTorch Lightning .ckpt file')
    parser.add_argument('--tokenizer_path', default="./MolGPT_pretrained-by-ZINC15", type=str,
                        help='Path to the tokenizer folder')

    parser.add_argument('--batch_size', default=5, type=int, required=False, help='batch size')
    parser.add_argument('--top_k', default=10, type=int, required=False, help='top k filtering')
    parser.add_argument('--top_p', default=1.0, type=float, required=False, help='top p filtering')

    # 这些参数如果是从 checkpoint 加载，通常会被覆盖，但保留着作为默认值
    parser.add_argument('--n_prefixes', default=2, type=int, help='Training used 2 prefixes')
    parser.add_argument('--n_tokens', default=10, type=int, help='Training used 10 tokens')
    parser.add_argument('--target_prefix_idx', default=1, type=int, help='0 for linear, 1 for head_to_tail')
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def top_k_top_p_filtering(
        logits: torch.FloatTensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    top_p = float(top_p)
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if 0 < top_p <= 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


def decode(matrix):
    chars = []
    for i in matrix:
        if i == '[SEP]' or i == '<eos>':
            break
        chars.append(i.upper())
    seq = "".join(chars)
    return seq


def predict(args, model, tokenizer, batch_size):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    time1 = time.time()

    # 这里的 n_tokens 需要从模型配置中获取，或者使用 args
    # 如果模型是重参数化后的，SoftEmbedding 会有 n_tokens 属性
    n_tokens = args.n_tokens
    if hasattr(model.transformer.wte, 'n_tokens'):
        n_tokens = model.transformer.wte.n_tokens

    max_length = 576 - n_tokens

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    input_tensor = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

    # 构造前缀索引 (Target Prefix)
    prefix_indices = torch.full((batch_size,), args.target_prefix_idx, dtype=torch.long, device=device)

    Seq_list = []
    finished = torch.zeros(batch_size, 1).byte().to(device)

    with torch.no_grad():
        for i in range(max_length):
            # 手动计算 Embedding (利用 SoftEmbedding)
            inputs_embeds = model.transformer.wte(input_tensor, prefix_indices=prefix_indices)

            outputs = model(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[:, -1, :]

            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            EOS_sampled = (next_token == tokenizer.eos_token_id)
            finished = torch.ge(finished + EOS_sampled, 1)

            token_str_list = [tokenizer.decode(t) for t in next_token]
            Seq_list.append(token_str_list)

            if torch.prod(finished) == 1:
                break

            input_tensor = torch.cat((input_tensor, next_token), dim=1)

    Seq_list = np.array(Seq_list).T
    print("Generation time cost: {:.4f}s".format(time.time() - time1))
    return Seq_list


if __name__ == '__main__':
    seed_everything(42)
    args = setup_args()

    # 1. 加载 Tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. 【关键步骤】从 .ckpt 文件加载 Lightning Module
    print(f"Loading Lightning model from checkpoint: {args.ckpt_path}...")

    # 注意：这里需要传入 tokenizer，因为你的 LightningModule __init__ 需要它
    # 如果 __init__ 参数变了，这里也需要调整
    # strict=False 可以忽略一些不匹配的键（例如 loss 记录等），通常设为 True 比较安全
    lightning_model = ContrastivePrefixModule.load_from_checkpoint(
        args.ckpt_path,
        tokenizer=tokenizer
    )

    # 3. 【关键步骤】执行推理前的准备（重参数化）
    # Lightning 保存 checkpoint 时，保存的是训练状态（包含 MLP 参数）。
    # 我们现在要生成，必须把 MLP "烘焙" 进 Embedding 里。
    # 检查是否需要重参数化（即 learned_embedding 是否为 None）
    if lightning_model.model.transformer.wte.learned_embedding is None:
        print("Performing reparameterization for inference...")
        lightning_model.model.transformer.wte.reparameterize()
    else:
        print("Model already reparameterized or in standard mode.")

    # 4. 提取底层的 GPT2 模型用于生成
    # lightning_model.model 就是我们在 LightningModule 里定义的 self.model (GPT2LMHeadModel)
    model = lightning_model.model

    # 确保模型在评估模式
    model.eval()
    model.cuda()  # 移动到 GPU

    # 5. 开始生成
    output_dir = '../output/generate/prompt_generate/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, 'cyc_prompt_generated_from_ckpt.csv')

    total_generated = 0
    target_num = 5  # 或者从 args 获取

    while total_generated < target_num:
        current_batch = min(args.batch_size, target_num - total_generated)
        print(f"Generating batch of {current_batch}...")

        Seq_list = predict(args, model, tokenizer, batch_size=current_batch)

        batch_decoded = []
        for seq_tokens in Seq_list:
            batch_decoded.append(decode(seq_tokens))

        df_batch = pd.DataFrame(batch_decoded)

        current_mode = 'w' if total_generated == 0 else 'a'
        header = False
        df_batch.to_csv(output_file, mode=current_mode, index=False, header=header, sep=' ')

        total_generated += current_batch
        print(f"Total generated: {total_generated}")

    print(f"Generation complete! Results saved to {output_file}")



# import os
# import sys
# import time
# import torch
# import argparse
# import numpy as np
# import pandas as pd
# import torch.nn.functional as F
# from soft_prompt_embedding import SoftEmbedding
# from transformers import (
#     GPT2Config,
#     GPT2LMHeadModel,
#     PreTrainedTokenizerFast,
#     Trainer,
#     TrainingArguments,
#     EarlyStoppingCallback,
#     get_scheduler
# )
# import random
# from safetensors.torch import load_file
#
# def setup_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', default="", type=str, help='')
#     parser.add_argument('--vocab_path', default="/vocab.txt", type=str, help='')
#     parser.add_argument('--batch_size', default=128, type=int, required=False, help='batch size')
#     parser.add_argument('--top_k', default=10, type=int, required=False, help='print log steps')
#     parser.add_argument('--top_p', default=1.0, type=float, required=False, help='print log steps')
#     return parser.parse_args()
#
#
# def seed_everything(seed: int):
#     """
#     固定所有随机种子以确保结果可复现。
#     """
#     # 1. 固定 Python 内置的 random 模块
#     random.seed(seed)
#
#     # 2. 固定 os.environ['PYTHONHASHSEED']
#     #    注意: 这需要你在启动 Python 脚本 *之前* 就设置好
#     #    e.g. export PYTHONHASHSEED=42
#     #    在脚本内部设置可能不会生效
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#     # 3. 固定 NumPy
#     np.random.seed(seed)
#
#     # 4. 固定 PyTorch (CPU)
#     torch.manual_seed(seed)
#
#     # 5. 固定 PyTorch (GPU, if available)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)  # 为当前 GPU 设置种子
#         torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置种子 (在 DDP 中很重要)
#
#     # 6. 固定 cuDNN 的行为
#     #    这将强制 cuDNN 使用确定性的（但可能更慢的）算法
#     torch.backends.cudnn.deterministic = True
#     # torch.backends.cudnn.benchmark = False
#
#
# def top_k_top_p_filtering(
#         logits: torch.FloatTensor,
#         top_k: int = 0,
#         top_p: float = 1.0,
#         filter_value: float = -float("Inf"),
#         min_tokens_to_keep: int = 1,
# ) -> torch.FloatTensor:
#     """
#     Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
#     Args:
#         logits: logits distribution shape (batch size, vocabulary size)
#         top_k (`int`, *optional*, defaults to 0):
#             If > 0, only keep the top k tokens with highest probability (top-k filtering)
#         top_p (`float`, *optional*, defaults to 1.0):
#             If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
#             filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
#         min_tokens_to_keep (`int`, *optional*, defaults to 1):
#             Minimumber of tokens we keep per batch example in the output.
#     From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
#     """
#     top_p = float(top_p)
#     if top_k > 0:
#         if not isinstance(top_k, int) or top_k <= 0:
#             raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
#         top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
#         # Remove all tokens with a probability less than the last token of the top-k
#         indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#         logits = logits.masked_fill(indices_to_remove, filter_value)
#
#     if 0 < top_p <= 1.0:
#         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#         cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
#
#         # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
#         sorted_indices_to_remove = cumulative_probs > top_p
#         if min_tokens_to_keep > 1:
#             # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
#             sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
#         # Shift the indices to the right to keep also the first token above the threshold
#         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#         sorted_indices_to_remove[..., 0] = 0
#         # scatter sorted tensors to original indexing
#         indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
#         logits = logits.masked_fill(indices_to_remove, filter_value)
#
#     return logits
#
# def decode(matrix):
#     chars = []
#     for i in matrix:
#         if i == '[SEP]':
#             break
#         chars.append(i.upper())
#     seq = "".join(chars)
#     return seq
#
# def predict(args,model, tokenizer, batch_size, text=""):
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     # model, _ = load_model(args.save_model_path, args.vocab_path)
#
#     model.to(device)
#     model.eval()
#     time1 = time.time()
#     max_length = 576
#
#     input_ids = list(range(100, 100 + 10))
#
#     input_ids.extend(tokenizer.encode(text))
#
#     input_ids = input_ids[:11]
#
#     input_tensor = torch.zeros(batch_size, 11).long()
#
#     for index,i in enumerate(input_ids):
#         input_tensor[:,index] = input_ids[index]
#
#         # input_tensor[:,11] = input_ids
#
#     Seq_list = []
#
#     finished = torch.zeros(batch_size,1).byte().to(device)
#
#     for i in range(max_length):
#         # input_tensor = torch.tensor([input_ids])
#         inputs = {"input_ids": input_tensor.to(device)}
#         try:
#             outputs = model(**inputs)
#         except Exception as e:
#             print(e)
#         logits = outputs.logits
#
#         # if topk
#         logits = top_k_top_p_filtering(logits, top_k=args.top_k, top_p=args.top_p)
#         logits = F.softmax(logits[:,-1,:])
#         last_token_id = torch.multinomial(logits, 1)
#         # .detach().to('cpu').numpy()
#         EOS_sampled = (last_token_id == tokenizer.sep_token_id)
#         finished = torch.ge(finished + EOS_sampled, 1)
#         if torch.prod(finished) == 1:
#             print('End')
#             break
#
#         last_token = tokenizer.convert_ids_to_tokens(last_token_id)
#         input_tensor = torch.cat((input_tensor, last_token_id.detach().to('cpu')), 1)
#
#
#
#         Seq_list.append(last_token)
#     # print(Seq_list)
#     Seq_list = np.array(Seq_list).T
#
#
#     print("time cost: {}".format(time.time() - time1))
#     return Seq_list
#     # print(Seq_list)
#
#
# def get_parameter_number(model):
#     total_num = sum(p.numel() for p in model.parameters())
#     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num}
#
#
# if __name__ == '__main__':
#     seed_everything(42)
#     args = setup_args()
#     # args.model_path, args.vocab_path = '', '../voc/vocab.txt'
#     # tokenizer = PreTrainedTokenizerFast.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")
#     tokenizer = PreTrainedTokenizerFast.from_pretrained("./MolGPT_pretrained-by-ZINC15")
#     prompt_model_load = load_file("../output/best_model_prompt/best_model_epoch_61/model.safetensors")
#     model= GPT2LMHeadModel.from_pretrained('../output/best_model_prompt/best_model_epoch_61/')
#     s_wte = SoftEmbedding(model.get_input_embeddings(),
#                           n_tokens=10,
#                           initialize_from_vocab=True)
#     s_wte.learned_embedding.data = prompt_model_load['transformer.wte.learned_embedding']
#     s_wte.wte.weight.data = prompt_model_load['transformer.wte.wte.weight']
#     del prompt_model_load
#     model.set_input_embeddings(s_wte)
#
#     output = []
#     Seq_all = []
#     if not os.path.exists('../output/generate/prompt_generate/'):
#         os.makedirs('../output/generate/prompt_generate/')
#     for i in range(500):
#         print(i)
#         Seq_list = predict(args,model,tokenizer,batch_size=64)
#         batch_decoded = []
#         for seq_tokens in Seq_list:
#             batch_decoded.append(decode(seq_tokens))
#         df_batch = pd.DataFrame(batch_decoded)
#         current_mode = 'w' if i == 0 else 'a'
#         df_batch.to_csv('../output/generate/prompt_generate/cyc_prompt_topk_500_64.csv',
#                         mode=current_mode,  # 关键：动态切换模式
#                         index=False,
#                         header=False,
#                         sep=' ')
#         Seq_all.extend(Seq_list)
#

# **************************************************************
# 以下是完全修改的代码
# **************************************************************
# import os
# import sys
# import time
# import torch
# import argparse
# import numpy as np
# import pandas as pd
# import torch.nn.functional as F
# # 确保 soft_prompt_embedding.py 是修改后支持 n_prefixes 的版本
# from soft_prompt_embedding import SoftEmbedding
# from transformers import (
#     GPT2Config,
#     GPT2LMHeadModel,
#     PreTrainedTokenizerFast,
# )
# import random
# from safetensors.torch import load_file
#
#
# def setup_args():
#     parser = argparse.ArgumentParser()
#     # 你的 checkpoint 路径 (包含 config.json 和 model.safetensors/pytorch_model.bin)
#     parser.add_argument('--model_path', default="../output/best_model_prompt/best_model_epoch_61/", type=str,
#                         help='Path to the trained model folder')
#     parser.add_argument('--batch_size', default=64, type=int, required=False, help='batch size')
#     parser.add_argument('--top_k', default=10, type=int, required=False, help='top k filtering')
#     parser.add_argument('--top_p', default=1.0, type=float, required=False, help='top p filtering')
#     parser.add_argument('--n_prefixes', default=2, type=int, help='Training used 2 prefixes')
#     parser.add_argument('--n_tokens', default=10, type=int, help='Training used 10 tokens')
#     parser.add_argument('--target_prefix_idx', default=1, type=int, help='0 for linear, 1 for head_to_tail')
#     return parser.parse_args()
#
#
# def seed_everything(seed: int):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#
#
# def top_k_top_p_filtering(
#         logits: torch.FloatTensor,
#         top_k: int = 0,
#         top_p: float = 1.0,
#         filter_value: float = -float("Inf"),
#         min_tokens_to_keep: int = 1,
# ) -> torch.FloatTensor:
#     top_p = float(top_p)
#     if top_k > 0:
#         top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
#         indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#         logits = logits.masked_fill(indices_to_remove, filter_value)
#
#     if 0 < top_p <= 1.0:
#         sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#         cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
#         sorted_indices_to_remove = cumulative_probs > top_p
#         if min_tokens_to_keep > 1:
#             sorted_indices_to_remove[..., : min_tokens_to_keep - 1] = 0
#         sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#         sorted_indices_to_remove[..., 0] = 0
#         indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
#         logits = logits.masked_fill(indices_to_remove, filter_value)
#     return logits
#
#
# def decode(matrix):
#     chars = []
#     for i in matrix:
#         # 遇到 SEP 或 EOS 停止解码
#         if i == '[SEP]' or i == '<eos>':
#             break
#         chars.append(i.upper())
#     seq = "".join(chars)
#     return seq
#
#
# def predict(args, model, tokenizer, batch_size):
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model.to(device)
#     model.eval()
#     time1 = time.time()
#
#     # 最大生成长度 (包含前缀长度的考虑，如果 tokenizer.model_max_length 是 576，这里生成内容最好不要超)
#     max_length = 576 - args.n_tokens
#
#     # 1. 构造初始输入 [BOS]
#     # 确保 tokenizer 有 bos_token
#     bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
#
#     # 初始输入: [Batch_Size, 1]
#     input_tensor = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
#
#     # 2. 构造前缀索引 (Target Prefix)
#     # 我们要生成 head_to_tail，所以全设为 1
#     prefix_indices = torch.full((batch_size,), args.target_prefix_idx, dtype=torch.long, device=device)
#
#     Seq_list = []
#     finished = torch.zeros(batch_size, 1).byte().to(device)
#
#     # 开始逐 token 生成
#     # 注意：这里的效率较低（每次重新计算整个序列），但逻辑最清晰。
#     # 如果要优化速度，需要利用 past_key_values (KV Cache)
#     with torch.no_grad():
#         for i in range(max_length):
#             # 关键步骤：手动计算 Embedding
#             # 这样我们才能把 prefix_indices 传进去
#             # SoftEmbedding 会在 input_tensor 前面拼接对应的前缀向量
#             inputs_embeds = model.transformer.wte(input_tensor, prefix_indices=prefix_indices)
#
#             # 将 Embedding 送入模型
#             outputs = model(inputs_embeds=inputs_embeds)
#
#             # 获取最后一个 token 的 logits
#             next_token_logits = outputs.logits[:, -1, :]
#
#             # 采样
#             next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
#             probs = F.softmax(next_token_logits, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1)
#
#             # 检查是否结束 (EOS)
#             EOS_sampled = (next_token == tokenizer.eos_token_id)
#             finished = torch.ge(finished + EOS_sampled, 1)
#
#             # 记录生成的 token (为了解码用，转回 cpu)
#             # 注意：这里我们只记录 token 的字符串形式，或者最后再解码
#             # 为了保持和你原代码一致，这里先存 token string
#             token_str_list = [tokenizer.decode(t) for t in next_token]
#             Seq_list.append(token_str_list)
#
#             # 如果所有 batch 都结束了，提前退出
#             if torch.prod(finished) == 1:
#                 break
#
#             # 拼接生成的 token 到输入，用于下一步预测
#             input_tensor = torch.cat((input_tensor, next_token), dim=1)
#
#     # 转置列表以匹配 dataframe 格式: [Seq_Len, Batch] -> [Batch, Seq_Len]
#     # 原代码逻辑似乎是按列存的，这里调整为按行存更直观，但为了兼容你的 decode 函数：
#     # 你的 decode 接收的是一个 list of tokens (one sequence)
#
#     # Seq_list 是 [Len, Batch] 的 list of strings
#     # 我们需要把它变成 [Batch, Len]
#     Seq_list = np.array(Seq_list).T  # [Batch, Len]
#
#     print("Generation time cost: {:.4f}s".format(time.time() - time1))
#     return Seq_list
#
#
# if __name__ == '__main__':
#     seed_everything(42)
#     args = setup_args()
#
#     # 1. 加载 Tokenizer
#     tokenizer = PreTrainedTokenizerFast.from_pretrained("./MolGPT_pretrained-by-ZINC15")
#
#     # 确保 pad/bos/eos 正常
#     if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
#
#     print(f"Loading model from {args.model_path}...")
#
#     # 2. 加载基础模型结构
#     # 注意：这里我们加载训练好的文件夹，它应该包含 config.json 和 safetensors
#     model = GPT2LMHeadModel.from_pretrained(args.model_path)
#
#     # 3. 重新挂载 SoftEmbedding 层
#     # 必须使用与训练时相同的参数 (n_prefixes=2, n_tokens=10)
#     # initialize_from_vocab=False 因为我们接下来要加载训练好的权重
#     s_wte = SoftEmbedding(model.get_input_embeddings(),
#                           n_prefixes=args.n_prefixes,
#                           n_tokens=args.n_tokens,
#                           initialize_from_vocab=False)
#
#     # 4. 加载训练好的前缀权重
#     # 如果你训练结束时调用了 reparameterize() 并保存了整个模型，
#     # 那么 from_pretrained 可能已经加载了权重，只是结构不对（它是标准的 Embedding）。
#     # 但因为我们替换了 wte，所以需要确保 s_wte 里的 learned_embedding 被正确加载。
#
#     # 检查 from_pretrained 是否自动加载了权重到 model.transformer.wte
#     # 如果你保存的是 state_dict，且 key 是 transformer.wte.learned_embedding
#
#     # 强制重新加载权重以防万一 (特别是如果你只保存了 adapter 或者 safetensors)
#     model_weights_path = os.path.join(args.model_path, "model.safetensors")
#     if not os.path.exists(model_weights_path):
#         model_weights_path = os.path.join(args.model_path, "pytorch_model.bin")
#
#     if os.path.exists(model_weights_path):
#         print("Reloading SoftEmbedding weights explicitly...")
#         state_dict = load_file(model_weights_path) if model_weights_path.endswith('.safetensors') else torch.load(
#             model_weights_path)
#
#         # 寻找 learned_embedding 权重
#         if 'transformer.wte.learned_embedding' in state_dict:
#             s_wte.learned_embedding = torch.nn.Parameter(state_dict['transformer.wte.learned_embedding'])
#             print(f"Loaded learned_embedding with shape: {s_wte.learned_embedding.shape}")
#         else:
#             print(
#                 "WARNING: 'transformer.wte.learned_embedding' not found in state_dict. Using random initialization or pre-loaded weights.")
#
#     # 将构建好的 SoftEmbedding 赋值给模型
#     model.set_input_embeddings(s_wte)
#
#     # 5. 生成
#     output_dir = '../output/generate/prompt_generate/'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     output_file = os.path.join(output_dir, 'cyc_prompt_topk_head_to_tail.csv')
#
#     total_generated = 0
#     target_num = 500
#
#     # 分批生成，直到达到目标数量
#     while total_generated < target_num:
#         current_batch = min(args.batch_size, target_num - total_generated)
#         print(f"Generating batch of {current_batch}...")
#
#         # 调用预测函数
#         Seq_list = predict(args, model, tokenizer, batch_size=current_batch)
#
#         # 解码并保存
#         batch_decoded = []
#         for seq_tokens in Seq_list:
#             batch_decoded.append(decode(seq_tokens))
#
#         df_batch = pd.DataFrame(batch_decoded)
#
#         # 写入文件
#         current_mode = 'w' if total_generated == 0 else 'a'
#         header = False  # 简单起见不写 header，或者第一次写
#         df_batch.to_csv(output_file, mode=current_mode, index=False, header=header, sep=' ')
#
#         total_generated += current_batch
#         print(f"Total generated: {total_generated}")
#
#     print("Generation complete!")