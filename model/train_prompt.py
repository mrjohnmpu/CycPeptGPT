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
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_scheduler
)
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
from sklearn.model_selection import train_test_split
from soft_prompt_embedding import SoftEmbedding
from torch.optim import AdamW
from early_stop.pytorchtools import EarlyStopping
from torch.nn import CrossEntropyLoss


def seed_everything(seed: int):
    """
    固定所有随机种子以确保结果可复现。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="", type=str, help='')
    parser.add_argument('--vocab_path', default="", type=str, help='')
    parser.add_argument('--best_model_dir', default="../output/best_model_prompt", type=str,
                        help='Trainer 将在此保存 checkpoint')
    parser.add_argument('--ckpt_model_path', default="../output/ckpt_model_prompt", type=str, help='最终模型保存路径')
    # (保留 best_ckpt_path 是为了你之前的 early_stop，Trainer 不需要它)
    # parser.add_argument('--best_ckpt_path', default="../output/best_checkpoint_with_mask.pt", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.txt', type=str, help='')
    parser.add_argument('--eval_raw_path', default='test_raw_data.txt', type=str, help='')
    parser.add_argument('--batch_size', default=4, type=int, required=False,
                        help='per_device batch size (每个 GPU 的 batch size)')
    parser.add_argument('--accumulation_steps', default=4, type=int, required=False)
    parser.add_argument('--epochs', default=200, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=500, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=1e-4, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)  # Trainer 会自动使用
    parser.add_argument('--log_step', default=10, type=int, required=False, help='logging steps')
    parser.add_argument('--patience', default=10, type=int, required=False, help='early stopping patience')
    parser.add_argument('--max_len', default=576, type=int, required=False, help='The max length for each sequence')
    parser.add_argument('--n_tokens', default=10, type=int, required=False, help='number of soft tokens')
    parser.add_argument("--balanced", default=True, help="use balanced dataset for training")
    parser.add_argument('--sup_data_num', default=0, type=int, help='the number of supervised data for each prefix')
    parser.add_argument('--sum_loss', default=False)
    parser.add_argument("--logit_scale", default=False, help="learns to scale logits for classification")
    parser.add_argument("--outbias", default=False, help="learns to add bias for classification")
    parser.add_argument("--gen_weight", default=0.9, type=float, help="scalar multiple for generative loss (lambda)")
    parser.add_argument('--n_prefixes', default=2, type=int, required=False, help='number of soft prefixes')
    parser.add_argument('--mid_dim', default=512, type=int, required=False, help='hidden dimension of soft prompt MLP')
    return parser.parse_args()


def compute_metrics(eval_pred):
    """
    在 Trainer 中计算 token 级别的准确率。
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    preds = np.argmax(shift_logits, axis=-1)

    not_ignore = shift_labels != -100

    num_targets = not_ignore.sum()
    if num_targets == 0:
        return {"accuracy": 0.0}

    correct = (preds == shift_labels) & not_ignore
    correct = correct.sum()

    accuracy = correct / num_targets

    return {"accuracy": accuracy}


def calculate_loss_and_accuracy_(outputs, labels, device):
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., 1:-1, :].contiguous()
    shift_labels = labels[..., 2:].contiguous().to(device)

    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    loss = loss.view(-1, shift_logits.shape[1])

    not_ignore = shift_labels.ne(tokenizer.pad_token_id)
    return loss, not_ignore


# def prompt_contrast_train(args, model, train_dataset):
#     train_sampler = RandomSampler(train_dataset)
#     train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
#
#     num_training_steps = args.epochs * len(train_dataloader)
#
#     for param in model.transformer.parameters():
#         param.requires_grad = False
#     for param in model.lm_head.parameters():
#         param.requires_grad = False
#
#     model.transformer.wte.learned_embedding.requires_grad = True
#
#     optimizer = AdamW([model.transformer.wte.learned_embedding], lr=args.lr)
#
#     lr_scheduler = get_scheduler(
#         name="linear",
#         optimizer=optimizer,
#         num_warmup_steps=args.warmup_steps,
#         num_training_steps=num_training_steps
#     )
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model.to(device)
#     model.train()
#     batch_steps = 0
#
#     early_stopping = EarlyStopping(patience=args.patience, verbose=True)  # 取决于 pytorchtools.py
#
#     print("--- 开始对比学习训练 ---")
#     print(f"Total steps: {num_training_steps}")
#     print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
#
#     for epoch in range(args.epochs):
#         epoch_loss_list = []
#         print("\n")
#         print("***********")
#         print(f"Epoch {epoch + 1}/{args.epochs}")
#         print(f"LR: {optimizer.state_dict()['param_groups'][0]['lr']}")
#         print("***********")
#         print("\n")
#
#         progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
#
#         for batch in progress_bar:
#             batch_steps += 1
#
#             batch = tuple(t.to(device) for t in batch)
#             batch_0 = batch[0]  # input_ids
#             batch_1 = batch[1]  # attention_mask
#             batch_3 = batch[3]  # labels
#
#             pt_id = tokenizer.unk_token_id
#             nt_id = tokenizer.eos_token_id
#
#             pt_token = (torch.ones(batch_0.shape[0]) * pt_id).type_as(batch_0).view(-1, 1)
#             nt_token = (torch.ones(batch_0.shape[0]) * nt_id).type_as(batch_0).view(-1, 1)
#
#             seq_a = torch.cat((pt_token, batch_0), 1)
#             seq_b = torch.cat((nt_token, batch_0), 1)
#
#             mask_token = torch.ones(batch_1.shape[0], 1).type_as(batch_1)
#             mask_a = torch.cat((mask_token, batch_1), 1)
#             mask_b = torch.cat((mask_token, batch_1), 1)
#
#             bsz = seq_a.shape[0]
#
#             inputs_pos = {"input_ids": seq_a, "labels": seq_a, "attention_mask": mask_a}
#             inputs_neg = {"input_ids": seq_b, "labels": seq_b, "attention_mask": mask_b}
#
#             outputs_a = model(**inputs_neg)
#             loss_a, loss_mask = calculate_loss_and_accuracy_(outputs_a, seq_a, device)
#             loss_lengths = torch.sum(loss_mask, 1, keepdim=True)
#
#             outputs_b = model(**inputs_pos)
#             loss_b, _ = calculate_loss_and_accuracy_(outputs_b, seq_b, device)
#
#             gen_loss_a = (batch_3 == 0).to(torch.float32).unsqueeze(1) * loss_a / loss_lengths
#             gen_loss_b = (batch_3 == 1).to(torch.float32).unsqueeze(1) * loss_b / loss_lengths
#             gen_loss = torch.sum(gen_loss_a + gen_loss_b) / bsz
#
#             if args.sum_loss:
#                 loss_a = loss_a.sum(dim=1)
#                 loss_b = loss_b.sum(dim=1)
#             else:
#                 loss_a = (loss_a / loss_lengths).sum(dim=1)
#                 loss_b = (loss_b / loss_lengths).sum(dim=1)
#
#             class_logits = torch.stack((-loss_a, -loss_b), dim=1)
#             class_labels = batch_3
#
#             if args.logit_scale:
#                 class_logits *= model.logit_scale
#             if args.outbias:
#                 class_logits += model.bias
#
#             loss_fn = torch.nn.CrossEntropyLoss()
#
#             class_loss = loss_fn(class_logits, class_labels)
#             loss = class_loss * (1 - args.gen_weight) + args.gen_weight * gen_loss
#
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_([model.transformer.wte.learned_embedding], args.max_grad_norm)
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()
#
#             epoch_loss_list.append(loss.item())
#             progress_bar.set_description(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")
#
#             print(f"  Step {batch_steps}/{num_training_steps}, Loss: {loss.item()}")
#
#         epoch_loss = np.mean(epoch_loss_list)
#         print(f"Epoch {epoch + 1} average loss: {epoch_loss}")
#         output_dir = os.path.join(args.best_model_dir, f"best_model_epoch_{epoch + 1}")
#         ckpt_dir = os.path.join(args.ckpt_model_path, f"best_checkpoint_epoch_{epoch + 1}")
#         early_stopping(epoch_loss, model, optimizer, lr_scheduler, epoch, output_dir, ckpt_dir)
#
#         if early_stopping.early_stop:
#             print("Early stopping")
#             break

def prompt_contrast_train(args, model, train_dataset):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    num_training_steps = args.epochs * len(train_dataloader)

    # for param in model.transformer.parameters():
    #     param.requires_grad = False
    # for param in model.lm_head.parameters():
    #     param.requires_grad = False
    #
    # # model.transformer.wte.learned_embedding.requires_grad = True
    # model.transformer.wte.input_tokens.requires_grad = True
    # # 2. 确保 MLP 网络 W 可训练
    # for param in model.transformer.wte.trans.parameters():
    #     param.requires_grad = True
    #
    #
    # # optimizer = AdamW([model.transformer.wte.learned_embedding], lr=args.lr)
    # # 我们需要优化的是 input_tokens 和 trans 中的所有参数
    # trainable_params = [
    #     {'params': [model.transformer.wte.input_tokens]},
    #     {'params': model.transformer.wte.trans.parameters()}
    # ]
    # optimizer = AdamW(trainable_params, lr=args.lr)

    # 1. 先冻结模型所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 2. 解冻重参数化涉及的组件
    # 注意：此时 learned_embedding 是 None，不要碰它
    if model.transformer.wte.learned_embedding is None:
        # 训练模式：解冻 H' 和 MLP
        model.transformer.wte.input_tokens.requires_grad = True
        for param in model.transformer.wte.trans.parameters():
            param.requires_grad = True
        print("Reparameterization mode detected: Training input_tokens and MLP.")

        # 3. 定义优化器 (针对 H' 和 MLP)
        optimizer_params = [
            {'params': [model.transformer.wte.input_tokens]},
            {'params': model.transformer.wte.trans.parameters()}
        ]
    else:
        # 推理模式或非重参数化模式 (兼容旧代码)
        model.transformer.wte.learned_embedding.requires_grad = True
        print("Standard mode detected: Training learned_embedding directly.")
        optimizer_params = [model.transformer.wte.learned_embedding]

    optimizer = AdamW(optimizer_params, lr=args.lr)



    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()
    batch_steps = 0

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)  # 取决于 pytorchtools.py

    print("--- 开始对比学习训练 ---")
    print(f"Total steps: {num_training_steps}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(args.epochs):
        epoch_loss_list = []
        print("\n")
        print("***********")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"LR: {optimizer.state_dict()['param_groups'][0]['lr']}")
        print("***********")
        print("\n")

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            batch_steps += 1

            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]  # SMILES 序列 ID
            attention_mask = batch[1]  # 原始 Mask
            labels_class = batch[3]  # 0 (Linear) 或 1 (Cyclic)
            batch_size = input_ids.shape[0]

            # 2. 构造前缀索引 (用于控制 SoftEmbedding)
            # 场景 A: 强制使用前缀 0 (Linear)
            idx_linear = torch.zeros(batch_size, dtype=torch.long, device=device)
            # 场景 B: 强制使用前缀 1 (Cyclic/Head_to_Tail)
            idx_cyclic = torch.ones(batch_size, dtype=torch.long, device=device)

            # 3. 获取 Embeddings (手动调用 model.transformer.wte)
            # 这里调用的是你修改后的 SoftEmbedding.forward
            # 注意：我们直接传入原始 input_ids，SoftEmbedding 会自动在前面拼上前缀向量
            embeds_linear = model.transformer.wte(input_ids, prefix_indices=idx_linear)  # 每条序列前面都加上线性前缀
            embeds_cyclic = model.transformer.wte(input_ids, prefix_indices=idx_cyclic)  # 每条序列前面都加上环化前缀

            # 4. 修正 Attention Mask
            # SoftEmbedding 增加的长度是 args.n_tokens。
            # 我们需要构造一个全是 1 的 mask 来覆盖前缀部分。
            prefix_mask = torch.ones(batch_size, args.n_tokens, device=device)
            extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            # 5. 构造生成任务的 Labels (用于计算 LM Loss)
            # 前缀部分不参与 Loss 计算，设为 -100
            prefix_labels = torch.full((batch_size, args.n_tokens), -100, dtype=torch.long, device=device)

            # 原始序列的 Labels 通常就是 input_ids 本身
            # 但需要把 padding 部分设为 -100 以忽略计算
            lm_labels = input_ids.clone()
            lm_labels[attention_mask == 0] = -100

            # 拼接
            extended_labels = torch.cat([prefix_labels, lm_labels], dim=1)

            # --- 计算 Linear 前缀下的输出 ---
            # 对应原本的 outputs_a
            outputs_linear = model(
                inputs_embeds=embeds_linear,
                attention_mask=extended_attention_mask,
                labels=extended_labels
            )

            # --- 计算 Cyclic 前缀下的输出 ---
            # 对应原本的 outputs_b
            outputs_cyclic = model(
                inputs_embeds=embeds_cyclic,
                attention_mask=extended_attention_mask,
                labels=extended_labels  # 这里的 label 只是为了让 model 内部算个 loss，我们后面主要用 logits
            )

            # -------------------------------------------------------
            # 前向传播计算 Logits
            # -------------------------------------------------------
            # 计算 P(x | Prefix_0)
            loss_linear = outputs_linear.loss  # 这是 mean loss，我们需要 per-sample loss 用于对比
            # 为了对比损失，我们需要每个样本的 loss，所以重新计算
            logits_linear = outputs_linear.logits
            shift_logits_linear = logits_linear[..., :-1, :].contiguous()
            shift_labels = extended_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
            # Sum over sequence length to get log probability (negative) ==> [batch_size, seq_len]
            nll_linear = loss_fct(
                shift_logits_linear.view(-1, shift_logits_linear.size(-1)), shift_labels.view(-1)
            ).view(batch_size, -1).sum(dim=1)

            # 计算 P(x | Prefix_1)
            logits_cyclic = outputs_cyclic.logits
            shift_logits_cyclic = logits_cyclic[..., :-1, :].contiguous()
            nll_cyclic = loss_fct(
                shift_logits_cyclic.view(-1, shift_logits_cyclic.size(-1)), shift_labels.view(-1)
            ).view(batch_size, -1).sum(dim=1)

            # -------------------------------------------------------
            # 计算 Loss
            # -------------------------------------------------------
            # L_LM: 语言模型损失 (只优化正确的那个前缀)
            # 如果 label=0, loss = nll_0; 如果 label=1, loss = nll_1
            loss_lm_vec = torch.where(labels_class == 0, nll_linear, nll_cyclic)
            loss_lm = loss_lm_vec.mean()

            # L_d: 判别式损失 (Contrastive)
            # 目标: Log( P(correct) / (P(correct) + P(wrong)) )
            #      = Log( exp(-nll_correct) / (exp(-nll_0) + exp(-nll_1)) )
            #      = -nll_correct - LogSumExp(-nll_0, -nll_1)

            nll_correct = torch.where(labels_class == 0, nll_linear, nll_cyclic)
            # 为了数值稳定，使用 log_softmax
            # stack logits: [batch, 2] -> col 0 is neg_nll_0, col 1 is neg_nll_1
            log_probs_stack = torch.stack([-nll_linear, -nll_cyclic], dim=1)
            loss_d = torch.nn.CrossEntropyLoss()(log_probs_stack, labels_class)

            loss = args.gen_weight * loss_lm + (1 - args.gen_weight) * loss_d

            loss.backward()
            # torch.nn.utils.clip_grad_norm_([model.transformer.wte.learned_embedding], args.max_grad_norm)
            # -------------------------------------------------------
            # 梯度裁剪 (Gradient Clipping)
            # -------------------------------------------------------
            if model.transformer.wte.learned_embedding is None:
                # 重参数化模式：裁剪 H' 和 MLP 的梯度
                # 收集所有需要裁剪的参数
                params_to_clip = [model.transformer.wte.input_tokens] + list(model.transformer.wte.trans.parameters())
            else:
                # 普通模式：裁剪 learned_embedding
                params_to_clip = [model.transformer.wte.learned_embedding]

            # 执行裁剪
            torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss_list.append(loss.item())
            progress_bar.set_description(
                f"Epoch {epoch + 1} | Loss: {loss.item():.4f} | LM Loss: {loss_lm.item():.4f} | D Loss: {loss_d.item():.4f}")

            print(f"  Step {batch_steps}/{num_training_steps}, Loss: {loss.item()}")

        epoch_loss = np.mean(epoch_loss_list)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss}")
        output_dir = os.path.join(args.best_model_dir, f"best_model_epoch_{epoch + 1}")
        ckpt_dir = os.path.join(args.ckpt_model_path, f"best_checkpoint_epoch_{epoch + 1}")
        # 在保存最终模型之前，必须调用 reparameterize()
        model.transformer.wte.reparameterize()
        early_stopping(epoch_loss, model, optimizer, lr_scheduler, epoch, output_dir, ckpt_dir)

        if early_stopping.early_stop:
            print("Early stopping")
            break


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def load_and_cache_examples(args, filepath, tokenizer):
    data_pos = []  # 存储正样本 (环肽, label 1)
    data_neg = []  # 存储负样本 (线肽, label 0)
    data = []  # 最终处理列表
    data_taski = {}  # 用于 sup_data_num 逻辑

    print(f"Loading data from: {filepath}")

    # 确保使用 utf-8 打开
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):

            if 'SMILES' not in row:
                raise ValueError("CSV 文件中未找到 'SMILES' 列")
            if 'Cyclization' not in row:
                raise ValueError("CSV 文件中未找到 'Cyclization' 列")

            text = row['SMILES']

            example_id = str(i)

            cyclization_type = row['Cyclization']
            if cyclization_type == 'linear':
                label = 0
            else:
                label = 1

            example = [text, example_id, label]

            if args.sup_data_num <= 0:
                if not args.balanced:
                    data.append(example)
                else:
                    if label == 1:
                        data_pos.append(example)
                    else:
                        assert (label == 0)
                        data_neg.append(example)

            else:
                label_str = str(label)
                if not label_str in data_taski.keys():
                    data_taski[label_str] = []
                data_taski[label_str].append([text, example_id, label, label])

    if args.sup_data_num <= 0 and args.balanced:
        print(f"Balancing data... Pos: {len(data_pos)}, Neg: {len(data_neg)}")
        if len(data_pos) > len(data_neg):
            data_neg_expand = data_neg * (len(data_pos) // len(data_neg))
            data = data_pos + data_neg_expand + random.sample(data_neg, len(data_pos) - len(data_neg_expand))
        elif len(data_neg) > len(data_pos):
            data_pos_expand = data_pos * (len(data_neg) // len(data_pos))
            data = data_neg + data_pos_expand + random.sample(data_pos, len(data_neg) - len(data_pos_expand))
        else:
            data = data_neg + data_pos
        print(f"Balanced data size: {len(data)}")

    elif args.sup_data_num > 0:
        for label_key in data_taski.keys():
            if len(data_taski[label_key]) > args.sup_data_num:
                add_data = random.sample(data_taski[label_key], args.sup_data_num)
            else:
                add_data = data_taski[label_key]
            for ex in add_data:
                data.append([ex[0], ex[1], ex[2]])
        print(f"Using sup_data_num: {len(data)} total")

    if not data and not (args.sup_data_num <= 0 and args.balanced):
        raise ValueError(f"从 {filepath} 加载数据失败，请检查文件路径和内容。")

    print(f"Total examples to process: {len(data)}")

    if args.max_len is None:
        # max_length = tokenizer.max_len
        max_length = tokenizer.model_max_length - args.n_tokens
    else:
        max_length = args.max_len - args.n_tokens

    print(f"Tokenizing {len(data)} examples with max_length: {max_length}...")

    batch_encoding = tokenizer(
        [example[0] for example in data],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_token_type_ids=True,
    )

    all_input_ids = torch.tensor([batch_encoding['input_ids'][i] for i in range(len(data))], dtype=torch.long)
    all_attention_mask = torch.tensor([batch_encoding['attention_mask'][i] for i in range(len(data))], dtype=torch.long)
    all_token_type_ids = torch.tensor([batch_encoding['token_type_ids'][i] for i in range(len(data))], dtype=torch.long)

    all_labels = torch.tensor([int(example[2]) for example in data], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


if __name__ == '__main__':
    seed_everything(42)
    args = setup_args()
    # args.train_raw_path = '../data/restored_validation.csv'
    args.train_raw_path = '../data/filtered_peptides.csv'

    initialize_from_vocab = False
    global tokenizer
    # tokenizer = PreTrainedTokenizerFast.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./MolGPT_pretrained-by-ZINC15")
    tokenizer.model_max_length = args.max_len

    if tokenizer.pad_token_id is None:
        print("Warning: pad_token_id not set. Setting to eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.unk_token is None:
        print("Warning: tokenizer.unk_token is None. Manually setting to '<unk>'.")
        tokenizer.unk_token = "<unk>"
        print(f"tokenizer.unk_token_id successfully set to: {tokenizer.unk_token_id}")

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer 必须有 eos_token (nt_id)")

    if tokenizer.unk_token_id == tokenizer.eos_token_id:
        raise ValueError("unk_token 和 eos_token 不能相同")

    model = GPT2LMHeadModel.from_pretrained(
        "/home/mrjohn/workingspace/CycPeptGPT/output/best_model_with_mask_trainer/checkpoint-177198")
    # model = GPT2LMHeadModel.from_pretrained("/home/xiongshuwen/workingspace/cyc_gpt/output/best_model_with_mask_trainer/checkpoint-177198")

    # Embedding(2140, 768)
    s_wte = SoftEmbedding(model.get_input_embeddings(),
                          n_prefixes=args.n_prefixes,
                          n_tokens=args.n_tokens,
                          initialize_from_vocab=initialize_from_vocab,
                          mid_dim=args.mid_dim)
    model.set_input_embeddings(s_wte)

    train_dataloader = load_and_cache_examples(args, args.train_raw_path, tokenizer=tokenizer)

    prompt_contrast_train(args, model, train_dataloader)
