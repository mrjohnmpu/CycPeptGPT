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

# class MyDataset(Dataset):
#     def __init__(self, data_list):
#         self.data_list = data_list
#
#     def __getitem__(self, index):
#         input_ids = self.data_list[index]
#         return input_ids
#
#     def __len__(self):
#         return len(self.data_list)


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
    parser.add_argument('--batch_size', default=64, type=int, required=False,
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
    return parser.parse_args()


# ⬇️ 1. 这是你原来的 calculate_loss_and_accuracy，转换成了 Trainer 需要的格式
def compute_metrics(eval_pred):
    """
    在 Trainer 中计算 token 级别的准确率。
    """
    # eval_pred 是一个 EvalPrediction 对象, 包含 predictions 和 label_ids
    # 它们是 numpy 数组
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    # --- 1. 移位 (Shift) ---
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]

    # --- 2. 计算预测 (Predictions) ---
    preds = np.argmax(shift_logits, axis=-1)

    # --- 3. 计算准确率 (Accuracy) ---
    not_ignore = shift_labels != -100  # 忽略 padding

    num_targets = not_ignore.sum()
    if num_targets == 0:
        return {"accuracy": 0.0}

    correct = (preds == shift_labels) & not_ignore
    correct = correct.sum()

    accuracy = correct / num_targets

    # 返回一个字典，键是指标名称
    return {"accuracy": accuracy}


def calculate_loss_and_accuracy_(outputs, labels, device):
    logits = outputs.logits
    # Shift so that tokens < n predict n
    shift_logits = logits[..., 1:-1, :].contiguous()
    shift_labels = labels[..., 2:].contiguous().to(device)

    # Flatten the tokens
    # **重要**: `tokenizer` 必须是全局可访问的，或者传递进来
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    loss = loss.view(-1, shift_logits.shape[1])

    not_ignore = shift_labels.ne(tokenizer.pad_token_id)
    return loss, not_ignore


# ⬇️ 2. 你的 DataCollator 保持不变，它写得很棒
# class DataCollatorForLanguageModeling:
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#         self.pad_token_id = tokenizer.pad_token_id
#         if self.pad_token_id is None:
#             raise ValueError("Tokenizer must have a pad_token_id.")
#
#     def __call__(self, batch):
#         max_len = max(len(seq) for seq in batch)
#         input_ids_batch = []
#         labels_batch = []
#         attention_mask_batch = []
#         for seq in batch:
#             pad_len = max_len - len(seq)
#             padded_inputs = seq + [self.pad_token_id] * pad_len
#             input_ids_batch.append(padded_inputs)
#             padded_labels = seq + [-100] * pad_len
#             labels_batch.append(padded_labels)
#             mask = [1] * len(seq) + [0] * pad_len
#             attention_mask_batch.append(mask)
#         return {
#             "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
#             "labels": torch.tensor(labels_batch, dtype=torch.long),
#             "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long)
#         }


# 3. data_loader 被修改为只返回 Dataset 对象
# def prepare_datasets(args, tokenizer):
#     data_list = []
#     eval_data_list = []
#
#     df = pd.read_csv(args.train_raw_path)
#     print("数据总行数:{}".format(len(df)))
#
#     df.dropna(subset=['SMILES'], inplace=True)
#     df = df[df['SMILES'].str.len() > 0]
#     print("清洗后数据总行数:{}".format(len(df)))
#
#     train_df, eval_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)
#     print(f"训练集大小: {len(train_df)}, 验证集大小: {len(eval_df)}")
#
#     max_seq_length = tokenizer.model_max_length
#     train_smiles_list = train_df['SMILES'].tolist()
#     eval_smiles_list = eval_df['SMILES'].tolist()
#
#     # (只在主进程显示 tqdm，DDP 环境下 rank 0 是主进程)
#     is_main_process = os.environ.get("RANK", "0") == "0"
#
#     for smiles_string in tqdm(train_smiles_list, desc="Tokenizing training data", disable=not is_main_process):
#         tokenized_ids = tokenizer.encode(
#             smiles_string,
#             truncation=True,
#             max_length=max_seq_length
#         )
#         data_list.append(tokenized_ids)
#
#     train_dataset = MyDataset(data_list)  #返回 Dataset
#
#     for smiles_string in tqdm(eval_smiles_list, desc="Tokenizing eval data", disable=not is_main_process):
#         tokenized_ids = tokenizer.encode(
#             smiles_string,
#             truncation=True,
#             max_length=max_seq_length
#         )
#         eval_data_list.append(tokenized_ids)
#
#     eval_dataset = MyDataset(eval_data_list)  #返回 Dataset
#
#     return train_dataset, eval_dataset


# ⬇️ 4. train 和 evaluate 函数被完全移除

def prompt_contrast_train(args, model, train_dataset):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    num_training_steps = args.epochs * len(train_dataloader)

    # --- 参数冻结 ---
    # 冻结除了 soft embedding 之外的所有参数
    # for param in model.parameters():
    #     param.requires_grad = False
    for param in model.transformer.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False

    # 唯一可训练的参数
    # 注意：在你的 SoftEmbedding 实现中，`self.learned_embedding` 已经是 Parameter
    # 我们需要设置它 `requires_grad=True`
    # 在替换嵌入层后，路径是 model.transformer.wte.learned_embedding
    model.transformer.wte.learned_embedding.requires_grad = True

    # --- 优化器 ---
    # 只优化这一个参数
    optimizer = AdamW([model.transformer.wte.learned_embedding], lr=args.lr)

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

    early_stopping = EarlyStopping(patience=args.patience, verbose=True) # 取决于 pytorchtools.py

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
            batch_0 = batch[0]  # input_ids
            batch_1 = batch[1]  # attention_mask
            batch_3 = batch[3]  # labels

            # **重要**: `tokenizer` 必须是全局可访问的
            pt_id = tokenizer.unk_token_id  # 你的参考代码用 unk 和 mask
            nt_id = tokenizer.eos_token_id

            pt_token = (torch.ones(batch_0.shape[0]) * pt_id).type_as(batch_0).view(-1, 1)
            nt_token = (torch.ones(batch_0.shape[0]) * nt_id).type_as(batch_0).view(-1, 1)

            seq_a = torch.cat((pt_token, batch_0), 1)
            seq_b = torch.cat((nt_token, batch_0), 1)

            mask_token = torch.ones(batch_1.shape[0], 1).type_as(batch_1)
            mask_a = torch.cat((mask_token, batch_1), 1)
            mask_b = torch.cat((mask_token, batch_1), 1)

            bsz = seq_a.shape[0]

            inputs_pos = {"input_ids": seq_a, "labels": seq_a, "attention_mask": mask_a}
            inputs_neg = {"input_ids": seq_b, "labels": seq_b, "attention_mask": mask_b}

            # 假设: 标签 0 应该匹配 neg 提示, 标签 1 应该匹配 pos 提示
            # outputs_a 是 neg 提示 (seq_b) 的输出
            outputs_a = model(**inputs_neg)
            loss_a, loss_mask = calculate_loss_and_accuracy_(outputs_a, seq_a, device)
            loss_lengths = torch.sum(loss_mask, 1, keepdim=True)
            # loss_lengths = torch.clamp(loss_lengths, min=1)
            # print("loss_lengths = {}".format(loss_lengths))

            # outputs_b 是 pos 提示 (seq_a) 的输出
            outputs_b = model(**inputs_pos)
            loss_b, _ = calculate_loss_and_accuracy_(outputs_b, seq_b, device)

            # --- 生成损失 (Gen Loss) ---
            # 标签 0 的样本 (gen_loss_a)
            gen_loss_a = (batch_3 == 0).to(torch.float32).unsqueeze(1) * loss_a / loss_lengths
            # 标签 1 的样本 (gen_loss_b)
            gen_loss_b = (batch_3 == 1).to(torch.float32).unsqueeze(1) * loss_b / loss_lengths
            gen_loss = torch.sum(gen_loss_a + gen_loss_b) / bsz

            # --- 分类损失 (Class Loss) ---
            if args.sum_loss:
                loss_a = loss_a.sum(dim=1)
                loss_b = loss_b.sum(dim=1)
            else:
                loss_a = (loss_a / loss_lengths).sum(dim=1)
                loss_b = (loss_b / loss_lengths).sum(dim=1)

            # loss_a 是 neg 提示的损失, loss_b 是 pos 提示的损失
            # 我们希望:
            # - 标签 0: neg 损失 (loss_a) 小, pos 损失 (loss_b) 大
            # - 标签 1: neg 损失 (loss_a) 大, pos 损失 (loss_b) 小

            # logit = -loss。 loss 越小，logit 越大
            # class_logits 维度: [bsz, 2]
            # 第 0 列: -loss_a (对应 "0" 类, neg)
            # 第 1 列: -loss_b (对应 "1" 类, pos)
            class_logits = torch.stack((-loss_a, -loss_b), dim=1)
            class_labels = batch_3  # 真实标签 (0 或 1)

            if args.logit_scale:
                # (省略 DataParallel 检查)
                class_logits *= model.logit_scale
            if args.outbias:
                class_logits += model.bias

            loss_fn = torch.nn.CrossEntropyLoss()

            # --- 最终损失 (Final Loss) ---
            class_loss = loss_fn(class_logits, class_labels)
            loss = class_loss * (1 - args.gen_weight) + args.gen_weight * gen_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.transformer.wte.learned_embedding], args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_loss_list.append(loss.item())
            progress_bar.set_description(f"Epoch {epoch + 1} | Loss: {loss.item():.4f}")

            print(f"  Step {batch_steps}/{num_training_steps}, Loss: {loss.item()}")

            # if batch_steps % args.log_step == 0:
            #     print(f"  Step {batch_steps}/{num_training_steps}, Loss: {loss.item()}")

        epoch_loss = np.mean(epoch_loss_list)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss}")
        output_dir = os.path.join(args.best_model_dir, f"best_model_epoch_{epoch + 1}")
        ckpt_dir = os.path.join(args.ckpt_model_path, f"best_checkpoint_epoch_{epoch + 1}")
        early_stopping(epoch_loss, model, optimizer, lr_scheduler, epoch, output_dir, ckpt_dir)

        if early_stopping.early_stop:
            print("Early stopping")
            break  # 退出 epoch 循环


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

        # 使用 enumerate 来为 'id' 提供一个占位符
        for i, row in enumerate(reader):

            # --- MODIFICATION START ---

            # 1. 获取文本: 你的 SMILES 字符串
            # 检查列名是否存在
            if 'SMILES' not in row:
                raise ValueError("CSV 文件中未找到 'SMILES' 列")
            if 'Cyclization' not in row:
                raise ValueError("CSV 文件中未找到 'Cyclization' 列")

            text = row['SMILES']

            # 2. 获取 ID: 参考代码需要一个 id，但后续没用。我们用行号
            example_id = str(i)

            # 3. 确定标签 (Label):
            #    线肽 (linear) = 0
            #    其他环肽 = 1
            cyclization_type = row['Cyclization']
            if cyclization_type == 'linear':
                label = 0
            else:
                # 包含了 'head_to_tail', 'sc_to_tail', 'disulfide_bridge'
                label = 1

            # 目标格式 [text, id, label]
            example = [text, example_id, label]

            # --- MODIFICATION END ---

            # --- 下面是参考代码的原始逻辑，现在可以无缝衔接 ---

            if args.sup_data_num <= 0:
                if not args.balanced:
                    data.append(example)
                else:
                    # 分类存储，用于后续平衡
                    if label == 1:
                        data_pos.append(example)
                    else:
                        assert (label == 0)
                        data_neg.append(example)

            else:
                # (这个逻辑是为多类准备的，但我们将其适配为二分类)
                label_str = str(label)
                if not label_str in data_taski.keys():
                    data_taski[label_str] = []
                # 原始代码 [text, id, label, label]
                data_taski[label_str].append([text, example_id, label, label])

    # --- 处理数据平衡 (如果启用) ---
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

    # --- 处理 sup_data_num 逻辑 (如果启用) ---
    elif args.sup_data_num > 0:
        for label_key in data_taski.keys():
            if len(data_taski[label_key]) > args.sup_data_num:
                add_data = random.sample(data_taski[label_key], args.sup_data_num)
            else:
                add_data = data_taski[label_key]
            for ex in add_data:
                # 转换回 [text, id, label] 格式
                data.append([ex[0], ex[1], ex[2]])
        print(f"Using sup_data_num: {len(data)} total")

    # 如果 data 为空，则
    if not data and not (args.sup_data_num <= 0 and args.balanced):
        raise ValueError(f"从 {filepath} 加载数据失败，请检查文件路径和内容。")

    print(f"Total examples to process: {len(data)}")

    # --- 后续 Tokenization 流程 (保持不变) ---
    if args.max_len is None:
        max_length = tokenizer.max_len
    else:
        # 原始长度必须减 1，为提示符 (pt_token) 腾出空间
        max_length = args.max_len - 1

    print(f"Tokenizing {len(data)} examples with max_length: {max_length}...")

    # example[0] 现在是 SMILES 字符串
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

    # example[2] 现在是 0 或 1
    all_labels = torch.tensor([int(example[2]) for example in data], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


if __name__ == '__main__':
    seed_everything(42)
    args = setup_args()
    # Path to the data which will be used to prompt fine-tune the model
    args.train_raw_path = '../data/restored_validation.csv'
    # args.train_raw_path = '../data/restored_test copy.csv'

    initialize_from_vocab = False
    global tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")
    # tokenizer = PreTrainedTokenizerFast.from_pretrained("./MolGPT_pretrained-by-ZINC15")
    tokenizer.model_max_length = args.max_len

    if tokenizer.pad_token_id is None:
        print("Warning: pad_token_id not set. Setting to eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.unk_token is None:
        print("Warning: tokenizer.unk_token is None. Manually setting to '<unk>'.")
        # 1. 告诉 tokenizer 对象，"<unk>" 字符串是它的 unk_token
        tokenizer.unk_token = "<unk>"
        # 2. (可选) 检查 unk_token_id 是否被正确设置 (现在它应该是 3)
        print(f"tokenizer.unk_token_id successfully set to: {tokenizer.unk_token_id}")

    # 确保 unk_token 和 eos_token 存在且不同，用于对比学习
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer 必须有 eos_token (nt_id)")

    if tokenizer.unk_token_id == tokenizer.eos_token_id:
        raise ValueError("unk_token 和 eos_token 不能相同")

    # can not use relative path here, use absolute path instead
    # model = GPT2LMHeadModel.from_pretrained("/home/mrjohn/workingspace/CycPeptGPT/output/best_model_with_mask_trainer/checkpoint-177198")
    model = GPT2LMHeadModel.from_pretrained("/home/xiongshuwen/workingspace/cyc_gpt/output/best_model_with_mask_trainer/checkpoint-177198")

    s_wte = SoftEmbedding(model.get_input_embeddings(),
                          n_tokens=args.n_tokens,
                          initialize_from_vocab=initialize_from_vocab)
    model.set_input_embeddings(s_wte)

    # --- 2. 准备 Datasets ---
    train_dataloader  = load_and_cache_examples(args, args.train_raw_path, tokenizer=tokenizer)

    prompt_contrast_train(args, model, train_dataloader)