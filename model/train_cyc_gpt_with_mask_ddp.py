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
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# (你手动的 EarlyStopping, get_scheduler, CrossEntropyLoss, AdamW 等不再需要)

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
    parser.add_argument('--best_model_dir', default="../output/best_model_with_mask", type=str,
                        help='Trainer 将在此保存 checkpoint')
    parser.add_argument('--final_model_path', default="../output/final_model", type=str, help='最终模型保存路径')
    # (保留 best_ckpt_path 是为了你之前的 early_stop，Trainer 不需要它)
    # parser.add_argument('--best_ckpt_path', default="../output/best_checkpoint_with_mask.pt", type=str, help='')
    parser.add_argument('--train_raw_path', default='train_raw_data.txt', type=str, help='')
    parser.add_argument('--eval_raw_path', default='test_raw_data.txt', type=str, help='')
    parser.add_argument('--batch_size', default=2, type=int, required=False,
                        help='per_device batch size (每个 GPU 的 batch size)')
    parser.add_argument('--epochs', default=1001, type=int, required=False, help='epochs')
    parser.add_argument('--warmup_steps', default=10000, type=int, required=False, help='warm up steps')
    parser.add_argument('--lr', default=5e-5, type=float, required=False, help='learn rate')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)  # Trainer 会自动使用
    parser.add_argument('--log_step', default=10, type=int, required=False, help='logging steps')
    parser.add_argument('--patience', default=10, type=int, required=False, help='early stopping patience')
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


# ⬇️ 2. 你的 DataCollator 保持不变，它写得很棒
class DataCollatorForLanguageModeling:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id.")

    def __call__(self, batch):
        max_len = max(len(seq) for seq in batch)
        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []
        for seq in batch:
            pad_len = max_len - len(seq)
            padded_inputs = seq + [self.pad_token_id] * pad_len
            input_ids_batch.append(padded_inputs)
            padded_labels = seq + [-100] * pad_len
            labels_batch.append(padded_labels)
            mask = [1] * len(seq) + [0] * pad_len
            attention_mask_batch.append(mask)
        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long)
        }


# 3. data_loader 被修改为只返回 Dataset 对象
def prepare_datasets(args, tokenizer):
    data_list = []
    eval_data_list = []

    df = pd.read_csv(args.train_raw_path)
    print("数据总行数:{}".format(len(df)))

    df.dropna(subset=['SMILES'], inplace=True)
    df = df[df['SMILES'].str.len() > 0]
    print("清洗后数据总行数:{}".format(len(df)))

    train_df, eval_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)
    print(f"训练集大小: {len(train_df)}, 验证集大小: {len(eval_df)}")

    max_seq_length = tokenizer.model_max_length
    train_smiles_list = train_df['SMILES'].tolist()
    eval_smiles_list = eval_df['SMILES'].tolist()

    # (只在主进程显示 tqdm，DDP 环境下 rank 0 是主进程)
    is_main_process = os.environ.get("RANK", "0") == "0"

    for smiles_string in tqdm(train_smiles_list, desc="Tokenizing training data", disable=not is_main_process):
        tokenized_ids = tokenizer.encode(
            smiles_string,
            truncation=True,
            max_length=max_seq_length
        )
        data_list.append(tokenized_ids)

    train_dataset = MyDataset(data_list)  #返回 Dataset

    for smiles_string in tqdm(eval_smiles_list, desc="Tokenizing eval data", disable=not is_main_process):
        tokenized_ids = tokenizer.encode(
            smiles_string,
            truncation=True,
            max_length=max_seq_length
        )
        eval_data_list.append(tokenized_ids)

    eval_dataset = MyDataset(eval_data_list)  #返回 Dataset

    return train_dataset, eval_dataset


# ⬇️ 4. train 和 evaluate 函数被完全移除

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    seed_everything(42)
    args = setup_args()
    args.train_raw_path = '../data/restored_test.csv'

    # --- 1. 加载 Tokenizer 和 Model (与你原来的一样) ---
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./MolGPT_pretrained-by-ZINC15")
    tokenizer.model_max_length = 576

    if tokenizer.pad_token_id is None:
        print("Warning: pad_token_id not set. Setting to eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    model_config = GPT2Config(
        architectures=["GPT2LMHeadModel"],
        model_type="GPT2LMHeadModel",
        vocab_size=tokenizer.vocab_size,
        n_positions=576,
        n_ctx=576,
        n_embd=768,
        n_layer=12,
        n_head=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        task_specific_params={
            "text-generation": {
                "do_sample": True,
                "max_length": 576
            }
        }
    )
    model = GPT2LMHeadModel(config=model_config)

    # --- 2. 准备 Datasets ---
    train_dataset, eval_dataset = prepare_datasets(args, tokenizer)

    # --- 3. 准备 Collator ---
    data_collator = DataCollatorForLanguageModeling(tokenizer)

    # --- 4. 定义 Training Arguments ---
    # 这里我们使用你 argparse 传入的参数
    training_args = TrainingArguments(
        output_dir=args.best_model_dir,  # Checkpoint 保存路径
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,  # 这是 *每个 GPU* 的 batch size
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,

        # 评估 和 日志
        eval_strategy="epoch",  # 每隔一个 epoch 评估一次
        logging_strategy="epoch",
        # logging_strategy="steps",
        # logging_steps=args.log_step,
        report_to="tensorboard",  # 报告给 tensorboard (或 "wandb")

        # 保存 和 Early Stopping
        save_strategy="epoch",  # 每隔一个 epoch 保存一次
        save_total_limit=3,  # 最多保留 3 个 checkpoint
        load_best_model_at_end=True,  # 训练结束时加载最佳模型
        metric_for_best_model="eval_loss",  # 监控 eval_loss
        greater_is_better=False,

        # 性能
        fp16=True,  # 启用混合精度训练 (需要 NVIDIA GPU)

        # ⬇️ 分布式训练：你不需要设置！
        # Trainer 会自动检测 torchrun 启动的环境变量 (RANK, WORLD_SIZE, LOCAL_RANK)
        # 它会自动将 batch size 应用到每个 device
        # 它会自动使用 DistributedSampler
    )

    # --- 5. ⬇️ 实例化 Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    # --- 6. ⬇️ 开始训练 ---
    print("开始使用 Trainer 进行训练...")
    trainer.train()

    # --- 7. ⬇️ 保存最终模型 ---
    # (由于 load_best_model_at_end=True, trainer.model 现在是最佳模型)
    print("训练完成。保存最佳模型。")
    trainer.save_model(args.final_model_path)
    print(f"最佳模型已保存至: {args.final_model_path}")

    # torchrun --nproc_per_node=1 train_cyc_gpt_with_mask_ddp.py