# -*- coding: utf-8 -*-

import os
import pandas as pd
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
    PreTrainedTokenizerFast
)
from datasets import load_dataset
import re
from rdkit import Chem

# --- 1. 定义全局变量和函数 ---

TRAIN_DATA_FILE = 'data/restored_test.csv'
TOKENIZER_PATH = '/home/xiongshuwen/workingspace/cyc_gpt/model/peptide_tokenizer'
MODEL_PATH = 'peptide_gpt2_model'
LOGGING_PATH = 'logs' # 新增：用于存放 TensorBoard 日志的目录


def prepare_dataset(tokenizer):
    """加载数据并对其进行分词处理"""
    print("--- 步骤 3: 准备数据集 ---")
    
    # 加载数据集
    dataset = load_dataset('csv', data_files=TRAIN_DATA_FILE)

    # 定义分词函数
    def tokenize_function(examples):
        return tokenizer(examples['SMILES'], truncation=True, max_length=1024, add_special_tokens=True, padding='max_length')

    # 对整个数据集进行分词
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    print("数据集准备完成。")
    print(tokenized_dataset)

    # 添加一个简单的检查，打印第一个分词后的结果
    if len(tokenized_dataset['train']) > 0:
        print("\n--- 查看第一条分词后的tokens ---")
        first_item_tokens = tokenizer.convert_ids_to_tokens(tokenized_dataset['train'][0]['input_ids'])
        print(first_item_tokens)
        print(tokenized_dataset['train'][0]['SMILES'])


    print("\n")
    return tokenized_dataset


def train_model(tokenized_dataset, tokenizer):
    """配置并训练GPT-2模型 (已更新以支持监控)"""
    print("--- 步骤 4: 配置与训练模型 ---")
    
    # 配置一个新的GPT-2模型
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=256,
        n_layer=6,
        n_head=8,
        n_ctx=1024
    )

    # 实例化模型
    model = GPT2LMHeadModel(config=config)

    # 定义数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 定义训练参数 (*** 主要修改部分 ***)
    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        save_steps=5000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,
        # --- 新增的监控相关参数 ---
        logging_dir=LOGGING_PATH,          # 指定日志目录
        logging_strategy="steps",          # 按步数记录日志
        logging_steps=100,                 # 每 100 步记录一次日志
        report_to="tensorboard"            # 告诉 Trainer 将日志报告给 TensorBoard
    )

    # 实例化训练器 (Trainer)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train'],
    )

    # 开始训练
    print("开始模型训练...")
    trainer.train()

    # 保存最终模型
    trainer.save_model(MODEL_PATH)
    print(f"模型训练完成并保存至 '{MODEL_PATH}' 目录。\n")

# def generate_and_sample():
# # ... a lot of code ...


# --- 主执行流程 ---
if __name__ == '__main__':

    # # 加载训练好的分词器
    # tokenizer = PreTrainedTokenizerFast("jonghyunlee/MolGPT_pretrained-by-ZINC15")
    # # 为GPT2Tokenizer明确设置特殊符号，确保行为一致
    # tokenizer.pad_token = "<pad>"
    # tokenizer.bos_token = "<bos>"
    # tokenizer.eos_token = "<eos>"
    # tokenizer.unk_token = "<unk>"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    
    # 步骤 3: 准备数据集
    prepared_data = prepare_dataset(tokenizer)
    
    # # 步骤 4: 训练模型
    train_model(prepared_data, tokenizer)
    
    # # 步骤 5: 生成新分子
    # generate_and_sample()
