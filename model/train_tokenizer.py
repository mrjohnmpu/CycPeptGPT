# -*- coding: utf-8 -*-
"""
使用Hugging Face Transformers训练一个肽链生成GPT-2模型 (修正版)

这个Python脚本将指导您完成从头开始训练一个专门用于生成肽链SMILES的GPT-2模型的全过程。

此版本修正了分词器训练的逻辑，确保它只在干净的SMILES数据上进行训练。

在运行此脚本前，请确保已安装所有必需的库:
pip install transformers datasets tokenizers pandas torch rdkit
"""

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
    pipeline
)
from datasets import load_dataset
import re
from rdkit import Chem

# --- 1. 定义全局变量和函数 ---

DATA_FILE = '../data/restored_train.csv'
TOKENIZER_PATH = 'peptide_tokenizer'
MODEL_PATH = 'peptide_gpt2_model'


def train_tokenizer():
    """
    修正版的分词器训练函数。
    它现在直接从 .txt 文件训练。
    """
    print("--- 步骤 2: 开始训练分词器 (修正版) ---")
    
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"错误: 未找到数据文件 '{DATA_FILE}'。")

    # 初始化分词器
    tokenizer = ByteLevelBPETokenizer()

    # *** 修改: 直接使用 .txt 文件进行训练 ***
    tokenizer.train(
        files=[DATA_FILE], 
        vocab_size=1000, 
        min_frequency=2, 
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    # 创建目录并保存
    if not os.path.exists(TOKENIZER_PATH):
        os.makedirs(TOKENIZER_PATH)
    tokenizer.save_model(TOKENIZER_PATH)

    print(f"分词器训练完成并保存至 '{TOKENIZER_PATH}' 目录。")
    
    # --- 增加验证步骤 ---
    print("\n--- 验证分词器词汇表 ---")
    trained_tokenizer = Tokenizer.from_file(os.path.join(TOKENIZER_PATH, "vocab.json"))
    vocabulary = trained_tokenizer.get_vocab()
    
    key_tokens = ['[C@@H]', '[C@H]', 'C(=O)', 'c1ccccc1']
    all_found = True
    for token in key_tokens:
        if token in vocabulary:
            print(f"✅ 成功学习到词元: '{token}'")
        else:
            print(f"❌ 未能将 '{token}' 作为单一词元学习。")
            all_found = False
            
    if not all_found:
        print("\n警告：部分关键化学片段未能学习成功，请检查数据文件或训练参数。")
    else:
        print("\n分词器看起来训练得不错！")

    print("\n")



# ... 后续函数 (prepare_dataset, train_model, generate_and_sample) 保持不变 ...
def prepare_dataset(tokenizer):
    """加载数据并对其进行分词处理"""
    print("--- 步骤 3: 准备数据集 ---")
    
    # 加载数据集
    dataset = load_dataset('csv', data_files=DATA_FILE)

    # 定义分词函数
    def tokenize_function(examples):
        return tokenizer(examples['SMILES'], truncation=True, max_length=1024)

    # 对整个数据集进行分词
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset['train'].column_names)

    print("数据集准备完成。")
    print(tokenized_dataset)
    print("\n")
    return tokenized_dataset

def train_model(tokenized_dataset, tokenizer):
    """配置并训练GPT-2模型"""
    print("--- 步骤 4: 配置与训练模型 ---")
    
    # 配置一个新的GPT-2模型
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_embd=256,
        n_layer=6,
        n_head=8
    )

    # 实例化模型
    model = GPT2LMHeadModel(config=config)

    # 定义数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=MODEL_PATH,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        save_steps=5000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,
    )

    # 实例化训练器
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

def generate_and_sample():
    """使用训练好的模型生成新的肽链分子"""
    print("--- 步骤 5: 生成与采样 ---")
    
    # 加载生成管道
    generator = pipeline('text-generation', model=MODEL_PATH, tokenizer=TOKENIZER_PATH)

    def generate_smiles(prompt="<s>", num_samples=5):
        generated_outputs = generator(
            prompt,
            max_length=200,
            num_return_sequences=num_samples,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        
        valid_smiles = []
        for output in generated_outputs:
            smiles = output['generated_text']
            smiles = smiles.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip()
            
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                print(f"有效 SMILES: {smiles}")
                valid_smiles.append(smiles)
            else:
                print(f"无效 SMILES: {smiles}")
        
        return valid_smiles

    print("\n--- 开始生成新的肽链分子 ---")
    new_peptides = generate_smiles(prompt="<s>", num_samples=10)

    print(f"\n成功生成了 {len(new_peptides)} 个有效的肽链分子。")


# --- 主执行流程 ---
if __name__ == '__main__':
    # 步骤 2: 训练分词器 (使用修正后的方法)
    train_tokenizer()
    
    # # 加载训练好的分词器
    # gpt2_tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
    
    # # 步骤 3: 准备数据集
    # prepared_data = prepare_dataset(gpt2_tokenizer)
    
    # # 步骤 4: 训练模型
    # train_model(prepared_data, gpt2_tokenizer)
    
    # # 步骤 5: 生成新分子
    # generate_and_sample()
