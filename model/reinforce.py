#!/usr/bin/env python
import sys
import argparse
import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# === 引入你的自定义模块 ===
# 假设这些文件都在当前目录下，或者已添加到 PYTHONPATH
from prompt_lightning_module import ContrastivePrefixModule
from soft_prompt_embedding import SoftEmbedding
from utils.PermeabilityAnalysis import *

# 如果你有 Experience Replay 的需求，可以保留并修改它，这里我们先实现一个简单的 Buffer
class SimpleExperience:
    """简单的经验回放缓冲区"""
    def __init__(self, max_size=100):
        self.memory = []
        self.max_size = max_size

    def add(self, sequences, rewards, prior_log_probs):
        """
        sequences: list of token_ids (list of list)
        rewards: numpy array
        prior_log_probs: numpy array
        """
        for seq, rew, prior_lp in zip(sequences, rewards, prior_log_probs):
            self.memory.append((seq, rew, prior_lp))
        
        # 排序并截断 (保留高分样本)
        self.memory.sort(key=lambda x: x[1], reverse=True)
        self.memory = self.memory[:self.max_size]

    def sample(self, n):
        if len(self.memory) < n:
            return [], [], []
        
        # 简单的随机采样
        indices = np.random.choice(len(self.memory), n, replace=False)
        batch = [self.memory[i] for i in indices]
        
        seqs = [x[0] for x in batch]
        rews = np.array([x[1] for x in batch])
        prior_lps = np.array([x[2] for x in batch])
        
        return seqs, rews, prior_lps

# === 核心工具函数 ===

def compute_log_probs(model, input_ids, prefix_indices):
    """
    计算给定序列的 Log Probability (对数概率)。
    """
    # 1. 通过 SoftEmbedding 获取向量 (加入前缀)
    # inputs_embeds shape: [batch, prefix_len + seq_len, hidden_dim]
    inputs_embeds = model.transformer.wte(input_ids, prefix_indices=prefix_indices)
    
    # 2. 前向传播
    outputs = model(inputs_embeds=inputs_embeds)
    logits = outputs.logits # [batch, prefix_len + seq_len, vocab]
    
    # 【关键修改】 3. 裁剪 Logits，去掉前缀部分
    # 我们只需要 input_ids 对应的 logits。
    # input_ids 的长度是 seq_len。
    # logits 的长度是 prefix_len + seq_len。
    # 所以我们需要取 logits 的最后 input_ids.size(1) 个时间步。
    
    seq_len = input_ids.size(1)
    # 取出与 input_ids 对应的部分 (即去掉前缀部分)
    # 注意：这里取的是后 seq_len 个
    logits = logits[:, -seq_len:, :] 
    
    # 4. Shift (错位): 用 t 时刻的 logits 预测 t+1 时刻的 token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # 5. 计算 CrossEntropy
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    
    # 此时 shift_logits 和 shift_labels 的维度应该完全匹配了
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1)
    ).view(input_ids.size(0), -1)
    
    seq_log_probs = -token_losses.sum(dim=1)
    
    return seq_log_probs

def generate_sequences(model, tokenizer, batch_size, device, max_len=60):
    """
    使用 Agent 模型生成一批序列。
    注意：这里我们只负责生成 Token ID，不计算梯度。
    梯度计算会在 update 步骤通过重新前向传播来实现（这是 PPO/REINFORCE 的标准做法）。
    """
    model.eval()
    
    # 构造输入 (BOS)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    input_tensor = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    
    # 构造前缀索引 (只优化 head_to_tail)
    prefix_indices = torch.full((batch_size,), 1, dtype=torch.long, device=device)
    
    finished = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        for _ in range(max_len):
            # SoftEmbedding
            inputs_embeds = model.transformer.wte(input_tensor, prefix_indices=prefix_indices)
            
            # Forward
            outputs = model(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Sampling (Multinomial)
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # EOS Check
            is_eos = (next_token == tokenizer.eos_token_id)
            finished = finished | is_eos
            
            # Update input
            input_tensor = torch.cat((input_tensor, next_token), dim=1)
            
            if finished.all():
                break
                
    return input_tensor # [batch, seq_len]

def decode_seqs(seqs, tokenizer):
    """解码为 SMILES 字符串"""
    smiles_list = []
    for seq in seqs:
        # Skip special tokens manually if needed, or use skip_special_tokens=True
        text = tokenizer.decode(seq, skip_special_tokens=True)
        # 去除空格
        text = text.replace(" ", "") 
        smiles_list.append(text)
    return smiles_list

# === 训练主循环 ===

def train_agent(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. 加载模型 (Agent & Prior)
    print("Loading models...")
    # Agent: 我们要训练的模型
    agent_module = ContrastivePrefixModule.load_from_checkpoint(args.ckpt_path, tokenizer=tokenizer)
    Agent = agent_module.model
    Agent.to(device)
    
    # Prior: 冻结的原始模型，作为 Baseline 和约束
    prior_module = ContrastivePrefixModule.load_from_checkpoint(args.ckpt_path, tokenizer=tokenizer)
    Prior = prior_module.model
    Prior.to(device)
    Prior.eval()
    for p in Prior.parameters(): p.requires_grad = False

    # 确保重参数化 (如果之前没做)
    if Agent.transformer.wte.learned_embedding is None:
        print("Reparameterizing Agent...")
        Agent.transformer.wte.reparameterize()
    if Prior.transformer.wte.learned_embedding is None:
        print("Reparameterizing Prior...")
        Prior.transformer.wte.reparameterize()

    # 3. 加载 Reward 模型 (透膜性预测器)
    print("Loading Permeability Predictor...")
    perm_model, perm_scalar = load_models()

    # 4. 优化器 (只优化 Agent 的 Soft Prompt 部分)
    # 如果你想微调整个 GPT2，就把 Agent.parameters() 放进去
    # 这里我们只微调 learned_embedding，保持轻量
    optimizer = AdamW([Agent.transformer.wte.learned_embedding], lr=args.lr)
    
    # 5. Experience Replay
    experience = SimpleExperience(max_size=500)
    # 定义非法分子的重罚分
    INVALID_PENALTY = -5.0
    
    print("Starting RL Training...")
    
    pbar = tqdm(range(args.n_steps), desc="RL Training")
    
    for step in pbar:
        # --- A. 采样 (Sampling) ---
        # Agent 生成一批序列
        seqs_tensor = generate_sequences(Agent, tokenizer, args.batch_size, device, max_len=args.max_len)
        
        # --- B. 计算 Reward ---
        smiles_list = decode_seqs(seqs_tensor, tokenizer)
        rewards = []
        valid_indices = [] # 记录哪些是合法的，用于经验回放
        # 定义惩罚分：必须显著低于正常得分区间(0~1)
        # 建议设为 -5.0 或 -10.0，让模型感到"痛苦"
        for i, smi in enumerate(smiles_list):
            score = predict_permeability_single(smi, perm_model, perm_scalar)
            if score is None:
                # 情况 A: 分子不合法 (Invalid SMILES)
                rewards.append(INVALID_PENALTY)
            else:
                # 情况 B: 分子合法
                rewards.append(score)
                valid_indices.append(i)
        rewards = np.array(rewards)
        rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
        validity_rate = len(valid_indices) / len(smiles_list)
        
        # --- C. 计算 Likelihoods ---
        # 我们只针对生成的 Head_to_Tail (idx=1) 进行计算
        prefix_indices = torch.full((len(seqs_tensor),), 1, dtype=torch.long, device=device)
        
        # Agent Log Probs (需要梯度!)
        Agent.train() # 开启 Dropout 等
        agent_log_probs = compute_log_probs(Agent, seqs_tensor, prefix_indices)
        
        # Prior Log Probs (不需要梯度)
        with torch.no_grad():
            prior_log_probs = compute_log_probs(Prior, seqs_tensor, prefix_indices)
            
        # --- D. 增强 Reward (Augmented Likelihood) ---
        # R_total = Prior_LogProb + sigma * Permeability_Score
        # 这相当于我们希望 Agent 生成的序列既像原始分布(Prior)，又有高分(Score)
        # score_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
        # augmented_log_probs = prior_log_probs + args.sigma * score_tensor
        target_log_probs = prior_log_probs + args.sigma * rewards_tensor
        loss = torch.pow((target_log_probs - agent_log_probs), 2)
        
        # Loss = (Augmented_LogProb - Agent_LogProb)^2
        # 这实际上是一种 Distillation (蒸馏) loss，逼近目标分布
        # 或者是传统的 REINFORCE: Loss = - (Reward * Agent_LogProb)
        
        # 这里沿用你参考代码中的思路 (MSE Loss for Augmented Likelihood)
        # loss = torch.pow((augmented_log_probs - agent_log_probs), 2)
        
        # --- E. Experience Replay ---
        if args.use_experience and len(experience.memory) > 4:
            exp_seqs_list, exp_score, exp_prior_lp = experience.sample(4)
            if len(exp_seqs_list) > 0:
                # Pad sequences to same length
                max_len_exp = max([len(s) for s in exp_seqs_list])
                exp_tensor = torch.full((len(exp_seqs_list), max_len_exp), tokenizer.pad_token_id, dtype=torch.long, device=device)
                for i, s in enumerate(exp_seqs_list):
                    exp_tensor[i, :len(s)] = s.clone().detach() # 这里要注意数据类型转换
                
                exp_score_t = torch.tensor(exp_score, device=device, dtype=torch.float32)
                exp_prior_lp_t = torch.tensor(exp_prior_lp, device=device, dtype=torch.float32)
                
                # Calculate Agent LogProb for experience
                exp_prefix_idx = torch.full((len(exp_seqs_list),), 1, dtype=torch.long, device=device)
                exp_agent_lp = compute_log_probs(Agent, exp_tensor, exp_prefix_idx)
                
                # Augmented
                exp_aug_lp = exp_prior_lp_t + args.sigma * exp_score_t
                exp_loss = torch.pow((exp_aug_lp - exp_agent_lp), 2)
                
                # Combine Loss
                loss = torch.cat((loss, exp_loss), 0)
        
        # 添加新经验
        # 注意要 detach 转为 CPU list 存入 buffer

        if len(valid_indices) > 0:
            # 只提取合法的子集
            valid_seqs_cpu = [seqs_tensor[idx].cpu() for idx in valid_indices]
            valid_rewards = rewards[valid_indices]
            # Prior LogProbs 也要对应提取
            valid_prior_lps = prior_log_probs[valid_indices].detach().cpu().numpy()
            
            experience.add(
                valid_seqs_cpu, 
                valid_rewards, 
                valid_prior_lps
            )
        
        # Average Loss
        final_loss = loss.mean()
        
        # --- F. Update ---
        optimizer.zero_grad()
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_([Agent.transformer.wte.learned_embedding], 1.0)
        optimizer.step()
        
        # --- Logging ---
        # if step % 1 == 0: # 每步都打，或者调大
        #     avg_score = rewards.mean()
        #     print(f"Step {step} | Loss: {final_loss.item():.4f} | Avg Perm Score: {avg_score:.4f}")
        #     # 打印一个 sample
        #     print(f"Sample: {smiles_list[0]} (Score: {rewards[0]:.4f})")
            
        # # Save Checkpoint
        # if step > 0 and step % 100 == 0:
        #     save_path = os.path.join(args.save_dir, f"agent_step_{step}.pt")
        #     torch.save(Agent.state_dict(), save_path)
        #     print(f"Saved agent to {save_path}")
        avg_reward = rewards.mean()
        
        pbar.set_postfix({
            "Loss": f"{final_loss.item():.4f}", 
            "Reward": f"{avg_reward:.4f}",
            "Valid%": f"{validity_rate:.1%}" 
        })
        
        # 每50步打印一次生成的分子示例
        if step % 50 == 0:
            # 找一个合法的打印，如果没有合法的就打印第一个
            print_idx = valid_indices[0] if len(valid_indices) > 0 else 0
            tqdm.write(f"\n[Step {step}] Sample: {smiles_list[print_idx]} (R: {rewards[print_idx]:.2f})")
            
        if step > 0 and step % args.save_every == 0:
            save_path = os.path.join(args.save_dir, f"agent_step_{step}.pt")
            torch.save(Agent.state_dict(), save_path)
            tqdm.write(f"Checkpoint saved to {save_path}")

    print("RL Training Complete.")
    torch.save(Agent.state_dict(), os.path.join(args.save_dir, "agent_final.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default="../output/best_model_prompt_pl/best-checkpoint-epoch=197-train_loss=61.2934.ckpt", help='Path to pre-trained .ckpt')
    # parser.add_argument('--ckpt_path', type=str, default="output/best_model_prompt_pl/best-checkpoint-epoch=197-train_loss=61.2934.ckpt", help='Path to pre-trained .ckpt')
    parser.add_argument('--tokenizer_path', type=str, default="jonghyunlee/MolGPT_pretrained-by-ZINC15")
    parser.add_argument('--save_dir', type=str, default="../output/rl_results")
    # parser.add_argument('--save_dir', type=str, default="output/rl_results")
    
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sigma', type=float, default=10.0, help='Weight for the reward in augmented likelihood')
    parser.add_argument('--max_len', type=int, default=60)
    parser.add_argument('--use_experience', action='store_true')
    parser.add_argument('--save_every', type=int, default=100)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    train_agent(args)