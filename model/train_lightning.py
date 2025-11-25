import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import PreTrainedTokenizerFast

from prompt_lightning_module import ContrastivePrefixModule
from prompt_data_module import PromptDataModule


def setup_args():
    parser = argparse.ArgumentParser()
    # ... 保留原有的参数 ...
    parser.add_argument('--model_path',
                        default="/home/xiongshuwen/workingspace/cyc_gpt/output/best_model_with_mask_trainer/checkpoint-177198",
                        type=str)
    parser.add_argument('--best_model_dir', default="../output/best_model_prompt_pl", type=str)
    parser.add_argument('--train_raw_path', default='../data/filtered_peptides.csv', type=str)

    # 训练超参
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Total batch size will be batch_size * gpus * accum_steps')
    parser.add_argument('--accumulation_steps', default=1, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--warmup_steps', default=500, type=int)
    parser.add_argument('--max_len', default=576, type=int)
    parser.add_argument('--n_tokens', default=10, type=int)
    parser.add_argument('--n_prefixes', default=2, type=int)
    parser.add_argument('--mid_dim', default=512, type=int)
    parser.add_argument('--gen_weight', default=0.9, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)

    # 数据相关
    parser.add_argument("--balanced", default=True, type=bool)  # 注意 argparse bool 的坑，最好用 int 或 action='store_true'
    parser.add_argument('--sup_data_num', default=0, type=int)

    # 分布式相关
    parser.add_argument('--gpus', default=-1, type=int, help='number of gpus to use, -1 for all')
    parser.add_argument('--strategy', default='ddp', type=str, help='ddp or deepspeed')
    parser.add_argument('--seed', default=42, type=int)

    return parser.parse_args()


def main():
    args = setup_args()
    torch.set_float32_matmul_precision('medium')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pl.seed_everything(args.seed)

    # 1. Tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained("jonghyunlee/MolGPT_pretrained-by-ZINC15")
    tokenizer.model_max_length = args.max_len
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.unk_token is None:
        tokenizer.unk_token = "<unk>"

    # 2. DataModule
    dm = PromptDataModule(args, tokenizer)

    # 3. Model
    model = ContrastivePrefixModule(args, tokenizer)

    # 4. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.best_model_dir,
        filename='best-checkpoint-{epoch:02d}-{train_loss:.4f}',
        save_top_k=3,
        monitor='train_loss',  # 注意：这里如果有验证集最好 monitor val_loss，如果没有只能 monitor train_loss
        mode='min',
        save_weights_only=False  # 保存整个模型状态
    )

    # 这里的 EarlyStopping 监控 train_loss 可能不太稳定，建议划分一部分验证集
    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=0.0001,
        patience=10,
        verbose=True,
        mode='min'
    )

    # 5. Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.gpus,  # 使用所有GPU
        strategy=args.strategy,  # 'ddp' for multi-gpu
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accumulation_steps,
        gradient_clip_val=args.max_grad_norm,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision=16,  # 开启混合精度训练，加速且省显存
        log_every_n_steps=10
    )

    # 6. Start Training
    trainer.fit(model, datamodule=dm)

    # 7. 手动保存最终的 Reparameterized 模型
    # 注意：Lightning 自动保存的 checkpoint 包含了 optimizer 状态等。
    # 我们这里额外保存一个干净的用于推理的模型状态字典
    # 只在主进程保存
    if trainer.global_rank == 0:
        print("Saving final reparameterized model state dict...")
        # 确保 reparameterize 已被调用 (在 on_train_end 中)
        # 此时 model.s_wte 已经是普通的 Embedding 了
        final_save_path = os.path.join(args.best_model_dir, "final_reparameterized_model.pt")
        torch.save(model.model.state_dict(), final_save_path)
        print(f"Saved to {final_save_path}")


if __name__ == '__main__':
    main()