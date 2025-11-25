import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from transformers import GPT2LMHeadModel, get_scheduler
from soft_prompt_embedding import SoftEmbedding


class ContrastivePrefixModule(pl.LightningModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer

        # 加载基础模型
        # 注意：在DDP模式下，每个进程都会加载一次模型，这是正常的。
        # 如果模型非常大，可以考虑使用 deepspeed 或 FSDP 策略。
        self.model = GPT2LMHeadModel.from_pretrained(args.model_path)

        # 替换 Embedding 层
        self.s_wte = SoftEmbedding(
            self.model.get_input_embeddings(),
            n_prefixes=args.n_prefixes,
            n_tokens=args.n_tokens,
            initialize_from_vocab=False,
            mid_dim=args.mid_dim
        )
        self.model.set_input_embeddings(self.s_wte)

        # 冻结模型参数逻辑
        self._freeze_parameters()

    def _freeze_parameters(self):
        """冻结不需要训练的参数，只开放前缀相关参数"""
        # 1. 先冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. 解冻 SoftEmbedding 中的参数
        # 注意：这里我们只处理重参数化模式，因为你的代码逻辑似乎总是使用重参数化
        if self.model.transformer.wte.learned_embedding is None:
            self.model.transformer.wte.input_tokens.requires_grad = True
            for param in self.model.transformer.wte.trans.parameters():
                param.requires_grad = True
            print("[Rank {}] Reparameterization mode: Unfrozen input_tokens and MLP.".format(self.global_rank))
        else:
            self.model.transformer.wte.learned_embedding.requires_grad = True
            print("[Rank {}] Standard mode: Unfrozen learned_embedding.".format(self.global_rank))

    def forward(self, input_ids, prefix_indices=None, attention_mask=None, labels=None, **kwargs):
        # 这里的 forward 主要用于推理或简单的 pass，
        # 但在 training_step 中我们会手动调用多次 model() 来实现对比损失

        # 如果使用 SoftEmbedding 的 forward 逻辑：
        # inputs_embeds = self.model.transformer.wte(input_ids, prefix_indices=prefix_indices)
        # return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)
        pass

    def training_step(self, batch, batch_idx):
        input_ids = batch[0]
        attention_mask = batch[1]
        labels_class = batch[3]  # 0 (Linear) 或 1 (Cyclic)

        batch_size = input_ids.shape[0]
        device = self.device

        # 2. 构造前缀索引
        idx_linear = torch.zeros(batch_size, dtype=torch.long, device=device)
        idx_cyclic = torch.ones(batch_size, dtype=torch.long, device=device)

        # 3. 获取 Embeddings
        # 显式调用 wte (SoftEmbedding)
        embeds_linear = self.model.transformer.wte(input_ids, prefix_indices=idx_linear)
        embeds_cyclic = self.model.transformer.wte(input_ids, prefix_indices=idx_cyclic)

        # 4. 修正 Attention Mask
        prefix_mask = torch.ones(batch_size, self.hparams.n_tokens, device=device)
        extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 5. 构造生成任务 Labels
        prefix_labels = torch.full((batch_size, self.hparams.n_tokens), -100, dtype=torch.long, device=device)
        lm_labels = input_ids.clone()
        lm_labels[attention_mask == 0] = -100
        extended_labels = torch.cat([prefix_labels, lm_labels], dim=1)

        # --- 前向传播 (Linear Prefix) ---
        outputs_linear = self.model(
            inputs_embeds=embeds_linear,
            attention_mask=extended_attention_mask,
            labels=extended_labels
        )

        # --- 前向传播 (Cyclic Prefix) ---
        outputs_cyclic = self.model(
            inputs_embeds=embeds_cyclic,
            attention_mask=extended_attention_mask,
            labels=extended_labels
        )

        # --- 计算 NLL ---
        # 辅助函数：计算每个样本的 NLL
        def compute_nll(logits, labels):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
            # [batch_size]
            return loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(batch_size, -1).sum(dim=1)

        nll_linear = compute_nll(outputs_linear.logits, extended_labels)
        nll_cyclic = compute_nll(outputs_cyclic.logits, extended_labels)

        # --- L_LM: 生成损失 ---
        loss_lm_vec = torch.where(labels_class == 0, nll_linear, nll_cyclic)
        loss_lm = loss_lm_vec.mean()

        # --- L_d: 判别损失 ---
        # nll_correct = torch.where(labels_class == 0, nll_linear, nll_cyclic) # 未使用，逻辑合并在 CrossEntropy 中
        log_probs_stack = torch.stack([-nll_linear, -nll_cyclic], dim=1)
        loss_d = CrossEntropyLoss()(log_probs_stack, labels_class)

        # 总 Loss
        loss = self.hparams.gen_weight * loss_lm + (1 - self.hparams.gen_weight) * loss_d

        # 记录日志 (prog_bar=True 会显示在进度条)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lm_loss', loss_lm, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('d_loss', loss_d, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # 定义优化器参数组
        if self.model.transformer.wte.learned_embedding is None:
            optimizer_params = [
                {'params': [self.model.transformer.wte.input_tokens]},
                {'params': self.model.transformer.wte.trans.parameters()}
            ]
        else:
            optimizer_params = [self.model.transformer.wte.learned_embedding]

        optimizer = AdamW(optimizer_params, lr=self.hparams.lr)

        # Scheduler
        # 注意：num_training_steps 在 Lightning 中通常需要自己计算或从 trainer 获取
        # 这里简单估算或者使用 step-based scheduler
        # Lightning 会自动处理 steps_per_epoch

        if self.trainer.max_steps == -1:
            # 如果是按 epoch 训练
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = self.trainer.max_steps

        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 每个 step 更新一次 lr
            },
        }

    def on_train_end(self):
        # 训练结束时的操作：重参数化
        # 只在主进程执行保存前的操作，但在 DDP 中模型是同步的，所以修改 self.model 状态需要小心
        # 不过 reparameterize 只是修改了内部结构，不涉及通信
        if self.global_rank == 0:
            print("Training finished. Collapsing reparameterization...")

        # 注意：这里直接修改了模型结构
        # 在 DDP 销毁前执行，确保保存 checkpoint 时是 reparameterized 状态
        # 但 Lightning 的 checkpoint saving 是在 on_train_end 之前还是之后是个问题
        # 通常 Lightning 会自动保存 best checkpoint。
        # 如果你想保存最终的 reparameterized 模型，最好在这里手动保存一个特定文件
        self.model.transformer.wte.reparameterize()

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm=None):
        # 自定义梯度裁剪逻辑
        # 因为我们需要裁剪特定的参数列表，而不是整个 model.parameters() (因为大部分是冻结的)

        if self.model.transformer.wte.learned_embedding is None:
            params_to_clip = [self.model.transformer.wte.input_tokens] + \
                             list(self.model.transformer.wte.trans.parameters())
        else:
            params_to_clip = [self.model.transformer.wte.learned_embedding]

        # Lightning 提供了手动裁剪的方法，但通常我们只需要传入 params 列表
        # 这里使用 pytorch 原生方法
        torch.nn.utils.clip_grad_norm_(params_to_clip, gradient_clip_val)