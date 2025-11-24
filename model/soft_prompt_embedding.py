import torch
import torch.nn as nn


class SoftEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_prefixes: int = 2,
                 n_tokens: int = 10,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = False, # 重参数化通常建议从随机开始，因为很难逆向初始化MLP
                 mid_dim: int = 512): # 中间维度，通常比 GPT2 的 hidden_size 小
        """appends learned embedding to
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_prefixes = n_prefixes
        self.n_tokens = n_tokens
        self.hidden_size = wte.weight.size(1)
        self.mid_dim = mid_dim

        # 【修改点 1：定义小矩阵 H'】
        # 这里的参数量更少，形状为 [N, M, mid_dim]
        # 论文中称之为 "smaller parameter"
        self.input_tokens = nn.Parameter(
            torch.randn(n_prefixes, n_tokens, mid_dim)
        )

        # 【修改点 2：定义映射网络 W (MLP)】
        # 论文中称之为 "large matrix W"
        # 结构通常为 Linear -> Tanh -> Linear
        self.trans = nn.Sequential(
            nn.Linear(mid_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        # 最终的 embedding (推理时使用)，训练时为 None
        self.learned_embedding = None

        # # 初始化参数: shape [n_prefixes, n_tokens, hidden_size] ==> [N, M, D]
        # self.learned_embedding = nn.parameter.Parameter(
        #     self.initialize_embedding(wte, n_prefixes, n_tokens, random_range, initialize_from_vocab)
        # )

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_prefixes: int = 2,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            # return self.wte.weight[:n_tokens].clone().detach()
            # 简单策略：从词表中复制前 n_tokens 个词，并复制 N 份作为初始值
            # shape: [n_tokens, dim] -> [1, n_tokens, dim] -> [n_prefixes, n_tokens, dim]
            return wte.weight[:n_tokens].clone().detach().unsqueeze(0).repeat(n_prefixes, 1, 1)
        # return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
        # 随机初始化：直接创建一个 [N, M, D] 的随机矩阵
        return torch.FloatTensor(n_prefixes, n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    # def forward(self, tokens):
    #     """run forward pass
    #     Args:
    #         tokens (torch.long): input tokens before encoding
    #     Returns:
    #         torch.float: encoding of text concatenated with learned task specifc embedding
    #     """
    #     input_embedding = self.wte(tokens[:, self.n_tokens:])
    #     learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
    #     return torch.cat([learned_embedding, input_embedding], 1)
    # def forward(self, tokens, prefix_indices=None):
    #     """
    #     Args:
    #         tokens: input tokens (batch, seq_len) - 这里的tokens应当不包含为了占位用的dummy prefix
    #         prefix_indices: (batch,) 指定每个样本使用哪个前缀 (0 或 1)
    #     """
    #     # 获取原始文本的 embedding
    #     input_embedding = self.wte(tokens)
    #     batch_size = tokens.size(0)
    #
    #     if prefix_indices is None:
    #         # 默认使用第一个前缀 (预测时如果未指定)
    #         prefix_indices = torch.zeros(batch_size, dtype=torch.long, device=tokens.device)
    #
    #     # 取出对应的 soft prompt [batch, n_tokens, hidden]
    #     # self.learned_embedding: [n_prefixes, n_tokens, hidden]
    #     prompts = self.learned_embedding[prefix_indices]
    #
    #     # 拼接: [Prefix_Embeds, Word_Embeds]
    #     return torch.cat([prompts, input_embedding], 1)

    def forward(self, tokens, prefix_indices=None):
        """
        Args:
            tokens: input tokens (batch, seq_len) - 这里的tokens应当不包含为了占位用的dummy prefix
            prefix_indices: (batch,) 指定每个样本使用哪个前缀 (0 或 1)
        """
        # 获取原始文本的 embedding
        input_embedding = self.wte(tokens)
        batch_size = tokens.size(0)

        if prefix_indices is None:
            # 默认使用第一个前缀 (预测时如果未指定)
            prefix_indices = torch.zeros(batch_size, dtype=torch.long, device=tokens.device)

        # 【修改点 3：实时计算高维前缀】
        if self.learned_embedding is None:
            # 训练阶段：H_theta = MLP(H_prime)
            # 先通过 MLP 将 [N, M, mid_dim] 映射为 [N, M, hidden_size]
            prefix_embedding = self.trans(self.input_tokens)
        else:
            # 推理阶段：直接使用固定好的 Parameter (已丢弃 MLP)
            prefix_embedding = self.learned_embedding

        # 取出对应的 soft prompt [batch, n_tokens, hidden]
        # self.learned_embedding: [n_prefixes, n_tokens, hidden]
        prompts = prefix_embedding[prefix_indices]

        # 拼接: [Prefix_Embeds, Word_Embeds]
        return torch.cat([prompts, input_embedding], 1)

    def reparameterize(self):
        """
        【修改点 4：训练结束后的“烘焙”操作】
        论文提到：Training finishes, only H_theta needs to be saved, W and H' can be discarded.
        """
        # 1. 计算最终的高维矩阵
        with torch.no_grad():
            full_embedding = self.trans(self.input_tokens)  # [N, M, D]

        # 2. 将其保存为普通的 nn.Parameter
        self.learned_embedding = nn.Parameter(full_embedding.detach())

        # 3. 删除 MLP 和 H' 以节省显存和模型大小
        del self.trans
        del self.input_tokens

        print("Reparameterization complete. MLP discarded.")