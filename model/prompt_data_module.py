import csv
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, RandomSampler


class PromptDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = None

    def setup(self, stage=None):
        # 类似于原本的 load_and_cache_examples
        # 在 DDP 模式下，setup 会在每个 GPU 上运行
        # 建议数据预处理逻辑要高效，或者预先处理好存成 pt 文件

        if self.train_dataset is not None:
            return

        filepath = self.args.train_raw_path
        data_pos = []
        data_neg = []
        data = []
        data_taski = {}

        print(f"Loading data from: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = row['SMILES']
                example_id = str(i)
                cyclization_type = row['Cyclization']
                label = 0 if cyclization_type == 'linear' else 1
                example = [text, example_id, label]

                if self.args.sup_data_num <= 0:
                    if not self.args.balanced:
                        data.append(example)
                    else:
                        if label == 1:
                            data_pos.append(example)
                        else:
                            data_neg.append(example)
                else:
                    label_str = str(label)
                    if label_str not in data_taski: data_taski[label_str] = []
                    data_taski[label_str].append([text, example_id, label, label])

        # 简单的平衡逻辑
        if self.args.sup_data_num <= 0 and self.args.balanced:
            if len(data_pos) > len(data_neg):
                data_neg_expand = data_neg * (len(data_pos) // len(data_neg))
                data = data_pos + data_neg_expand + random.sample(data_neg, len(data_pos) - len(data_neg_expand))
            elif len(data_neg) > len(data_pos):
                data_pos_expand = data_pos * (len(data_neg) // len(data_pos))
                data = data_neg + data_pos_expand + random.sample(data_pos, len(data_neg) - len(data_pos_expand))
            else:
                data = data_neg + data_pos
        elif self.args.sup_data_num > 0:
            for label_key in data_taski.keys():
                if len(data_taski[label_key]) > self.args.sup_data_num:
                    add_data = random.sample(data_taski[label_key], self.args.sup_data_num)
                else:
                    add_data = data_taski[label_key]
                for ex in add_data:
                    data.append([ex[0], ex[1], ex[2]])

        # Tokenization
        if self.args.max_len is None:
            max_length = self.tokenizer.model_max_length - self.args.n_tokens
        else:
            max_length = self.args.max_len - self.args.n_tokens

        batch_encoding = self.tokenizer(
            [example[0] for example in data],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )

        all_input_ids = torch.tensor(batch_encoding['input_ids'], dtype=torch.long)
        all_attention_mask = torch.tensor(batch_encoding['attention_mask'], dtype=torch.long)
        all_token_type_ids = torch.tensor(batch_encoding['token_type_ids'], dtype=torch.long)
        all_labels = torch.tensor([int(example[2]) for example in data], dtype=torch.long)

        self.train_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
        print(f"Dataset created with {len(self.train_dataset)} examples.")

    def train_dataloader(self):
        # Lightning 在 DDP 下会自动处理 DistributedSampler
        # 所以这里只需要返回普通的 DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,  # 训练时开启 shuffle
            num_workers=4,  # 适当增加 worker
            pin_memory=True
        )