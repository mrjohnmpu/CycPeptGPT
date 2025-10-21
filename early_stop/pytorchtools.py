# _*_ encoding: utf-8 _*_
__author__ = 'wjk'
__date__ = '2019/12/18 16:07'
'''https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py'''

import numpy as np
import torch
import os


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self,patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        # self.model_name = model_name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = 100
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, optimizer, lr_scheduler, epoch, model_path, ckpt_path):

        # score = -val_loss
        score = val_loss


        if self.best_score == 100:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, lr_scheduler, epoch, model_path, ckpt_path)
            print('self.best_score:',self.best_score)

        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print('best_score:',self.best_score, 'now:',score)
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, lr_scheduler, epoch, model_path, ckpt_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer, lr_scheduler, epoch, model_path, ckpt_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        print('now best_score:', self.best_score)

        # --- 1. 保存最佳模型 (Hugging Face 格式, 用于推理) ---
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(model_path)

        # --- 2. 保存完整的 Checkpoint (torch 格式, 用于恢复训练) ---
        # ckpt_name 应该是一个文件名, 比如 './best_checkpoint.pt'
        ckpt_dir = os.path.dirname(ckpt_path)
        if ckpt_dir: # 确保目录不是空的
            os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
        }, ckpt_path)

        self.val_loss_min = val_loss