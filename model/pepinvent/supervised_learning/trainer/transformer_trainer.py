import os

import numpy as np
import torch

from pepinvent.supervised_learning.trainer.architecturedto import ArchitectureConfig
from pepinvent.supervised_learning.trainer.base_trainer import BaseTrainer
from pepinvent.supervised_learning.trainer.create_vocabulary import VocabularyMaker
from pepinvent.supervised_learning.utils.chem import tanimoto_similarity
from pepinvent.supervised_learning.utils.file import make_directory
from pepinvent.supervised_learning.utils.log import progress_bar
from pepinvent.supervised_learning.utils.torch_util import allocate_gpu
from reinvent_models.mol2mol.models.encode_decode.model import EncoderDecoder
from reinvent_models.mol2mol.models.module.decode import decode
from reinvent_models.mol2mol.models.module.label_smoothing import LabelSmoothing
from reinvent_models.mol2mol.models.module.noam_opt import NoamOpt
from reinvent_models.mol2mol.models.module.simpleloss_compute import SimpleLossCompute
from reinvent_models.mol2mol.models.vocabulary import SMILESTokenizer
from reinvent_models.mol2mol.mol2mol_model import Mol2MolModel


class TransformerTrainer(BaseTrainer):
    def __init__(self, opt: ArchitectureConfig):
        super().__init__(opt)

    def get_model(self, vocab) -> Mol2MolModel:
        vocab_size = len(vocab.tokens())
        if self._config.starting_epoch == 1:
            network = EncoderDecoder(vocab_size, num_layers=self._config.N, model_dimension=self._config.d_model,
                                   feedforward_dimension=self._config.d_ff, num_heads=self._config.H,
                                   dropout=self._config.dropout)
            model = Mol2MolModel(vocabulary=vocab, network=network,
                                 max_sequence_length=self._config.max_sequence_length,
                                 no_cuda=not self._config.use_cuda)
        else:
            file_name = os.path.join(self.save_path, f'checkpoint/model_{self._config.starting_epoch - 1}.ckpt')
            model = Mol2MolModel.load_from_file(file_name)

        return model

    def _initialize_optimizer(self, model):
        optim = NoamOpt(model.src_embed[0].d_model, self._config.factor, self._config.warmup_steps,
                        torch.optim.Adam(model.parameters(), lr=0,
                                         betas=(self._config.adam_beta1, self._config.adam_beta2),
                                         eps=self._config.adam_eps))
        return optim

    def _load_optimizer_from_epoch(self, model, file_name):
        # load optimization
        checkpoint = torch.load(file_name, map_location='cuda:0')
        optim_dict = checkpoint['optimizer_state_dict']
        optim = NoamOpt(optim_dict['model_size'], optim_dict['factor'], optim_dict['warmup'],
                        torch.optim.Adam(model.parameters(), lr=0))
        optim.load_state_dict(optim_dict)
        return optim

    def get_optimization(self, model):
        if self._config.starting_epoch == 1:
            optim = self._initialize_optimizer(model)
        else:
            # load optimization
            file_name = os.path.join(self.save_path, f'checkpoint/optimizer_{self._config.starting_epoch - 1}.ckpt')
            optim = self._load_optimizer_from_epoch(model, file_name)
        return optim

    def execute(self):
        # Load vocabulary
        vocabulary_maker = VocabularyMaker()
        vocab = vocabulary_maker.create_vocabulary(self._config.training_data_path, self._config.validation_data_path)
        vocab_size = len(vocab.tokens())

        # Data loader
        dataloader_train = self.initialize_dataloader(self._config.training_data_path, self._config.batch_size, vocab)
        dataloader_validation = self.initialize_dataloader(self._config.validation_data_path, self._config.batch_size, vocab)

        device = allocate_gpu()

        model = self.get_model(vocab)
        optimization = self.get_optimization(model.network)

        pad_idx = self._config.padding_value
        criterion = LabelSmoothing(size=len(vocab), padding_idx=pad_idx, smoothing=self._config.label_smoothing)

        # Train epoch
        for epoch in range(self._config.starting_epoch, self._config.starting_epoch + self._config.num_epoch):
            self.LOG.info("Starting EPOCH #%d", epoch)

            self.LOG.info("Training start")
            model.network.train()
            loss_epoch_train = self.train_epoch(dataloader_train, model.network,
                                                SimpleLossCompute(model.network.generator, criterion, optimization), device)

            self.LOG.info("Training end")
            self.save(model, optimization, epoch, vocab_size)

            self.LOG.info("Validation start")
            model.network.eval()

            loss_epoch_validation, accuracy, token_accuracy, similarities, nlls = self.validation_stat(
                dataloader_validation, model.network,
                SimpleLossCompute(model.network.generator, criterion, None),
                device, vocab)
            self.LOG.info("Validation end")
            self.LOG.info(
                "Train loss, Validation loss, identity_accuracy, token_accuracy, sim_avg: {}, {}, {}, {}, {}".format(
                    round(loss_epoch_train, 5), round(loss_epoch_validation, 5),
                    round(accuracy, 5), round(token_accuracy, 5), round(similarities, 5)))
            self.LOG.info("Mean NLL: {}".format(np.mean(nlls)))
            self.to_tensorboard(loss_epoch_train, loss_epoch_validation, accuracy, token_accuracy, similarities, epoch,
                                np.mean(nlls))

    def train_epoch(self, dataloader, model, loss_compute, device):
        pad = self._config.padding_value
        total_loss, total_tokens = 0.0, 0.0
        for i, batch in enumerate(progress_bar(dataloader, total=len(dataloader))):
            src, source_length, trg, src_mask, trg_mask, _, _ = batch
            trg_y = trg[:, 1:].to(device)  # skip start token

            # number of tokens without padding
            ntokens = float((trg_y != pad).data.sum())

            # Move to GPU
            src = src.to(device)
            trg = trg[:, :-1].to(device)  # save start token, skip end token
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)

            # Compute loss
            out = model.forward(src, trg, src_mask, trg_mask)
            loss = loss_compute(out, trg_y, ntokens)
            total_tokens += ntokens
            total_loss += float(loss)

        loss_epoch = total_loss / total_tokens
        return loss_epoch

    def _get_model_parameters(self, vocab_size):
        return {
            'vocab_size': vocab_size,
            'N': self._config.N,
            'd_model': self._config.d_model,
            'd_ff': self._config.d_ff,
            'H': self._config.H,
            'dropout': self._config.dropout
        }

    def save(self, model: Mol2MolModel, optim, epoch, vocab_size):
        """
        Saves the model, optimizer and model hyperparameters
        """
        save_dict = {
            'optimizer_state_dict': optim.save_state_dict(),
        }

        file_name = os.path.join(self.save_path, f'checkpoint/optimizer_{epoch}.ckpt')
        make_directory(file_name, is_dir=False)
        torch.save(save_dict, file_name)

        file_name = os.path.join(self.save_path, f'checkpoint/model_{epoch}.ckpt')
        model.save_to_file(file_name)

    def validation_stat(self, dataloader, model, loss_compute, device, vocab):
        pad = self._config.padding_value
        total_loss, total_n_trg, total_tokens, n_correct, n_correct_token = 0, 0, 0, 0, 0
        similarities, nll_list = [], []
        tokenizer = SMILESTokenizer()

        for i, batch in enumerate(progress_bar(dataloader, total=len(dataloader))):
            src, source_length, trg, src_mask, trg_mask, _, _ = batch
            trg_y = trg[:, 1:].to(device)  # skip start token

            # number of tokens without padding
            ntokens = float((trg_y != pad).data.sum())

            # Move to GPU
            src = src.to(device)
            trg = trg[:, :-1].to(device)  # save start token, skip end token
            src_mask = src_mask.to(device)
            trg_mask = trg_mask.to(device)

            with torch.no_grad():
                # Compute loss with teaching forcing
                out = model.forward(src, trg, src_mask, trg_mask)
                loss = loss_compute(out, trg_y, ntokens)
                total_loss += float(loss)
                total_tokens += ntokens
                # Decode
                max_length_target = self._config.max_sequence_length
                smiles, nlls = decode(model, src, src_mask, max_length_target, device, decode_type='greedy')
                nll_list.append(torch.mean(nlls).cpu().numpy())

                # Compute accuracy_harsh, accuracy_smooth
                for j in range(trg.size()[0]):
                    seq = smiles[j, :]
                    target = trg[j]
                    target_tokens = target.cpu().numpy()
                    seq_tokens = seq.cpu().numpy()

                    target = tokenizer.untokenize(vocab.decode(target_tokens))
                    seq = tokenizer.untokenize(vocab.decode(seq_tokens))
                    if seq == target:
                        n_correct += 1

                    # token accuracy
                    for k in range(len(target)):
                        if k < len(seq) and seq[k] == target[k]:
                            n_correct_token += 1

                    start_ind = 0
                    source_seq = tokenizer.untokenize(vocab.decode(src[j].cpu().numpy()[start_ind:]))
                    sim = tanimoto_similarity(seq, source_seq)
                    if sim:
                        similarities.append(sim)

            # number of samples in current batch
            n_trg = trg.size()[0]
            # total samples
            total_n_trg += n_trg

        # Accuracy
        accuracy_harsh = n_correct * 1.0 / total_n_trg
        loss_epoch = total_loss / total_tokens
        token_accuracy = n_correct_token / total_tokens

        sim_avg = 0
        if len(similarities) > 0:
            sim_avg = sum(similarities) / len(similarities)
        return loss_epoch, accuracy_harsh, token_accuracy, sim_avg, nll_list
