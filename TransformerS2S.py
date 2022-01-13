import numpy as np
import torch as th
import torch.nn as nn
import spacy
import pytorch_lightning as pl
import torch.nn.functional as F

from torchtext.data import Field
from torchtext.datasets import WikiText2, IMDB, WMT14
from torchtext.vocab import FastText
from torchtext.data import BucketIterator

import math

# Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017).
# Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

# @inproceedings{opennmt,
#   author    = {Guillaume Klein and
#                Yoon Kim and
#                Yuntian Deng and
#                Jean Senellart and
#                Alexander M. Rush},
#   title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
#   booktitle = {Proc. ACL},
#   year      = {2017},
#   url       = {https://doi.org/10.18653/v1/P17-4012},
#   doi       = {10.18653/v1/P17-4012}
# }

class WikiText2DataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.text_field = Field(sequential=True, tokenize='spacy', init_token='<sos>', eos_token='<eos>', include_lengths=True, fix_length=None)
            train, val = WikiText2.splits(self.text_field, root='WikiText2', test=None)
            self.text_field.build_vocab(train, val)
            self.train, self.val = BucketIterator.splits((train, val), batch_size=self.batch_size, device='cuda')

    def train_dataloader(self):
        return self.train

    def val_dataloader(self):
        return self.val


class IMDBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.text_field = Field(sequential=True, tokenize='spacy', init_token='<sos>', eos_token='<eos>', include_lengths=True, fix_length=200)
            self.label_field = Field(sequential=False)

            train, test = IMDB.splits(self.text_field, self.label_field, root='IMDB', train='train', test='test')
            self.text_field.build_vocab(train, test)
            self.label_field.build_vocab(train)

            self.train_iter, self.test_iter = BucketIterator.splits((train, test), batch_size=self.batch_size, device='cuda')

    def train_dataloader(self):
        return self.train_iter

    def test_dataloader(self):
        return self.test_iter


# class WMT14DataModule(pl.LightningDataModule):
#     def __init__(self, batch_size):
#         super().__init__()
#         self.batch_size = batch_size
#
#     def setup(self, stage=None):
#         if stage == 'fit' or stage is None:
#             self.en_field = Field(sequential=True, tokenize='spacy', init_token='<sos>', eos_token='<eos>', include_lengths=True, fix_length=100)
#             self.de_field = Field(sequential=True, tokenize='spacy', init_token='<sos>', eos_token='<eos>', include_lengths=True, fix_length=100)
#             train = WMT14.splits(('.en', '.de'), (self.en_field, self.de_field), root='WMT14', test=None, validation=None)
#             print('past loading')
#             self.en_field.build_vocab(train)
#             self.de_field.build_vocab(train)
#             print(self.en_field.vocab.stoi['das'], 'didnt raise error?')
#             self.train = BucketIterator.splits((train), batch_size=self.batch_size, device='cuda')
#
#     def train_dataloader(self):
#         return self.train


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(seq_len, d_model, device='cuda')
        position = th.arange(0, seq_len, dtype=th.float, device='cuda').unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2, device='cuda').float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0), :])


class Transformer(pl.LightningModule):
    def __init__(self, vocab_size, d_model, nheads, num_encoder_layers, num_decoder_layers, d_ff, p_dropout, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, p_dropout)

        self.transformer = nn.Transformer(d_model, nheads, num_encoder_layers, num_decoder_layers, d_ff, p_dropout)
        self.out = nn.Linear(d_model, vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters())

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, tgt_mask):
        pass

    def training_step(self, batch, batch_idx):
        src = batch.text[0]
        tgt = src[:-1, :]
        y = src[1:, :].transpose(0, 1)

        src_key_padding_mask = (src == 1).T
        tgt_key_padding_mask = (tgt == 1).T
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).cuda()

        src = self.pos_enc(self.embedding(src) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embedding(tgt) * math.sqrt(self.d_model))

        out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)

        logits = self.out(out.permute(1, 0, 2))
        loss = self.criterion(logits.permute(0, 2, 1), y)
        return loss

if __name__ == '__main__':
    imdb = IMDBDataModule(batch_size=8)
    imdb.setup()
    vocab_size = len(imdb.text_field.vocab)

    model = Transformer(vocab_size, d_model=256, nheads=8, num_encoder_layers=2, num_decoder_layers=2, d_ff=1024, p_dropout=.1, max_seq_len=200)
    trainer = pl.Trainer(gpus=1, max_epochs=3)
    trainer.fit(model, imdb.train_dataloader())
