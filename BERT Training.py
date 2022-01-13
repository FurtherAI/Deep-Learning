import numpy as np
import torch as th
import torch.nn as nn
import pandas as pd
import nltk
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F

import transformers
from transformers import BertTokenizer
import random
import math

# Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018).
# Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.


# dataset (load data, normalize (lowercase), tokenize batch, organize examples for NSP and labels), datamodule
regex_tokenizer = nltk.RegexpTokenizer('\w+')

def normalize(text):
    text = str(text).lower()
    text = text.encode('utf-8', 'ignore').decode()
    text = ' '.join(regex_tokenizer.tokenize(text))
    return text


class SubtitleDataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()

        with open('En Subtitles/OpenSubtitles.af-en.en', encoding='utf-8') as fi:
            text = [*fi.readlines()]
            text = list(map(normalize, text))
            text = shuffle(text)

        text_pair = []
        is_next = []
        self.length = len(text)
        for i in range(self.length):
            choice = random.choice(['self', 'other'])
            if choice == 'self':
                idx = i + 1 if i != (self.length - 1) else i
                is_next.append(1)
            else:
                idx = random.randint(0, self.length - 1)
                is_next.append(0)
            text_pair.append(text[idx])

        self.data = tokenizer(text,
                              text_pair,
                              padding='max_length',
                              truncation=True,
                              max_length=100,
                              return_tensors='pt',
                              return_token_type_ids=True,
                              return_attention_mask=True
                              )
        self.data['is_next'] = th.LongTensor(is_next, device='cpu')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.data}


class SubtitleDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer

    def prepare_data(self):
        self.dataset = SubtitleDataset(self.tokenizer)

    def transfer_batch_to_device(self, batch, device):
        assert isinstance(batch, dict)
        return {key: batch[key].cuda() for key in batch}

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, drop_last=True)


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


class BERT(pl.LightningModule):
    def __init__(self, tokenizer, d_model, nheads, num_layers, d_ff, p_dropout, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.word_embedding = nn.Embedding(self.vocab_size, d_model)
        self.sequence_embedding = nn.Embedding(2, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, p_dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=d_ff, dropout=p_dropout, nhead=nheads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, self.vocab_size)
        self.nsp = nn.Linear(d_model, 2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        pass

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=2e-5)

    def training_step(self, batch, batch_idx):
        inp = batch['input_ids']
        nsp_labels = batch['is_next']
        type_ids = batch['token_type_ids']
        src_key_padding_mask = (batch['attention_mask'] == 0)

        # masking
        prob = th.full(inp.size(), .15, device='cuda')
        mask = (th.bernoulli(prob) == 0)
        mask_id = self.tokenizer.convert_tokens_to_ids('[MASK]')

        masked_inp = inp.where(mask, th.LongTensor([mask_id]).cuda())

        # embedding
        src = self.pos_enc(self.word_embedding(masked_inp.T) * math.sqrt(self.d_model) + self.sequence_embedding(type_ids.T))

        # forward
        out = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        out = out.permute(1, 0, 2)
        logits = self.out(out)
        nsp = self.nsp(out[:, 0, :].squeeze())

        # loss
        mlm_loss = self.criterion(logits[th.logical_not(mask)], inp[th.logical_not(mask)])
        nsp_loss = self.criterion(nsp, nsp_labels)
        return mlm_loss + nsp_loss

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    subtitles = SubtitleDataModule(tokenizer, batch_size=32)
    bert = BERT(tokenizer=tokenizer, d_model=256, nheads=8, num_layers=6, d_ff=1024, p_dropout=.1, max_seq_len=100)
    trainer = pl.Trainer(gpus=1, max_epochs=1)
    trainer.fit(bert, datamodule=subtitles)
