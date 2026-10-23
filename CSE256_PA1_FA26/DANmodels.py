# DANmodels.py

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sentiment_data import read_sentiment_examples, read_word_embeddings
from utils import Indexer


PAD_IDX = 0
UNK_IDX = 1


# Dataset class for DAN models: each example becomes a list of token indices
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, encode):
        """
        :param infile: path to a sentiment data file
        :param encode: function mapping a list of words to a list of token indices
        """
        self.examples = read_sentiment_examples(infile)
        self.indices = [torch.tensor(encode(ex.words), dtype=torch.long) for ex in self.examples]
        self.labels = torch.tensor([ex.label for ex in self.examples], dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.indices[idx], self.labels[idx]


def collate_dan(batch):
    """
    Pads a batch of variable-length index sequences with PAD_IDX so they can be stacked into one tensor.
    """
    seqs, labels = zip(*batch)
    return pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX), torch.stack(labels)


def word_encoder(indexer):
    """
    :return: function mapping a list of words to indices in the given Indexer, using UNK for unknown words
    """
    def encode(words):
        idxs = [indexer.index_of(w) for w in words]
        return [i if i != -1 else UNK_IDX for i in idxs]
    return encode


def build_train_vocab(infile):
    """
    Builds a word Indexer (with PAD at 0 and UNK at 1) from the words in the training file. Used when the
    embeddings are randomly initialized, so that words never seen in training map to UNK rather than to an
    untrained random vector.
    """
    indexer = Indexer()
    indexer.add_and_get_index("PAD")
    indexer.add_and_get_index("UNK")
    for ex in read_sentiment_examples(infile):
        for w in ex.words:
            indexer.add_and_get_index(w)
    return indexer


# Deep Averaging Network (Iyyer et al., 2015)
class DAN(nn.Module):
    def __init__(self, embedding, hidden_size=300, num_layers=2, dropout=0.3, word_dropout=0.0):
        """
        :param embedding: nn.Embedding layer (pretrained or randomly initialized)
        :param hidden_size: size of each hidden layer
        :param num_layers: number of hidden (Linear + ReLU) layers before the output layer
        :param dropout: dropout probability applied to the averaged embedding and after each hidden layer
        :param word_dropout: probability of dropping each token from the average during training
        """
        super().__init__()
        self.embedding = embedding
        self.word_dropout = word_dropout
        self.dropout = nn.Dropout(dropout)

        layers = []
        in_size = embedding.embedding_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            in_size = hidden_size
        self.hidden = nn.ModuleList(layers)
        self.out = nn.Linear(in_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x: (batch, seq_len) token indices, padded with PAD_IDX
        mask = (x != PAD_IDX).float()
        if self.training and self.word_dropout > 0:
            keep = (torch.rand_like(mask) >= self.word_dropout).float()
            # Never drop every token of a sentence
            keep[(mask * keep).sum(1) == 0] = 1.0
            mask = mask * keep

        emb = self.embedding(x)                                   # (batch, seq_len, dim)
        summed = (emb * mask.unsqueeze(-1)).sum(1)                # (batch, dim)
        avg = summed / mask.sum(1, keepdim=True).clamp(min=1.0)   # average over real tokens only

        h = self.dropout(avg)
        for layer in self.hidden:
            h = self.dropout(F.relu(layer(h)))
        return self.log_softmax(self.out(h))


def load_glove(dim):
    return read_word_embeddings(f"data/glove.6B.{dim}d-relativized.txt")


def random_embedding(vocab_size, dim):
    return nn.Embedding(vocab_size, dim, padding_idx=PAD_IDX)
