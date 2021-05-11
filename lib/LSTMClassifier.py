import numpy as np
import string
import os
import glob
import warnings
import json
from pathlib import Path
import statistics
import argparse
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import nltk
from nltk.stem.porter import *
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

warnings.filterwarnings("ignore", category=DeprecationWarning)


stemmer = PorterStemmer()
from lib.stopwords import stopwords as STOPWORDS
from lib.data_prep import (
    train_dataset,
    val_dataset,
    test_dataset,
    create_dataset,
    get_vocab,
    get_tok2id,
    LABELS,
    LABEL_TO_IX,
)
from lib.ProcessEmbeddings import WordEmbeddings
from tools.Blogger import Blogger

logger = Blogger()
#################################################
BATCH_SIZE = 1
DEVICE = "cpu"
vocab, id2tok, tok2id = get_vocab(train_dataset)
VOCAB_SIZE = len(vocab)
EMBED_DIM = 300
HIDDEN_SIZE = 32
ALPHA = 0.003
NUM_EPOCHS = 15
THRESHOLD = 0.4
POS_LOSS_WEIGHT = 1.5
DROPOUT = 0.6
USE_GLOVE = True
#################################################

if USE_GLOVE:
    WE = WordEmbeddings(vector_file="embeds/glove.6B/glove.6B.300d.txt")


def get_embed_weights(vocab, tok2id):
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (len(vocab), 300)), dtype="float32")
    vector_dict = WE.get_vector_dict()
    for tok in vocab:
        i = tok2id[tok]
        if tok in vector_dict:
            embeddings_matrix[i] = vector_dict[tok]
    return torch.from_numpy(embeddings_matrix)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, bidirectional, embed_weights=None):
        super(LSTMClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional

        if torch.is_tensor(embed_weights):
            logger.status_update("Loading custom weights...")
            self.embedding = nn.Embedding(vocab_size, embedding_dim, _weight=embed_weights)
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=bidirectional, batch_first=True)

        self.softmax = nn.LogSoftmax(dim=-1)
        self.dropout_layer = nn.Dropout(p=DROPOUT)

        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size):
        return (
            autograd.Variable(torch.randn(2 if self.bidirectional else 1, batch_size, self.hidden_dim)),
            autograd.Variable(torch.randn(2 if self.bidirectional else 1, batch_size, self.hidden_dim)),
        )

    def forward(self, batch, lengths):
        self.hidden = self.init_hidden(batch.size(-1))
        embeds = self.embedding(batch)  # embeds = (largest_seq, batch_size, embedding_dim)

        """
        We pack to make handling the pads in largest_seq more efficient
        when passing to the rnn
        """
        packed_input = pack_padded_sequence(
            embeds, lengths, enforce_sorted=False
        )  # packed_input = (sum(lengths), embedding_dim)
        """
        Difference between ht and output: https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
            - For depth 1 lstms, ht == output
        """
        packed_output, (ht, ct) = self.lstm(packed_input, self.hidden)  # packed_output = (sum(lengths), hidden_dim)
        lstm_out, _ = pad_packed_sequence(packed_output)  # lstm_out = (sum(lengths), batch_size, hidden_dim)

        output = self.dropout_layer(ht[-1])
        # output = self.max_pool(lstm_out)[-1] # (batch_size, hidden_dim)
        output = self.hidden2out(output)
        # output = (batch_size, output_dims)
        return output


def evaluate_validation_set(model, devset, id2tok, tok2id, TARGET_LABEL, loss_func, final=False):
    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in create_dataset(devset, id2tok, tok2id, TARGET_LABEL, batch_size=1):
        pred = model(batch.T, lengths)
        # loss = loss_func(pred, targets)
        # loss = loss_func(pred.type(torch.FloatTensor), targets.unsqueeze(0).type(torch.FloatTensor))
        loss = loss_func(pred.type(torch.FloatTensor), targets.unsqueeze(0).type(torch.FloatTensor))
        # loss = loss_func(pred, targets)
        y_true += list(targets.int())
        # pred_idx = torch.max(pred, 1)[1]
        # y_pred += list(pred_idx.data.int())
        y_pred += [int(pred.float() >= THRESHOLD)]
        total_loss += loss
    acc = accuracy_score(y_true, y_pred)
    print(confusion_matrix(y_true, y_pred))
    if final:
        print(classification_report(y_true, y_pred))
        return classification_report(y_true, y_pred, output_dict=True)
    return total_loss.data.float() / len(devset), acc, classification_report(y_true, y_pred, output_dict=True)


"""
https://github.com/claravania/lstm-pytorch
"""


def train(run_test=False):
    results = {}
    vocab, id2tok, tok2id = get_vocab(train_dataset)
    embed_weights = None
    if USE_GLOVE:
        embed_weights = get_embed_weights(vocab, tok2id)
    # from scipy.spatial import distance
    # print(distance.cosine(embed_weights[tok2id['obama']], embed_weights[tok2id['clinton']]))
    model = LSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_SIZE, bidirectional=False, embed_weights=embed_weights)
    loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([POS_LOSS_WEIGHT]))

    """
    Demo of weights in loss function.
    """
    # loss_func(torch.FloatTensor([0.6]), torch.FloatTensor([1]))
    # loss_func(torch.FloatTensor([0.6]), torch.FloatTensor([0]))
    if glob.glob("models/lstm/*"):
        model_id = max([int(i[-1]) for i in glob.glob("models/lstm/*")]) + 1
    else:
        model_id = 1

    optimizer = optim.Adam(model.parameters(), lr=ALPHA)
    all_best_f1 = []
    for label in LABELS:
        logger.green(f"Building classifier for {label}...")
        model.train()
        best_f1 = 0.0
        for epoch in range(NUM_EPOCHS):
            print()
            print(f"Epoch: {epoch}")
            y_true = list()
            y_pred = list()
            total_loss = 0
            for batch, targets, lengths, raw_data in create_dataset(
                train_dataset, id2tok, tok2id, label, batch_size=BATCH_SIZE
            ):
                pred = model(batch.T, lengths)
                loss = loss_func(pred.type(torch.FloatTensor), targets.unsqueeze(0).type(torch.FloatTensor))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                y_true += list(targets.int())
                # pred_idx = torch.max(pred, 1)[1]
                # y_pred += list(pred_idx.data.int())
                y_pred += [int(pred.float() >= THRESHOLD)]
                total_loss += loss
            acc = accuracy_score(y_true, y_pred)
            val_loss, val_acc, report = evaluate_validation_set(model, val_dataset, id2tok, tok2id, label, loss_func)
            print(
                "Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(
                    total_loss.data.float() / len(train_dataset), acc, val_loss, val_acc
                )
            )
            val_f1 = report["1"]["f1-score"]
            if best_f1 < val_f1:
                logger.green(f"New best F1 score at {val_f1}")
                best_f1 = val_f1
                if not os.path.exists(f"models/lstm/{model_id}/{LABEL_TO_IX[label]}"):
                    Path(f"models/lstm/{model_id}/{LABEL_TO_IX[label]}").mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), f"models/lstm/{model_id}/{LABEL_TO_IX[label]}/{LABEL_TO_IX[label]}.pt")
                results[label] = report
                if os.path.exists(f"models/lstm/{model_id}/{LABEL_TO_IX[label]}/results_{LABEL_TO_IX[label]}.json"):
                    os.remove(f"models/lstm/{model_id}/{LABEL_TO_IX[label]}/results_{LABEL_TO_IX[label]}.json")
                with open(f"models/lstm/{model_id}/{LABEL_TO_IX[label]}/results_{LABEL_TO_IX[label]}.json", "w") as f:
                    json.dump(results, f)
        all_best_f1.append(best_f1)
    logger.green(f"Final mean F1: {statistics.mean(all_best_f1)}")
    with open(f"models/lstm/{model_id}/summary.txt", "w") as f:
        f.write(f"Mean F1: {str(statistics.mean(all_best_f1))}\n")
        for ix, score in enumerate(all_best_f1):
            f.write(f"{ix}: {score} \n")
        f.write("\n")
        f.write(f"HIDDEN_SIZE: {HIDDEN_SIZE}\n")
        f.write(f"ALPHA: {ALPHA}\n")
        f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
        f.write(f"POS_LOSS_WEIGHT: {POS_LOSS_WEIGHT}\n")
        f.write(f"DROPOUT: {DROPOUT}\n")


if __name__ == "__main__":
    train()
