import nltk
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re
from lib.datasets import FNRDataSet, PaddedTensorDataset
from torch.utils.data import Dataset, DataLoader
import string
import torch

IX_TO_LABEL = {
    0: "contradictory quote",
    1: "exaggeration",
    2: "quantitative data",
    3: "evidence lacking",
    4: "dubious reference",
    5: "out of context",
    6: "qualitative data",
}
LABEL_TO_IX = {v: k for k, v in IX_TO_LABEL.items()}
LABELS = list(IX_TO_LABEL.values())


def clean_text(text):
    return re.sub(r"(\<p\>|\<\/p\>)", "", text)


def read_split_data():
    results_df = pd.read_csv("data/model_data/AnnotationResults.csv")
    with open("data/model_data/annotation_data.json", "r") as f:
        annotation_data = json.load(f)
    predict_labels = LABELS
    results_df = results_df.replace("FALSE", 0).replace("TRUE", 1)
    results_df = results_df.fillna(0)
    results_df[LABELS] = results_df[LABELS].astype("int32")

    # Constructing x, y
    raw_X = []
    raw_y = []
    datapoint_ids = []
    for datapoint in annotation_data:
        selection = results_df[results_df["DATAPOINT"] == int(datapoint)][LABELS]
        if len(selection) > 0:
            max_score = selection.sum(0).max()
            if max_score > 0:
                score_to_labels = defaultdict(list)
                for label, score in selection.sum(0).to_dict().items():
                    score_to_labels[score].append(label)
                correct_class = score_to_labels[max_score]
                # if correct_class in predict_labels:
                raw_y.append(correct_class)
                raw_X.append(clean_text(annotation_data[datapoint]["summary"]))
                datapoint_ids.append(int(datapoint))
    print(
        f"{len([i for i in raw_y if len(i) > 1])} out of {len(raw_y)} are multilabel."
    )

    # Converting to word embeddings, encoding labels
    label_to_ix = {c: ix for ix, c in enumerate(predict_labels)}
    ix_to_label = {ix: c for ix, c in enumerate(predict_labels)}

    return raw_X, raw_y


def vectorize(data, tok2id, unk_id=0):
    return [
        [
            tok2id[tok] if tok in tok2id else unk_id
            for tok in tokenized_sentence(sentence)
        ]
        for sentence, _ in data
    ]


def pad_sequences(vectorized_seqs, seq_lengths):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
    return seq_tensor


def clean_text(text):
    return re.sub(r"(\<p\>|\<\/p\>)", "", text)


def get_binary_labels(label, y):
    """
    Reformulates multilabel instances to binary, with respect to a given class.
    @param label (str): The label to restructure labels in respect to.
    @return binary_y_train (List[int]), binary_y_test (List[int]): Lists of 1, 0

    """
    return label in y


def tokenized_sentence(sentence):
    text = "".join([ch for ch in sentence if ch not in string.punctuation])
    tokens = [i.lower() for i in nltk.word_tokenize(sentence)]
    # stems = [stemmer.stem(i.lower()) for i in tokens]
    return tokens


def get_vocab(data):
    """
    @param X: List[str] of X data
    @return vocab, dicts id2tok, tok2id
    """
    vocab = set()
    for sentence, _ in data:
        for tok in tokenized_sentence(sentence):
            vocab.add(tok)
    return (
        vocab,
        {ix: tok for ix, tok in enumerate(vocab)},
        {tok: ix for ix, tok in enumerate(vocab)},
    )


def create_dataset(data, id2tok, tok2id, target_label, batch_size=4):
    # vocab, id2tok, tok2id = get_vocab(data)
    vocab_size = len(id2tok)
    vectorized_seqs = vectorize(data, tok2id)
    seq_lengths = torch.LongTensor([len(s) for s in vectorized_seqs])
    seq_tensor = pad_sequences(vectorized_seqs, seq_lengths)
    target_tensor = torch.LongTensor([target_label in y for _, y in data])
    raw_data = [x for x, _ in data]
    print(batch_size)
    return DataLoader(
        PaddedTensorDataset(seq_tensor, target_tensor, seq_lengths, raw_data),
        batch_size=batch_size,
    )


def get_tok2id(data):
    vocab, id2tok, tok2id = get_vocab(data)
    return vocab, tok2id


# Converting to word embeddings, encoding labels
label_to_ix = {c: ix for ix, c in enumerate(LABELS)}
ix_to_label = {ix: c for ix, c in enumerate(LABELS)}
raw_X, raw_y = read_split_data()

raw_X_train, raw_X_test_val, raw_y_train, raw_y_test_val = train_test_split(
    raw_X, raw_y, test_size=0.3, random_state=50
)

raw_X_test, raw_X_val, raw_y_test, raw_y_val = train_test_split(
    raw_X_test_val, raw_y_test_val, test_size=0.5, random_state=50
)


train_dataset = FNRDataSet(raw_X_train, raw_y_train)
print(f"{len(train_dataset)} in train set...")
val_dataset = FNRDataSet(raw_X_val, raw_y_val)
print(f"{len(val_dataset)} in validation set...")
test_dataset = FNRDataSet(raw_X_test, raw_y_test)
print(f"{len(test_dataset)} in test set...")
