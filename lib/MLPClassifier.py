import os
from pathlib import Path
import warnings
import json
import glob
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import statistics
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore", category=DeprecationWarning)

from lib.stopwords import stopwords as STOPWORDS
from lib.data_prep import (
    train_dataset,
    val_dataset,
    test_dataset,
    create_dataset,
    get_vocab,
    get_tok2id,
    LABELS,
    tokenized_sentence,
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
NUM_EPOCHS = 20
THRESHOLD = 0.4
POS_LOSS_WEIGHT = 1.5
DROPOUT = 0.6
FEATS_TO_ADD = [
    "num_count",
    "year_count",
]  # Features added in the add_features function
FEAT_ADD_SOFTENER = 0.3
USE_FEATS = [
    1,
    2,
    3,
]  # Labels to apply the added features to (here, quantitative and qualitative data)
TFIDF_WEIGHTS = True  # Whether to take into consideration the tfidf matrix in weighing the sentence vectors
#################################################
WE = WordEmbeddings(vector_file="embeds/glove.6B/glove.6B.300d.txt")

if TFIDF_WEIGHTS:
    corpus = [i[0] for i in train_dataset]
    tf = TfidfVectorizer(tokenizer=tokenized_sentence)
    tfidf_matrix = tf.fit_transform(corpus)
    feature_names = tf.get_feature_names()
    train_doc_to_tfidf_ix = {doc[0]: ix for ix, doc in enumerate(train_dataset)}


def get_tfidf_vals(doc_id):
    feature_index = tfidf_matrix[doc_id, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc_id, x] for x in feature_index])
    out = {}
    for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
        out[w] = s
    return out


class MLPClassifier(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout):
        super(MLPClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.hidden_layer1 = nn.Linear(self.embedding_size, self.hidden_size)
        self.hidden_layer2 = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.hidden_layer3 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.output_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, text_vector):
        output = self.hidden_layer1(text_vector)
        output = self.hidden_layer2(output)
        output = self.dropout_layer(output)
        output = self.hidden_layer3(output)
        output = self.dropout_layer(output)
        output = self.output_layer(output)
        return output


def add_features(input_vector, raw_data, FEAT_ADD_SOFTENER=FEAT_ADD_SOFTENER):
    """
    hstacks new feature rows to existing input_vector.
    IDEAS:
        - Adjectives count
        - Numbers
        - Years
    """
    num_count = len(re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?(?!\d)", raw_data))
    year_count = len(re.findall(r"\d{4}", raw_data))
    # adj_count = 0
    # for token in nlp(raw_data):
    #     if token.pos_ == "ADJ":
    #         adj_count += 1
    all_feats = {
        "num_count": num_count * FEAT_ADD_SOFTENER,
        "year_count": year_count * FEAT_ADD_SOFTENER,
    }
    feats_to_add_vector = torch.FloatTensor(
        [all_feats[i] for i in FEATS_TO_ADD]
    ).unsqueeze(0)
    return torch.hstack((input_vector.unsqueeze(0), feats_to_add_vector)).view(-1)


def train():
    results = {}
    vocab, id2tok, tok2id = get_vocab(train_dataset)
    if glob.glob("models/mlp/*"):
        model_id = (
            max([int(re.search(r"\d+", i).group()) for i in glob.glob("models/mlp/*")])
            + 1
        )
    else:
        model_id = 1
    """
    Demo of weights in loss function.
    """
    # loss_func(torch.FloatTensor([0.6]), torch.FloatTensor([1]))
    # loss_func(torch.FloatTensor([0.6]), torch.FloatTensor([0]))
    vector_dict = WE.get_vector_dict()
    all_best_f1 = []
    for label in LABELS:
        if LABEL_TO_IX[label] in USE_FEATS:
            logger.yellow(f"Using additional features for label {label}...")
            model = MLPClassifier(EMBED_DIM + len(FEATS_TO_ADD), HIDDEN_SIZE, DROPOUT)
        else:
            model = MLPClassifier(EMBED_DIM, HIDDEN_SIZE, DROPOUT)
        loss_func = nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor([POS_LOSS_WEIGHT])
        )
        optimizer = optim.Adam(model.parameters(), lr=ALPHA)
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
                tokenized = tokenized_sentence(raw_data[0])
                sentence_weights = None
                if TFIDF_WEIGHTS:
                    sentence_weights = get_tfidf_vals(
                        train_doc_to_tfidf_ix[raw_data[0]]
                    )
                    sentence_weights = [
                        sentence_weights[tok] if tok in sentence_weights else 0
                        for tok in tokenized
                    ]
                input_vector = torch.FloatTensor(
                    WE.get_sentence_vector(
                        tokenized_sentence(raw_data[0]),
                        vector_dict=vector_dict,
                        weights=sentence_weights,
                    )
                )
                if LABEL_TO_IX[label] in USE_FEATS:
                    input_vector = add_features(input_vector, raw_data[0])
                pred = model(input_vector)
                loss = loss_func(
                    pred.type(torch.FloatTensor), targets.type(torch.FloatTensor)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                y_true += list(targets.int())
                y_pred += [int(pred.float() >= THRESHOLD)]
                total_loss += loss
            acc = accuracy_score(y_true, y_pred)
            val_loss, val_acc, report = evaluate_validation_set(
                model, val_dataset, id2tok, tok2id, label, loss_func, vector_dict
            )
            print(
                "Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(
                    total_loss.data.float() / len(train_dataset), acc, val_loss, val_acc
                )
            )
            val_f1 = report["1"]["f1-score"]
            if best_f1 < val_f1:
                logger.green(f"New best F1 score at {val_f1}")
                best_f1 = val_f1
                if not os.path.exists(f"models/mlp/{model_id}/{LABEL_TO_IX[label]}"):
                    Path(f"models/mlp/{model_id}/{LABEL_TO_IX[label]}").mkdir(
                        parents=True, exist_ok=True
                    )
                torch.save(
                    model.state_dict(),
                    f"models/mlp/{model_id}/{LABEL_TO_IX[label]}/{LABEL_TO_IX[label]}.pt",
                )
                results[label] = report
                if os.path.exists(
                    f"models/mlp/{model_id}/{LABEL_TO_IX[label]}/results_{LABEL_TO_IX[label]}.json"
                ):
                    os.remove(
                        f"models/mlp/{model_id}/{LABEL_TO_IX[label]}/results_{LABEL_TO_IX[label]}.json"
                    )
                with open(
                    f"models/mlp/{model_id}/{LABEL_TO_IX[label]}/results_{LABEL_TO_IX[label]}.json",
                    "w",
                ) as f:
                    json.dump(results, f)
        all_best_f1.append(best_f1)
    logger.green(f"Final mean F1: {statistics.mean(all_best_f1)}")
    with open(f"models/mlp/{model_id}/summary.txt", "w") as f:
        f.write(f"Mean F1: {str(statistics.mean(all_best_f1))}\n")
        for ix, score in enumerate(all_best_f1):
            f.write(f"{ix}: {score} \n")
        f.write("\n")
        f.write(f"HIDDEN_SIZE: {HIDDEN_SIZE}\n")
        f.write(f"ALPHA: {ALPHA}\n")
        f.write(f"NUM_EPOCHS: {NUM_EPOCHS}\n")
        f.write(f"POS_LOSS_WEIGHT: {POS_LOSS_WEIGHT}\n")
        f.write(f"DROPOUT: {DROPOUT}\n")
        f.write(f"TFIDF_WEIGHTS: {TFIDF_WEIGHTS}\n")
        if FEATS_TO_ADD:
            f.write(f"FEATS_TO_ADD: {FEATS_TO_ADD}\n")
            f.write(f"FEAT_ADD_SOFTENER: {FEAT_ADD_SOFTENER}\n")
        if USE_FEATS:
            f.write(f"USE_FEATS: {USE_FEATS}\n")
    mark_best_results()


def mark_best_results():
    best_score_dir = ""
    best_score = 0
    for model_dir in glob.glob("models/mlp/*"):
        for summary_file in glob.glob(f"{model_dir}/summary.txt"):
            with open(summary_file, "r") as f:
                score = float(f.readline().replace("Mean F1: ", "").replace("\n", ""))
            if score > best_score:
                best_score = score
                best_score_dir = model_dir
    if "*" not in best_score_dir:
        os.rename(best_score_dir, "/*".join(os.path.split(best_score_dir)))


def evaluate_validation_set(
    model, devset, id2tok, tok2id, label, loss_func, vector_dict, final=False
):
    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in create_dataset(
        devset, id2tok, tok2id, label, batch_size=1
    ):
        input_vector = torch.FloatTensor(
            WE.get_sentence_vector(
                tokenized_sentence(raw_data[0]), vector_dict=vector_dict
            )
        )
        if LABEL_TO_IX[label] in USE_FEATS:
            input_vector = add_features(input_vector, raw_data[0])
        pred = model(input_vector)
        loss = loss_func(pred.type(torch.FloatTensor), targets.type(torch.FloatTensor))
        y_true += list(targets.int())
        y_pred += [int(pred.float() >= THRESHOLD)]
        total_loss += loss
    acc = accuracy_score(y_true, y_pred)
    print(confusion_matrix(y_true, y_pred))
    if final:
        print(classification_report(y_true, y_pred))
        return classification_report(y_true, y_pred, output_dict=True)
    return (
        total_loss.data.float() / len(devset),
        acc,
        classification_report(y_true, y_pred, output_dict=True),
    )


if __name__ == "__main__":
    train()
