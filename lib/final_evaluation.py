import glob
from collections import defaultdict
import re
import ast
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
    IX_TO_LABEL,
    tokenized_sentence
)
from lib.ProcessEmbeddings import WordEmbeddings
from lib.MLPClassifier import MLPClassifier, add_features, WE, vector_dict
vocab, id2tok, tok2id = get_vocab(train_dataset)
vector_dict=WE.get_vector_dict()
BATCH_SIZE=1
from tools.Blogger import Blogger
logger = Blogger()
"""
Final evaluation of models on test set.
"""
def evaluate_best_model(pool_across_models=False):
    best_label_scores = defaultdict(int)
    best_label_paths = defaultdict(str)
    best_label_feat_info = defaultdict(list)
    dropout_info = defaultdict(float)
    for model_dir in glob.glob('models/mlp/*'):
        print(model_dir)
        with open(f'{model_dir}/summary.txt', 'r') as f:
            text = f.read()
            for match in re.finditer(r'\d: \d\.\d+', text):
                label = int(match.group().split(':')[0])
                score = float(match.group().split(':')[1])
                if best_label_scores[label] < score:
                    best_label_scores[label] = score
                    best_label_paths[label] = f'{model_dir}/{label}/{label}.pt'
                    best_label_feat_info[label] = []
                    if re.search(r'(?<=USE_FEATS: ).*]', text):
                        use_feats = ast.literal_eval(re.search(r'(?<=USE_FEATS: ).*]', text).group())
                        best_label_feat_info[label] = use_feats
                    if re.search(r'(?<=DROPOUT: )\d\.\d', text):
                        dropout_info[label] = float(re.search(r'(?<=DROPOUT: )\d\.\d', text).group())
    for label_ix in best_label_scores:
        model = MLPClassifier(302 if label_ix in best_label_feat_info[label_ix] else 300,
                              32,
                              dropout_info[label_ix])
        model.load_state_dict(torch.load(best_label_paths[label_ix]))
        model.eval()
        preds= []
        actual = []
        for batch, targets, lengths, raw_data in create_dataset(
            test_dataset, id2tok, tok2id, IX_TO_LABEL[label_ix], batch_size=BATCH_SIZE
        ):
            tokenized = tokenized_sentence(raw_data[0])
            vector = torch.tensor(WE.get_sentence_vector(tokenized, vector_dict))
            if label_ix in best_label_feat_info[label_ix]:
                vector = add_features(vector, raw_data[0], FEAT_ADD_SOFTENER=0.3)
            preds.append(int(model(vector.type(torch.FloatTensor)) > 0.4))
            actual.append(targets.item())
        print(IX_TO_LABEL[label_ix])
        print(classification_report(actual, preds))
        print(confusion_matrix(actual, preds))
        print()
