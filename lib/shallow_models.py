from collections import Counter
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, classification_report, plot_confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
import json

from lib.stopwords import stopwords as STOPWORDS
from lib.data_prep import (
    raw_X_train,
    raw_y_train,
    raw_X_test,
    raw_y_test,
    raw_X_val,
    raw_y_val,
    LABELS,
    create_dataset,
    LABEL_TO_IX,
    get_vocab,
    IX_TO_LABEL,
    train_dataset,
    val_dataset,
    tokenized_sentence,
)
from lib.ProcessEmbeddings import WordEmbeddings

WE = WordEmbeddings(vector_file="embeds/glove.6B/glove.6B.300d.txt")


def predict(clf, text):
    return clf.predict(
        WE.get_sentence_vector(text.lower().split(), vector_dict).reshape(1, -1)
    )[0]


raw_X = raw_X_test + raw_X_train


def to_one_hot(raw_y, LABELS):
    y = np.zeros((len(raw_y), len(LABELS)))
    for ix, datapoint in enumerate(raw_y):
        for label in datapoint:
            y[ix][LABEL_TO_IX[label]] = 1
    return y


def one_hot_to_LABELS(y):
    return ", ".join([IX_TO_LABEL[ix] for ix in np.where(y != 0)[0]])


def examine_prediction(datapoint, predicted):
    print(raw_X_test[datapoint])
    print()
    print("Predicted:")
    print(one_hot_to_LABELS(predicted[datapoint]))
    print()
    print("Actual:")
    print(one_hot_to_LABELS(y_test[datapoint]))


# Zooming in on specific pieces of data
# ix=440
# raw_X[ix]
# raw_y[ix]
# results_df[results_df['DATAPOINT'] == datapoint_ids[ix]][classes]


"""
LogisticRegression with WordEmbeddings
Hamming Loss: 0.29
F1: 0.32

With l2 pentalty: 81 out of 120 predictions are null
With no penalty: 21 out of 120 predictions are null
    - "None" loss decreases sparsity of coefficients, gives "more freedom" to the model.
    - Typically, penalty shrinks estimates of coefficients to avoid overfitting
    - L2 penalty: square root of the sum of the squared vector values (ridge regression)
    - L1 penalty: absolute values, rather than squared values. (lasso regression)
        - Tends to pick one variable at random when predictor variables are correlated.
    - ElasticNet combines L2 and L1 penalties
"""

y_train, y_val = to_one_hot(raw_y_train, LABELS), to_one_hot(raw_y_val, LABELS)
vector_dict = WE.get_vector_dict()

X_train_embeds, X_val_embeds = [
    WE.get_sentence_vector(tokenized_sentence(x), vector_dict, stopwords=STOPWORDS)
    for x in raw_X_train
], [
    WE.get_sentence_vector(tokenized_sentence(x), vector_dict, stopwords=STOPWORDS)
    for x in raw_X_val
]


lr_embed_clf = MultiOutputClassifier(
    LogisticRegression(
        max_iter=300, multi_class="multinomial", penalty="none", solver="lbfgs"
    )
).fit(X_train_embeds, y_train)
print(hamming_loss(y_val, lr_embed_clf.predict(X_val_embeds)))
print(classification_report(y_val, lr_embed_clf.predict(X_val_embeds)))
## Seeing where no prediction was made
null_predictions = len(
    [i for i in lr_embed_clf.predict(X_val_embeds) if not np.any(np.nonzero(i))]
)
print(f"{null_predictions} out of {len(y_val)} predictions were null.")

dub_ref_model = lr_embed_clf.estimators_[4]
vocab, id2tok, tok2id = get_vocab(train_dataset)
target_label = "dubious reference"
BATCH_SIZE = 1
pred = []
actual = []
vectors = []
for batch, targets, lengths, raw_data in create_dataset(
    val_dataset, id2tok, tok2id, target_label, batch_size=BATCH_SIZE
):
    actual.append(targets.item())
    pred.append(int(predict(dub_ref_model, raw_data[0])))
    vectors.append(WE.get_sentence_vector(raw_data[0].lower().split(), vector_dict))
print(classification_report(actual, pred))
plot_confusion_matrix(dub_ref_model, vectors, actual)


def analyze_sentence(label_ix, sent, stopwords):
    # Explaining with SHAP
    WE.task_data["fake_news"]["train_text"] = [x.lower().split() for x in raw_X_train]
    WE.task_data["fake_news"]["X_train"] = np.array(X_train_embeds)
    WE.task_data["fake_news"]["clf"] = lr_embed_clf.estimators_[label_ix]
    out = WE.analyze_sentence("fake_news", sent, stopwords=stopwords)
    print("1:")
    for pred in out[1]:
        print(pred)
    print("0:")
    for pred in out[0]:
        print(pred)


def explain_with_shap(label_ix, k=10):
    # Explaining with SHAP
    WE.task_data["fake_news"]["train_text"] = [x.lower().split() for x in raw_X_train]
    WE.task_data["fake_news"]["X_train"] = np.array(X_train_embeds)
    WE.task_data["fake_news"]["clf"] = lr_embed_clf.estimators_[label_ix]
    out = WE.top_ngrams_per_class(
        task="fake_news", clf=lr_embed_clf.estimators_[label_ix]
    )
    for class_ix, data in out.items():
        if class_ix == 1:
            print(f"Top words for {IX_TO_LABEL[class_ix]}:")
            for i in range(50):
                print(" ".join(data["ngrams"][i][0]))
            print()
            print()


datapoint = [ix for ix, x in enumerate(raw_y_train) if x == ["quantitative data"]][25]
raw_y_train[datapoint]
sent = raw_X_train[datapoint]
analyze_sentence(
    LABEL_TO_IX[raw_y_train[datapoint][0]], raw_X_train[datapoint], stopwords=STOPWORDS
)


"""
LogisticRegression with BOW
Hamming Loss: 0.21
F1: 0.29
"""
vectorizer = CountVectorizer(stop_words=STOPWORDS)
vectorizer = vectorizer.fit(raw_X)
X_train_count, X_val_count = vectorizer.transform(raw_X_train), vectorizer.transform(
    raw_X_val
)
lr_count_clf = MultiOutputClassifier(
    LogisticRegression(
        max_iter=200, multi_class="multinomial", penalty="none", solver="lbfgs"
    )
).fit(X_train_count, y_train)
print(hamming_loss(y_val, lr_count_clf.predict(X_val_count)))
print(classification_report(y_val, lr_count_clf.predict(X_val_count)))
## Seeing where no prediction was made
null_predictions = len(
    [i for i in lr_count_clf.predict(X_val_count) if not np.any(np.nonzero(i))]
)
print(f"{null_predictions} out of {len(y_val)} predictions were null.")

"""
NaiveBayes
F1: 0.31
Hamming Loss: 0.28
"""
mnb_count_clf = MultiOutputClassifier(MultinomialNB()).fit(X_train_count, y_train)
print(hamming_loss(y_val, mnb_count_clf.predict(X_val_count)))
print(classification_report(y_val, mnb_count_clf.predict(X_test_count)))


def label_breakdown(label_ix, k=10):
    print(IX_TO_LABEL[label_ix])
    print()
    top_indices = np.argsort(
        -mnb_count_clf.estimators_[label_ix].feature_log_prob_, axis=1
    )[:, :100]
    counts = Counter(np.ravel(top_indices))

    def get_unique(arr):
        return np.array([i for i in arr if counts[i] == 1])[:k]

    top_indices = np.array(
        [get_unique(top_indices[row, :]) for row in range(top_indices.shape[0])]
    )
    for class_ix in range(top_indices.shape[0]):
        print(f"Most influential words for {class_ix}:")
        for word_ix in top_indices[class_ix]:
            print(vectorizer.get_feature_names()[word_ix])
        print()
        print()


label_breakdown(2, k=20)
