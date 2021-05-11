import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from lib.data_prep import (
    ix_to_label,
    label_to_ix,
    tokenized_sentence,
    LABEL_TO_IX,
    IX_TO_LABEL,
    get_vocab,
    val_dataset,
    LABELS,
    raw_y_train,
    raw_y_val,
    raw_y_test,
    create_dataset,
    train_dataset,
)
from sklearn.decomposition import PCA
from lib.ProcessEmbeddings import WordEmbeddings
import re
import pandas as pd
import statistics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.graph_objects as go

raw_y = raw_y_train + raw_y_val + raw_y_test


"""
Decision boundary
"""
import torch
from lib.MLPClassifier import MLPClassifier, add_features, WE

target_label = "quantitative data"
vector_dict = WE.get_vector_dict()
vocab, id2tok, tok2id = get_vocab(train_dataset)
BATCH_SIZE = 1


def compare_pca_features(target_label, save=False):
    split_vectors = {0: [], 1: []}
    for batch, targets, lengths, raw_data in create_dataset(
        val_dataset, id2tok, tok2id, target_label, batch_size=BATCH_SIZE
    ):
        tokenized = tokenized_sentence(raw_data[0])
        vector = torch.tensor(WE.get_sentence_vector(tokenized, vector_dict))
        # vector = add_features(vector, raw_data[0], FEAT_ADD_SOFTENER=0.3)
        split_vectors[targets.item()].append(vector.tolist())
    df = pd.DataFrame(columns=["label", "vector"])
    for label, vectors in split_vectors.items():
        for vector in vectors:
            df = df.append({"label": label, "vector": vector}, ignore_index=True)
    pca = PCA(n_components=2)
    pca_vals = pca.fit_transform(np.array(df["vector"].tolist()))
    df["pc1"] = [i[0] for i in pca_vals]
    df["pc2"] = [i[1] for i in pca_vals]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x="pc1", y="pc2", hue="label", data=df, s=100)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle(f'"{target_label.title()}" Glove PCA', fontsize=20)
    # plt.title("Number Count, Year Count")
    if save:
        plt.savefig(
            f"visualizations/{target_label}_glove_plot.png",
            dpi=400,
            bbox_inches="tight",
            pad_inches=0.2,
            facecolor="w",
        )

    split_vectors = {0: [], 1: []}
    for batch, targets, lengths, raw_data in create_dataset(
        val_dataset, id2tok, tok2id, target_label, batch_size=BATCH_SIZE
    ):
        tokenized = tokenized_sentence(raw_data[0])
        vector = torch.tensor(WE.get_sentence_vector(tokenized, vector_dict))
        vector = add_features(vector, raw_data[0], FEAT_ADD_SOFTENER=0.3)
        split_vectors[targets.item()].append(vector.tolist())
    df = pd.DataFrame(columns=["label", "vector"])
    for label, vectors in split_vectors.items():
        for vector in vectors:
            df = df.append({"label": label, "vector": vector}, ignore_index=True)
    pca = PCA(n_components=2)
    pca_vals = pca.fit_transform(np.array(df["vector"].tolist()))
    df["pc1"] = [i[0] for i in pca_vals]
    df["pc2"] = [i[1] for i in pca_vals]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x="pc1", y="pc2", hue="label", data=df, s=100)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle(f'"{target_label.title()}" Glove PCA with Added Features', fontsize=20)
    plt.title("Number Count, Year Count")
    if save:
        plt.savefig(
            f"visualizations/{target_label}_glove_plot_added_features.png",
            dpi=400,
            bbox_inches="tight",
            pad_inches=0.2,
            facecolor="w",
        )
    return pca_vals


target_label = "evidence lacking"
pca_vals = compare_pca_features(target_label, save=True)
checkpoint = torch.load("models/mlp/*14/3/3.pt")
model = MLPClassifier(302, 32, 0.6)
model.load_state_dict(checkpoint)
model.eval()
df = pd.DataFrame(columns=["actual", "predicted", "vector", "pc1", "pc2"])
preds = []
actual = []
vectors = []
for batch, targets, lengths, raw_data in create_dataset(
    val_dataset, id2tok, tok2id, target_label, batch_size=BATCH_SIZE
):
    tokenized = tokenized_sentence(raw_data[0])
    vector = torch.tensor(WE.get_sentence_vector(tokenized, vector_dict))
    vector = add_features(vector, raw_data[0], FEAT_ADD_SOFTENER=0.3)
    vectors.append(vector.tolist())
    preds.append(int(model(vector.type(torch.FloatTensor)) > 0.3))
    actual.append(targets.item())
print(classification_report(actual, preds))
pca_vectors = pca.fit_transform(np.array(vectors))
df["actual"] = actual
df["Model Prediction"] = preds
df["pc1"] = [i[0] for i in pca_vectors]
df["pc2"] = [i[1] for i in pca_vectors]
df["Correct?"] = df["predicted"] == df["actual"]
fig, ax = plt.subplots(figsize=(10, 7))
# sizes = [100 if i==True else 80 for i in df["Correct?"].tolist()]
sns.scatterplot(
    x="pc1",
    y="pc2",
    hue="Model Prediction",
    data=df,
    style="Correct?",
    markers={True: "o", False: "X"},
    s=100,
)
ax.set_xticks([])
ax.set_yticks([])
fig.suptitle(f'"{target_label.title()}" Predictions with Added Features', fontsize=20)
plt.title("0.80 Accuracy, 0.64 F1-Score")
plt.savefig(
    f"visualizations/{target_label}_predictions.png",
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.2,
    facecolor="w",
)


"""
+/- of added features
"""
BATCH_SIZE = 1
vocab, id2tok, tok2id = get_vocab(train_dataset)
feature_counts = {}
for label in LABELS:
    feature_counts[LABEL_TO_IX[label]] = {
        0: {"year_count": [], "num_count": []},
        1: {"year_count": [], "num_count": []},
    }
    for batch, targets, lengths, raw_data in create_dataset(
        val_dataset, id2tok, tok2id, label, batch_size=BATCH_SIZE
    ):
        num_count = len(
            re.findall(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?(?!\d)", raw_data[0])
        )
        year_count = len(re.findall(r"\d{4}", raw_data[0]))
        feature_counts[LABEL_TO_IX[label]][targets.item()]["year_count"].append(
            year_count
        )
        feature_counts[LABEL_TO_IX[label]][targets.item()]["num_count"].append(
            num_count
        )
        if num_count + year_count > 3:
            print(raw_data[0])
            print()


mean_plus_minus = {}
for label_ix, feat_data in feature_counts.items():
    print(label_ix)
    mean_pos_year = statistics.mean(feat_data[1]["year_count"])
    mean_neg_year = statistics.mean(feat_data[0]["year_count"])
    mean_pos_number = statistics.mean(feat_data[1]["num_count"])
    mean_neg_number = statistics.mean(feat_data[0]["num_count"])
    mean_plus_minus[label_ix] = {}
    mean_plus_minus[label_ix]["year"] = mean_pos_year - mean_neg_year
    mean_plus_minus[label_ix]["number"] = mean_pos_number - mean_neg_number
    mean_plus_minus[label_ix]["sum"] = (mean_pos_year - mean_neg_year) + (
        mean_pos_number - mean_neg_number
    )
X = [mean_plus_minus[ix]["sum"] for ix in range(len(mean_plus_minus))]
Y = [ix for ix in range(len(mean_plus_minus))]
fig = go.Figure()
for ix, (x, y) in enumerate(zip(X, Y)):
    fig.add_trace(
        go.Bar(
            x=[x],
            y=[IX_TO_LABEL[ix]],
            orientation="h",
            name=IX_TO_LABEL[ix],
            hovertemplate="%{y}: %{x}",
        )
    )
fig.update_layout(
    barmode="relative",
    height=400,
    width=700,
    yaxis_autorange="reversed",
    bargap=0.5,
    showlegend=False,
)
fig.update_layout(
    title={
        "text": "Class Correlations with Added Features",
        "y": 0.9,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    },
    xaxis_title="Correlation with class 1",
)
fig.write_image(f"visualizations/added_feature_correlations.png", scale=2)


"""
Label Distribution
"""
flattened_y = [item for sublist in raw_y for item in sublist]
to_plot = sorted(Counter(flattened_y).items(), key=lambda x: label_to_ix[x[0]])
count_df = pd.DataFrame(to_plot, columns=["name", "count"])
fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
fig.suptitle("Label Distributions", fontsize=20)
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="name", y="count", data=count_df)
ax.set_xlabel("Label", size=14)
ax.set_ylabel("Count", size=14)
ax.tick_params(axis="both", which="major", labelsize=14)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
plt.savefig(
    "visualizations/label_distribution", dpi=400, bbox_inches="tight", pad_inches=0.2
)

"""
Label Correlations
"""
mutliclass_instances = defaultdict(int)
total_instances = defaultdict(int)
class_corr_arr = np.zeros((len(LABELS), len(LABELS)))
for data in raw_y:
    for a in data:
        total_instances[label_to_ix[a]] += 1
        if len(data) > 1:
            mutliclass_instances[label_to_ix[a]] += 1
        for b in data:
            if a != b:
                class_corr_arr[label_to_ix[a]][label_to_ix[b]] += 1
# cm = confusion_matrix(expected, actual)
sums = np.sum(class_corr_arr, axis=0).reshape(-1, 1)
mutliclass_instances_arr = np.array(
    [mutliclass_instances[ix] for ix in range(len(mutliclass_instances))]
)
df_cm = pd.DataFrame(
    class_corr_arr,
    index=[ix_to_label[ix] for ix in range(len(ix_to_label))],
    columns=[ix_to_label[ix] for ix in range(len(ix_to_label))],
)
fig, ax = plt.subplots(figsize=(10, 7))
cmap = sns.light_palette((260, 75, 60), input="husl", n_colors=20)
sns.heatmap(df_cm, annot=True, cmap=cmap, linecolor="black", linewidths=0.1)
ax.tick_params(right=False, top=True, labelright=False, labeltop=False, labelsize=12)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
plt.suptitle("Label Correlations", fontsize=20)
# plt.savefig("visualizations/label_correlation", dpi=400, bbox_inches = "tight", pad_inches = 0.2)

"""
% of Occurences in Multilabel Setting
"""
ix_to_counts = {
    label_to_ix[label]: v for label, v in dict(Counter(flattened_y)).items()
}
total_counts = np.array([ix_to_counts[ix] for ix in range(len(ix_to_counts))])
ordered_multilabel = dict(
    zip(LABELS, (mutliclass_instances_arr / total_counts).tolist())
).items()
multilabel_df = pd.DataFrame(ordered_multilabel, columns=["Label", "% Multilabel"])
fig, ax = plt.subplots()
fig.set_size_inches(15, 10)
fig.suptitle("% of Occurences in Multilabel Setting", fontsize=20)
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="Label", y="% Multilabel", data=multilabel_df)
ax.set_xlabel("Label", size=14)
ax.set_ylabel("% Multilabel", size=14)
ax.tick_params(axis="both", which="major", labelsize=12)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
# plt.savefig("visualizations/mutlilabel_instances", dpi=400, bbox_inches = "tight", pad_inches = 0.2)
