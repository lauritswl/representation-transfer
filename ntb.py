#%% 
from src.embedder import Embedder
from conceptdefiner import ConceptVector
print("Modules imported successfully.")

# %%
import pandas as pd
# Load the Fiction4 Corpus
Fiction4 = pd.read_csv("data/clean_text/fiction4.csv")

# Load the EmoBank Corpus
Emobank = pd.read_csv("data/clean_text/emobank.csv")

# Load the Facebook Corpus
Facebook = pd.read_csv("data/clean_text/facebook.csv")


# Create a dictionary of corpora dictionaries; fill 'valence_label' and 'embedding' later
corpora = {
    "Fiction4": {"valence_label": None, "embedding": None, "dataframe": Fiction4},
    "Emobank": {"valence_label": None, "embedding": None, "dataframe": Emobank},
    "Facebook": {"valence_label": None, "embedding": None, "dataframe": Facebook},
}

# %%
# --- Embedding Text Data with Caching ---
from src.embedder import Embedder
MultiLingMPNET = Embedder(model_name="paraphrase-multilingual-mpnet-base-v2")

# Embed fiction4 corpus with caching
corpora["Fiction4"]["embedding"] = MultiLingMPNET.embed(corpora["Fiction4"]["dataframe"]["text"], cache_path="data/embeddings/fiction4.csv")

# Embed Emobank corpus with caching
corpora["Emobank"]["embedding"] = MultiLingMPNET.embed(corpora["Emobank"]["dataframe"]["text"], cache_path="data/embeddings/emobank.csv")

# Embed Facebook corpus with caching
corpora["Facebook"]["embedding"] = MultiLingMPNET.embed(corpora["Facebook"]["dataframe"]["text"], cache_path="data/embeddings/facebook.csv")

# %%
# --- Setting Up Valence Labels ---
# Follow a scheme like this
# df['binary_label'] = df['continuous_label'].apply(
#     lambda x: "positive" if x >= positive_threshold else ("negative" if x <= negative_threshold else "neutral")
# )
# But where the cutoff points are one standard deviation from the mean.
import numpy as np
for corpus_name, corpus_info in corpora.items():
    df = corpus_info["dataframe"]
    if "valence" in df.columns:
        mean_valence = df["valence"].mean()
        std_valence = df["valence"].std()
        positive_threshold = mean_valence + std_valence
        negative_threshold = mean_valence - std_valence

        def label_valence(x):
            if x >= positive_threshold:
                return "positive"
            elif x <= negative_threshold:
                return "negative"
            else:
                return "neutral"

        corpus_info["valence_label"] = df["valence"].apply(label_valence)
    else:
        print(f"Valence column not found in {corpus_name} dataframe.")
    # For debugging, print the distribution of labels
    print(f"{corpus_name} valence label distribution:\n{corpus_info['valence_label'].value_counts()}\n")
# %%

# Plot distributions of valence scores for valence labels in emobank:
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# define consistent label order and colors for all plots
label_order = ["positive", "neutral", "negative"]
palette = {"positive": "tab:green", "neutral": "tab:blue", "negative": "tab:red"}

# Plot all three
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # a bit taller to allow legend space
for ax, (corpus_name, corpus_info) in zip(axes, corpora.items()):
    df = corpus_info["dataframe"].copy()
    # ensure the dataframe has a valence_label column for seaborn hue
    df["valence_label"] = corpus_info["valence_label"]
    sns.histplot(
        data=df,
        x="valence",
        hue="valence_label",
        hue_order=label_order,
        palette=palette,
        bins=30,
        kde=True,
        ax=ax,
        legend=False,  # we'll add a single consistent legend below
    )
    ax.set_title(f"Distribution of Valence Scores in {corpus_name} Corpus")
    ax.set_xlabel("Valence Score")
    ax.set_ylabel("Frequency")

# add a single legend for the figure with consistent colors, positioned a bit higher
legend_handles = [Patch(color=palette[l], label=l) for l in label_order]
fig.legend(handles=legend_handles, title="Valence Label", loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12))
plt.tight_layout(rect=[0, 0, 1, 0.92])  # leave extra space at top for the legend
plt.show()

# %%

# --- Concept Vector Creation (three directions) ---
from conceptdefiner import ConceptVector
import numpy as np

concept_vectors = {}
for corpus_name, corpus_info in corpora.items():
    embeddings = np.asarray(corpus_info["embedding"])
    labels = np.asarray(corpus_info["valence_label"])

    # helper to get embeddings for a given label
    def get_emb(lbl):
        mask = labels == lbl
        return embeddings[mask]

    pos_embeddings = get_emb("positive")
    neg_embeddings = get_emb("negative")
    neu_embeddings = get_emb("neutral")

    cv_dict = {}
    pairs = {
        "negative_to_positive": (neg_embeddings, pos_embeddings),
        "neutral_to_positive": (neu_embeddings, pos_embeddings),
        "negative_to_neutral": (neg_embeddings, neu_embeddings),
    }

    for pair_name, (src_emb, tgt_emb) in pairs.items():
        src_count = 0 if src_emb.size == 0 else src_emb.shape[0]
        tgt_count = 0 if tgt_emb.size == 0 else tgt_emb.shape[0]
        if src_count == 0 or tgt_count == 0:
            print(f"Skipping {corpus_name} {pair_name}: insufficient examples (src={src_count}, tgt={tgt_count})")
            cv_dict[pair_name] = None
            continue

        cv = ConceptVector(normalize=True)
        cv.fit(src_emb, tgt_emb)
        cv_dict[pair_name] = cv
        print(f"Concept vector for {corpus_name} ({pair_name}) created.")

    concept_vectors[corpus_name] = cv_dict



# %%
# Show structure of cv_dict
print("Concept Vectors Dictionary Structure: \n")

import pprint
pprint.pprint({k: list(v.keys()) if v is not None else None for k, v in concept_vectors.items()})

# %%
# Cosine similarity matrix between the concept vectors
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
# Prepare data for cosine similarity
cv_names = []
cv_vectors = []
for corpus_name, cv_dict in concept_vectors.items():
    for pair_name, cv in cv_dict.items():
        if cv is not None:
            cv_names.append(f"{corpus_name}_{pair_name}")
            cv_vectors.append(cv.vector)
# Compute cosine similarity matrix
cv_matrix = cosine_similarity(cv_vectors)
# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cv_matrix, xticklabels=cv_names, yticklabels=cv_names, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Cosine Similarity Between Concept Vectors")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# %%
# Create three cosine similarity plots, grouped by concept direction
for pair_name in ["negative_to_positive", "neutral_to_positive", "negative_to_neutral"]:
    cv_names = []
    cv_vectors = []
    for corpus_name, cv_dict in concept_vectors.items():
        cv = cv_dict.get(pair_name)
        if cv is not None:
            cv_names.append(f"{corpus_name}_{pair_name}")
            cv_vectors.append(cv.vector)
    if len(cv_vectors) < 2:
        print(f"Not enough concept vectors for {pair_name} to compute similarity matrix.")
        print("cv_vectors:", cv_vectors)
        continue
    cv_matrix = cosine_similarity(cv_vectors)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cv_matrix, xticklabels=cv_names, yticklabels=cv_names, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Cosine Similarity Between Concept Vectors ({pair_name})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
# %%
# Check cosine similarity between Negative to Positive and the sum of negative to neutral and neutral to positive
for corpus_name, cv_dict in concept_vectors.items():
    cv_neg_pos = cv_dict.get("negative_to_positive")
    cv_neg_neu = cv_dict.get("negative_to_neutral")
    cv_neu_pos = cv_dict.get("neutral_to_positive")
    if cv_neg_pos is None or cv_neg_neu is None or cv_neu_pos is None:
        print(f"Skipping {corpus_name}: missing concept vectors for comparison.")
        continue
    combined_vector = cv_neg_neu.vector + cv_neu_pos.vector
    combined_vector_norm = combined_vector / np.linalg.norm(combined_vector)
    cos_sim = np.dot(cv_neg_pos.vector, combined_vector_norm)
    print(f"Cosine similarity between Negative to Positive and sum of Negative to Neutral + Neutral to Positive for {corpus_name}: {cos_sim:.4f}")


# %%
# Find the best vector that projects from the mean fiction4 neutral text to the neg_to_pos vector.
import numpy as np
neutral_mask = corpora["Fiction4"]["valence_label"] == "neutral"
neutral_embeddings = np.asarray(corpora["Fiction4"]["embedding"])[neutral_mask]
mean_neutral = neutral_embeddings.mean(axis=0)
negative_mask = corpora["Fiction4"]["valence_label"] == "negative"
negative_embeddings = np.asarray(corpora["Fiction4"]["embedding"])[negative_mask]
mean_negative = negative_embeddings.mean(axis=0)
neg_to_pos_vector = concept_vectors["Fiction4"]["negative_to_positive"].vector

# Find best_vector projecting from mean_neutral onto the vector neg_to_pos and taking the difference between the projected point and mean_neutral
proj_distance_from_negative = np.dot(mean_neutral, neg_to_pos_vector) * neg_to_pos_vector
proj_point = mean_negative + proj_distance_from_negative
best_vector = proj_point - mean_neutral


# Plot all datapoints with axis 1 being neg_to_pos projection and axis 2 being best_vector projection
import matplotlib.pyplot as plt
projections_neg_to_pos = concept_vectors["Fiction4"]["negative_to_positive"].project(corpora["Fiction4"]["embedding"])
# Project corpora["Fiction4"]["embedding"] onto best_vector
# First, normalize best_vector
best_vector_norm = best_vector / np.linalg.norm(best_vector)
projections_best_vector = np.dot(corpora["Fiction4"]["embedding"], best_vector_norm)

df_plot = pd.DataFrame({
    "neg_to_pos_projection": projections_neg_to_pos,
    "best_vector_projection": projections_best_vector,
    "valence_label": corpora["Fiction4"]["valence_label"]
})

# Calculate mean projections for each label
positive_mask = corpora["Fiction4"]["valence_label"] == "positive"
positive_embeddings = np.asarray(corpora["Fiction4"]["embedding"])[positive_mask]
mean_positive = positive_embeddings.mean(axis=0)

mean_neutral_proj_neg_to_pos = np.dot(mean_neutral, neg_to_pos_vector)
mean_neutral_proj_best = np.dot(mean_neutral, best_vector_norm)
mean_negative_proj_neg_to_pos = np.dot(mean_negative, neg_to_pos_vector)
mean_negative_proj_best = np.dot(mean_negative, best_vector_norm)
mean_positive_proj_neg_to_pos = np.dot(mean_positive, neg_to_pos_vector)
mean_positive_proj_best = np.dot(mean_positive, best_vector_norm)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_plot,
    x="neg_to_pos_projection",
    y="best_vector_projection",
    hue="valence_label",
    palette={"positive": "tab:green", "neutral": "tab:blue", "negative": "tab:red"},
    alpha=0.6,
    size=0.001
)
# Add mean points for each label
plt.scatter([mean_negative_proj_neg_to_pos], [mean_negative_proj_best], color="darkred", s=200, marker="X", edgecolors="black", linewidth=2, label="Mean Negative", zorder=5)
plt.scatter([mean_neutral_proj_neg_to_pos], [mean_neutral_proj_best], color="darkblue", s=200, marker="X", edgecolors="black", linewidth=2, label="Mean Neutral", zorder=5)
plt.scatter([mean_positive_proj_neg_to_pos], [mean_positive_proj_best], color="darkgreen", s=200, marker="X", edgecolors="black", linewidth=2, label="Mean Positive", zorder=5)

plt.xlabel("Projection onto Negative to Positive Concept Vector")
plt.ylabel("Projection onto Best Vector from Mean Neutral to Neg→Pos")
plt.title("Scatterplot of Projections onto Neg→Pos vs. Best Vector (Fiction4 Corpus)")
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.axvline(0, color="gray", linestyle="--", linewidth=1)
plt.legend(title="Valence Label")
plt.tight_layout()
plt.show()
# %%
# Distributions of both projections
import matplotlib.pyplot as plt
import seaborn as sns
projections_neg_to_pos = concept_vectors["Fiction4"]["negative_to_positive"].project(corpora["Fiction4"]["embedding"])
projections_best_vector = np.dot(corpora["Fiction4"]["embedding"], best_vector_norm)
df_plot = pd.DataFrame({
    "neg_to_pos_projection": projections_neg_to_pos,
    "best_vector_projection": projections_best_vector,
    "valence_label": corpora["Fiction4"]["valence_label"]
})
plt.figure(figsize=(12, 5))
# Neg to Pos Projection
plt.subplot(1, 2, 1)
sns.histplot(
    data=df_plot,
    x="neg_to_pos_projection",
    hue="valence_label",
    palette={"positive": "tab:green", "neutral": "tab:blue", "negative": "tab:red"},
    bins=30,
    kde=True,
    alpha=0.5
)
plt.title("Distribution of Projections onto Negative to Positive Concept Vector")
plt.xlabel("Projection onto Negative to Positive Concept Vector")
plt.ylabel("Frequency")
# Best Vector Projection
plt.subplot(1, 2, 2)
sns.histplot(
    data=df_plot,
    x="best_vector_projection",
    hue="valence_label",
    palette={"positive": "tab:green", "neutral": "tab:blue", "negative": "tab:red"},
    bins=30,
    kde=True,
    alpha=0.5
)
plt.title("Distribution of Projections onto Best Vector from Mean Neutral to Neg→Pos")
plt.xlabel("Projection onto Best Vector from Mean Neutral to Neg→Pos")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Cosine similarity between best_vector and neg_to_pos_vector
cos_sim_best_negpos = np.dot(best_vector / np.linalg.norm(best_vector), neg_to_pos_vector / np.linalg.norm(neg_to_pos_vector))
print(f"Cosine similarity between Best Vector and Negative to Positive Concept Vector: {cos_sim_best_negpos:.4f}")

# %%
# Count-normalized KDE plot (KDE area equals count of points per label)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

projections = concept_vectors["Fiction4"]["negative_to_positive"].project(corpora["Emobank"]["embedding"])
labels = corpora["Emobank"]["valence_label"]
df_plot = pd.DataFrame({"projection": projections, "valence_label": labels})

plt.figure(figsize=(10, 6))
palette = {"negative": "tab:red", "neutral": "tab:blue", "positive": "tab:green"}
bins = np.linspace(-2, 2, 41)

# Plot histogram for each label with alpha=0.5 (counts)
for label in ["negative", "neutral", "positive"]:
    subset = df_plot[df_plot["valence_label"] == label]["projection"]
    if subset.shape[0] > 1:
        sns.histplot(
            subset,
            bins=bins,
            color=palette[label],
            label=label,
            alpha=0.5,
            stat="count",
            kde=False,
        )
    else:
        sns.rugplot(subset, color=palette[label], height=0.05)

plt.xlabel("Projection onto Negative to Positive Concept Vector")
plt.ylabel("Count")
plt.title("Distribution of Projections by Valence Label (Histogram, alpha=0.5) — Fiction4 Vector on Emobank")
plt.legend(title="Valence Label")
plt.xlim(-2, 2)
plt.tight_layout()
plt.show()

# %%
# Count-normalized KDE plot (KDE area equals count of points per label)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

projections = concept_vectors["Fiction4"]["negative_to_positive"].project(corpora["Emobank"]["embedding"])
labels = corpora["Emobank"]["valence_label"]
df_plot = pd.DataFrame({"projection": projections, "valence_label": labels})

plt.figure(figsize=(10, 6))
palette = {"negative": "tab:red", "neutral": "tab:blue", "positive": "tab:green"}
clip_range = (-2, 2)

# Plot a KDE for each label where the KDE area equals the count of points (weights=sum -> area = count)
for label in ["negative", "neutral", "positive"]:
    subset = df_plot[df_plot["valence_label"] == label]["projection"].dropna()
    n = subset.shape[0]
    if n > 1:
        sns.kdeplot(
            x=subset,
            fill=True,
            label=f"{label} (n={n})",
            bw_adjust=0.5,
            clip=clip_range,
            color=palette[label],
            alpha=0.5,
            weights=np.ones_like(subset),  # area of KDE == sum(weights) == n (count-normalized)
            common_norm=False
        )
    elif n == 1:
        # Single point: show a rug and a vertical line scaled to 1
        x0 = subset.values[0]
        sns.rugplot([x0], color=palette[label], height=0.05)
        plt.vlines(x0, 0, 1, color=palette[label], alpha=0.6, label=f"{label} (n=1)")

plt.xlabel("Projection onto Negative to Positive Concept Vector")
plt.ylabel("Count-normalized KDE (area = count)")
plt.title("Count-normalized KDE of Projections by Valence Label — Fiction4 Vector on Emobank")
plt.legend(title="Valence Label (count)")
plt.xlim(*clip_range)
plt.tight_layout()
plt.show()

# %%
# Scatterplot between dataframe["valence"] and projections onto the negative_to_positive concept vector
import matplotlib.pyplot as plt
projections = concept_vectors["Fiction4"]["negative_to_positive"].project(corpora["Emobank"]["embedding"])
valence_scores = corpora["Emobank"]["dataframe"]["valence"].reset_index(drop=True)

# Ensure projections is a 1D numpy array aligned with valence_scores
projections = np.asarray(projections).ravel()

df_scatter = pd.DataFrame({"valence": valence_scores, "projection": projections})

# Compute mean and std and create a hue/category column based on std thresholds
mean_v = df_scatter["valence"].mean()
std_v = df_scatter["valence"].std()

def valence_std_category(x):
    if x >= mean_v + std_v:
        return "Positive (>= mean + 1σ)"
    elif x <= mean_v - std_v:
        return "Negative (<= mean - 1σ)"
    else:
        return "Neutral (Within 1σ)"

df_scatter["valence_hue"] = df_scatter["valence"].apply(valence_std_category)

# Plot with hue and show mean / mean±std lines
plt.figure(figsize=(10, 6))
palette = {"Positive (>= mean + 1σ)": "tab:green", "Neutral (Within 1σ)": "tab:blue", "Negative (<= mean - 1σ)": "tab:red"}
sns.scatterplot(data=df_scatter, x="valence", y="projection", hue="valence_hue", palette=palette, alpha=0.7)

plt.xlabel("Valence Score")
plt.ylabel("Projection onto Negative to Positive Concept Vector")
plt.title("Scatterplot of Valence Scores vs. Projections (Fiction4 Vector - Emobank Corpus) — hue by ±1σ from mean")
plt.legend(title="Valence (std category)", loc="best")
plt.tight_layout()
plt.show()

# %%
# Creating a 9x9 confusion matrix, testing training on a concept vector from one corpus and projecting another corpus.
# The values in the matrix will be the correlation coefficient between the projections and valence scores.
import numpy as np
import pandas as pd
corpus_names = list(corpora.keys())
n = len(corpus_names)
confusion_matrix = pd.DataFrame(index=corpus_names, columns=corpus_names, dtype=float)
for train_name in corpus_names:
    for test_name in corpus_names:
        cv = concept_vectors[train_name]["negative_to_positive"]
        if cv is None:
            confusion_matrix.loc[train_name, test_name] = np.nan
            continue
        test_embeddings = corpora[test_name]["embedding"]
        test_valence = corpora[test_name]["dataframe"]["valence"]
        projections = cv.project(test_embeddings)
        if len(projections) < 2:
            confusion_matrix.loc[train_name, test_name] = np.nan
            continue
        corr = np.corrcoef(projections, test_valence)[0, 1]
        confusion_matrix.loc[train_name, test_name] = corr
# Plot the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Confusion Matrix of Correlation Coefficients (Train on Row, Test on Column)")
plt.xlabel("Test Corpus")
plt.ylabel("Train Corpus")
plt.tight_layout()
plt.show()




# %%
# Creating a n x n scatter matrix, testing training on a concept vector from one corpus and projecting another corpus.
# Emobank valence (1-5) is standardized per test corpus so correlations and plots are comparable across corpora.
import numpy as np
import pandas as pd

corpus_names = list(corpora.keys())
n = len(corpus_names)
fig, axes = plt.subplots(n, n, figsize=(15, 15), sharex=True, sharey=True)

# track global y-limits for projections so we can set consistent y-axis after plotting
ymin, ymax = np.inf, -np.inf
plotted_axes = []

for i, train_name in enumerate(corpus_names):
    for j, test_name in enumerate(corpus_names):
        ax = axes[i, j]
        cv = concept_vectors[train_name]["negative_to_positive"]
        if cv is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        test_embeddings = corpora[test_name]["embedding"]
        test_valence = corpora[test_name]["dataframe"]["valence"]

        projections = cv.project(test_embeddings)
        projections = np.asarray(projections).ravel()

        # align lengths and coerce valence to numeric (handles Emobank's 1-5 and any other scales)
        valence = pd.to_numeric(test_valence, errors="coerce").reset_index(drop=True).values
        if len(projections) != len(valence):
            m = min(len(projections), len(valence))
            projections = projections[:m]
            valence = valence[:m]

        # mask out NaNs
        mask = (~np.isnan(projections)) & (~np.isnan(valence))
        if mask.sum() < 2:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        proj_masked = projections[mask]
        val_masked = valence[mask]

        # standardize valence per test corpus (z-score). This handles Emobank's 1-5 scale.
        if np.nanstd(val_masked) > 0:
            val_std = (val_masked - np.nanmean(val_masked)) / np.nanstd(val_masked)
        else:
            val_std = np.zeros_like(val_masked)

        # compute correlation on the masked standardized valence
        corr = np.corrcoef(proj_masked, val_std)[0, 1]

        ax.scatter(val_std, proj_masked, alpha=0.5)
        ax.set_title(f"{train_name} → {test_name}\nCorr: {np.nan_to_num(corr):.2f}")

        if i == n - 1:
            ax.set_xlabel("Standardized Valence (z-score)")
        if j == 0:
            ax.set_ylabel("Projection")

        ymin = min(ymin, np.min(proj_masked))
        ymax = max(ymax, np.max(proj_masked))
        plotted_axes.append(ax)

# set consistent x and y limits
xlim = (-5, 5)  # standardized valence will generally lie in this range
if ymin < np.inf:
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    ylim = (ymin - pad, ymax + pad)
    for ax in axes.flat:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

plt.suptitle("Scatter Matrix of Projections vs. Standardized Valence (Train on Row, Test on Column)", y=1.02)
plt.tight_layout()
plt.show()


# %%
# Create a dictionary of corpora dictionaries using arousal labels
arousal_corpora = {
    "Emobank": {"arousal_label": None, "embedding": corpora["Emobank"]["embedding"], "dataframe": Emobank},
    "Facebook": {"arousal_label": None, "embedding": corpora["Facebook"]["embedding"], "dataframe": Facebook},
}

import numpy as np
for corpus_name, corpus_info in arousal_corpora.items():
    df = corpus_info["dataframe"]
    if "arousal" in df.columns:
        mean_arousal = df["arousal"].mean()
        std_arousal = df["arousal"].std()
        positive_threshold = mean_arousal + std_arousal
        negative_threshold = mean_arousal - std_arousal
        def label_arousal(x):
            if x >= positive_threshold:
                return "positive"
            elif x <= negative_threshold:
                return "negative"
            else:
                return "neutral"

        corpus_info["arousal_label"] = df["arousal"].apply(label_arousal)
    else:
        print(f"Arousal column not found in {corpus_name} dataframe.")
    # For debugging, print the distribution of labels
    print(f"{corpus_name} arousal label distribution:\n{corpus_info['arousal_label'].value_counts()}\n")


arousal_concept_vectors = {}
for corpus_name, corpus_info in arousal_corpora.items():
    embeddings = np.asarray(corpus_info["embedding"])
    labels = np.asarray(corpus_info["arousal_label"])

    # helper to get embeddings for a given label
    def get_emb(lbl):
        mask = labels == lbl
        return embeddings[mask]

    pos_embeddings = get_emb("positive")
    neg_embeddings = get_emb("negative")
    neu_embeddings = get_emb("neutral")

    cv_dict = {}
    pairs = {
        "negative_to_positive": (neg_embeddings, pos_embeddings),
        "neutral_to_positive": (neu_embeddings, pos_embeddings),
        "negative_to_neutral": (neg_embeddings, neu_embeddings),
    }

    for pair_name, (src_emb, tgt_emb) in pairs.items():
        src_count = 0 if src_emb.size == 0 else src_emb.shape[0]
        tgt_count = 0 if tgt_emb.size == 0 else tgt_emb.shape[0]
        if src_count == 0 or tgt_count == 0:
            print(f"Skipping {corpus_name} {pair_name}: insufficient examples (src={src_count}, tgt={tgt_count})")
            cv_dict[pair_name] = None
            continue

        cv = ConceptVector(normalize=True)
        cv.fit(src_emb, tgt_emb)
        cv_dict[pair_name] = cv
        print(f"Concept vector for {corpus_name} ({pair_name}) created.")

    arousal_concept_vectors[corpus_name] = cv_dict



# %%
# Count-normalized histogram for arousal corpora using arousal concept vectors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# choose corpus and direction from arousal_concept_vectors / arousal_corpora
train_corpus = "Emobank"
direction = "negative_to_positive"

cv = arousal_concept_vectors.get(train_corpus, {}).get(direction)
if cv is None:
    print(f"No concept vector found for {train_corpus} ({direction}). Skipping plot.")
else:
    projections = cv.project(arousal_corpora[train_corpus]["embedding"])
    labels = arousal_corpora[train_corpus]["arousal_label"]

    df_plot = pd.DataFrame({"projection": np.asarray(projections).ravel(), "arousal_label": labels}).dropna()

    plt.figure(figsize=(10, 6))
    palette = {"negative": "tab:red", "neutral": "tab:blue", "positive": "tab:green"}
    bins = np.linspace(-2, 2, 41)

    # Plot histogram for each arousal label with alpha=0.5 (counts)
    for label in ["negative", "neutral", "positive"]:
        subset = df_plot[df_plot["arousal_label"] == label]["projection"]
        if subset.shape[0] > 1:
            sns.histplot(
                subset,
                bins=bins,
                color=palette[label],
                label=label,
                alpha=0.5,
                stat="count",
                kde=False,
            )
        else:
            sns.rugplot(subset, color=palette[label], height=0.05)

    plt.xlabel("Projection onto Negative → Positive Arousal Concept Vector")
    plt.ylabel("Count")
    plt.title(f"Distribution of Projections by Arousal Label — {train_corpus} ({direction})")
    plt.legend(title="Arousal Label")
    plt.xlim(-2, 2)
    plt.tight_layout()
    plt.show()

# %%
# Creating a n x n scatter matrix, testing training on a concept vector from one corpus and projecting another corpus.
# Emobank arousal (1-5) is standardized per test corpus so correlations and plots are comparable across corpora.
import numpy as np
import pandas as pd

corpus_names = list(arousal_corpora.keys())
n = len(corpus_names)
fig, axes = plt.subplots(n, n, figsize=(15, 15), sharex=True, sharey=True)

# track global y-limits for projections so we can set consistent y-axis after plotting
ymin, ymax = np.inf, -np.inf
plotted_axes = []

for i, train_name in enumerate(corpus_names):
    for j, test_name in enumerate(corpus_names):
        ax = axes[i, j]
        cv = arousal_concept_vectors[train_name]["negative_to_positive"]
        if cv is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        test_embeddings = arousal_corpora[test_name]["embedding"]
        test_arousal = arousal_corpora[test_name]["dataframe"]["arousal"]

        projections = cv.project(test_embeddings)
        projections = np.asarray(projections).ravel()

        # align lengths and coerce arousal to numeric (handles Emobank's 1-5 and any other scales)
        arousal = pd.to_numeric(test_arousal, errors="coerce").reset_index(drop=True).values
        if len(projections) != len(arousal):
            m = min(len(projections), len(arousal))
            projections = projections[:m]
            arousal = arousal[:m]

        # mask out NaNs
        mask = (~np.isnan(projections)) & (~np.isnan(arousal))
        if mask.sum() < 2:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        proj_masked = projections[mask]
        arousal_masked = arousal[mask]

        # standardize arousal per test corpus (z-score). This handles Emobank's 1-5 scale.
        if np.nanstd(arousal_masked) > 0:
            arousal_std = (arousal_masked - np.nanmean(arousal_masked)) / np.nanstd(arousal_masked)
        else:
            arousal_std = np.zeros_like(arousal_masked)

        # compute correlation on the masked standardized arousal
        corr = np.corrcoef(proj_masked, arousal_std)[0, 1]

        ax.scatter(arousal_std, proj_masked, alpha=0.5)
        ax.set_title(f"{train_name} → {test_name}\nCorr: {np.nan_to_num(corr):.2f}")

        if i == n - 1:
            ax.set_xlabel("Standardized Arousal (z-score)")
        if j == 0:
            ax.set_ylabel("Projection")

        ymin = min(ymin, np.min(proj_masked))
        ymax = max(ymax, np.max(proj_masked))
        plotted_axes.append(ax)

# set consistent x and y limits
xlim = (-5, 5)  # standardized arousal will generally lie in this range
if ymin < np.inf:
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    ylim = (ymin - pad, ymax + pad)
    for ax in axes.flat:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

plt.suptitle("Scatter Matrix of Projections vs. Standardized Valence (Train on Row, Test on Column)", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Create a dataframe of the 5 text with the highest and lowest projections from the neutral arousal label in Emobank using the Emobank arousal negative_to_positive concept vector (include original arousal)
import pandas as pd
cv = arousal_concept_vectors["Emobank"]["negative_to_positive"]
embeddings = arousal_corpora["Emobank"]["embedding"]
df_emobank = arousal_corpora["Emobank"]["dataframe"]
labels = arousal_corpora["Emobank"]["arousal_label"]

neutral_mask = labels == "neutral"
neutral_embeddings = np.asarray(embeddings)[neutral_mask]
neutral_texts = df_emobank.loc[neutral_mask, "text"].reset_index(drop=True)
neutral_arousal = df_emobank.loc[neutral_mask, "arousal"].reset_index(drop=True)

projections = cv.project(neutral_embeddings)
df_projections = pd.DataFrame({
    "text": neutral_texts,
    "arousal": neutral_arousal,
    "projection": np.asarray(projections).ravel()
})

# Get top 5 and bottom 5
top_5 = df_projections.nlargest(5, "projection")
bottom_5 = df_projections.nsmallest(5, "projection")
result_df = pd.concat([top_5, bottom_5]).reset_index(drop=True)
print("Top and Bottom 5 Texts by Projection onto Emobank Arousal Negative to Positive Concept Vector (including original arousal):\n")
result_df



# %%
# For the emobank dataset, project onto arousal and valence and display results in a scatterplot. Compare to a scatterplot with original scores.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

emobank_embeddings = corpora["Emobank"]["embedding"]
emobank_valence = corpora["Emobank"]["dataframe"]["valence"]
emobank_arousal = corpora["Emobank"]["dataframe"]["arousal"]
valence_cv = concept_vectors["Emobank"]["negative_to_positive"]
arousal_cv = arousal_concept_vectors["Emobank"]["negative_to_positive"]

valence_projections = np.asarray(valence_cv.project(emobank_embeddings)).ravel()
arousal_projections = np.asarray(arousal_cv.project(emobank_embeddings)).ravel()

# Align lengths and mask NaNs for projected arrays
m_proj = min(len(valence_projections), len(arousal_projections))
vp = valence_projections[:m_proj]
ap = arousal_projections[:m_proj]
mask_proj = (~np.isnan(vp)) & (~np.isnan(ap))
r_proj = np.nan
if mask_proj.sum() > 1:
    r_proj = np.corrcoef(vp[mask_proj], ap[mask_proj])[0, 1]
print(f"Pearson correlation (projected valence vs projected arousal): {r_proj:.4f}")

# Scatterplot of projected valence vs. projected arousal
plt.figure(figsize=(10, 6))
plt.scatter(vp, ap, alpha=0.5)
plt.xlabel("Projected Valence (Negative to Positive Concept Vector)")
plt.ylabel("Projected Arousal (Negative to Positive Concept Vector)")
plt.title("Scatterplot of Projected Valence vs. Projected Arousal (Emobank Corpus)")
plt.text(0.05, 0.95, f"Pearson r = {np.nan_to_num(r_proj):.3f}", transform=plt.gca().transAxes,
         ha="left", va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.show()

# Prepare original scores (coerce to numeric), align lengths and mask NaNs
orig_v = pd.to_numeric(emobank_valence, errors="coerce").reset_index(drop=True).values
orig_a = pd.to_numeric(emobank_arousal, errors="coerce").reset_index(drop=True).values
m_orig = min(len(orig_v), len(orig_a))
ov = orig_v[:m_orig]
oa = orig_a[:m_orig]
mask_orig = (~np.isnan(ov)) & (~np.isnan(oa))
r_orig = np.nan
if mask_orig.sum() > 1:
    r_orig = np.corrcoef(ov[mask_orig], oa[mask_orig])[0, 1]
print(f"Pearson correlation (original valence vs original arousal): {r_orig:.4f}")

# Scatterplot of original valence vs. original arousal
plt.figure(figsize=(10, 6))
plt.scatter(ov, oa, alpha=0.5)
plt.xlabel("Original Valence Score")
plt.ylabel("Original Arousal Score")
plt.title("Scatterplot of Original Valence vs. Original Arousal (Emobank Corpus)")
plt.text(0.05, 0.95, f"Pearson r = {np.nan_to_num(r_orig):.3f}", transform=plt.gca().transAxes,
         ha="left", va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.show()

# %%
# Replace scatterplots with 2D density heatmaps (projected vs projected, original vs original)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Ensure concept vectors exist
valence_cv = concept_vectors.get("Emobank", {}).get("negative_to_positive")
arousal_cv = arousal_concept_vectors.get("Emobank", {}).get("negative_to_positive")
if valence_cv is None or arousal_cv is None:
    print("Missing Emobank valence/arousal concept vectors. Cannot plot heatmaps.")
else:
    # get projections and original scores, align lengths and drop NaNs
    proj_v = np.asarray(valence_cv.project(emobank_embeddings)).ravel()
    proj_a = np.asarray(arousal_cv.project(emobank_embeddings)).ravel()
    orig_v = pd.to_numeric(emobank_valence, errors="coerce").reset_index(drop=True).values
    orig_a = pd.to_numeric(emobank_arousal, errors="coerce").reset_index(drop=True).values

    m = min(len(proj_v), len(proj_a), len(orig_v), len(orig_a))
    proj_v, proj_a = proj_v[:m], proj_a[:m]
    orig_v, orig_a = orig_v[:m], orig_a[:m]

    # helper to compute 2D histogram and plot as heatmap
    def plot_heatmap(x, y, xlabel, ylabel, title, bins=60, cmap="viridis"):
        mask = (~np.isnan(x)) & (~np.isnan(y))
        x, y = x[mask], y[mask]
        if x.size == 0:
            print(f"No valid points to plot for {title}")
            return
        H, xedges, yedges = np.histogram2d(x, y, bins=bins)
        # H shape = (len(xedges)-1, len(yedges)-1). transpose for pcolormesh so x->horizontal, y->vertical
        fig, ax = plt.subplots(figsize=(8, 6))
        pcm = ax.pcolormesh(xedges, yedges, H.T, cmap=cmap, shading="auto")
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("Count")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    # Heatmap: projected valence vs projected arousal
    plot_heatmap(
        proj_v,
        proj_a,
        xlabel="Projected Valence (Negative → Positive)",
        ylabel="Projected Arousal (Negative → Positive)",
        title="Heatmap: Projected Valence vs Projected Arousal (Emobank)"
    )

    # Heatmap: original valence vs original arousal
    plot_heatmap(
        orig_v,
        orig_a,
        xlabel="Original Valence Score",
        ylabel="Original Arousal Score",
        title="Heatmap: Original Valence vs Original Arousal (Emobank)",
        bins=50,
        cmap="magma"
    )


# %%
# Redo the plot above but display valence as as deviations from mean, so there is no difference between positive and negative standard deviations.
# For the emobank dataset, project onto arousal and valence and display results in a scatterplot. Compare to a scatterplot with original scores.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

emobank_embeddings = corpora["Emobank"]["embedding"]
# Make valence distance from mean
emobank_valence = np.abs(corpora["Emobank"]["dataframe"]["valence"] - np.mean(corpora["Emobank"]["dataframe"]["valence"]))
emobank_arousal = corpora["Emobank"]["dataframe"]["arousal"]
# Make valence distance from mean
valence_cv = concept_vectors["Emobank"]["negative_to_positive"]
arousal_cv = arousal_concept_vectors["Emobank"]["negative_to_positive"]

valence_projections = np.asarray(valence_cv.project(emobank_embeddings)).ravel()
valence_projections = np.abs(valence_projections - np.mean(valence_projections))
arousal_projections = np.asarray(arousal_cv.project(emobank_embeddings)).ravel()

# Align lengths and mask NaNs for projected arrays
m_proj = min(len(valence_projections), len(arousal_projections))
vp = valence_projections[:m_proj]
ap = arousal_projections[:m_proj]
mask_proj = (~np.isnan(vp)) & (~np.isnan(ap))

slope_proj = intercept_proj = rvalue_proj = p_proj = stderr_proj = np.nan
if mask_proj.sum() > 1:
    # linear regression
    reg_proj = linregress(vp[mask_proj], ap[mask_proj])
    slope_proj, intercept_proj, rvalue_proj, p_proj, stderr_proj = (
        reg_proj.slope,
        reg_proj.intercept,
        reg_proj.rvalue,
        reg_proj.pvalue,
        reg_proj.stderr,
    )

print(f"Regression (projected): slope={slope_proj:.4f}, intercept={intercept_proj:.4f}, r={rvalue_proj:.4f}, p={p_proj:.4e}, stderr={stderr_proj:.4f}")

# Scatterplot of projected valence vs. projected arousal with linear regression line
x_proj = vp[mask_proj]
y_proj = ap[mask_proj]
plt.figure(figsize=(10, 6))
plt.scatter(x_proj, y_proj, alpha=0.5)
# add linear regression line (no scatter from regplot)
if mask_proj.sum() > 1:
    sns.regplot(x=x_proj, y=y_proj, scatter=False, ci=None, color="black", line_kws={"linewidth": 2, "alpha": 0.8})
    txt = f"slope={slope_proj:.3f}\nintercept={intercept_proj:.3f}\nr={rvalue_proj:.3f}\np={p_proj:.2e}"
else:
    txt = "Insufficient points for regression"
plt.xlabel("Projected Valence (|deviation from mean|, Negative→Positive)")
plt.ylabel("Projected Arousal (Negative→Positive)")
plt.title("Scatterplot of Projected Valence vs. Projected Arousal (Emobank Corpus)")
plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes,
         ha="left", va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.show()

# Prepare original scores (coerce to numeric), align lengths and mask NaNs
orig_v = pd.to_numeric(emobank_valence, errors="coerce").reset_index(drop=True).values
orig_a = pd.to_numeric(emobank_arousal, errors="coerce").reset_index(drop=True).values
m_orig = min(len(orig_v), len(orig_a))
ov = orig_v[:m_orig]
oa = orig_a[:m_orig]
mask_orig = (~np.isnan(ov)) & (~np.isnan(oa))

slope_orig = intercept_orig = rvalue_orig = p_orig = stderr_orig = np.nan
if mask_orig.sum() > 1:
    reg_orig = linregress(ov[mask_orig], oa[mask_orig])
    slope_orig, intercept_orig, rvalue_orig, p_orig, stderr_orig = (
        reg_orig.slope,
        reg_orig.intercept,
        reg_orig.rvalue,
        reg_orig.pvalue,
        reg_orig.stderr,
    )

print(f"Regression (original): slope={slope_orig:.4f}, intercept={intercept_orig:.4f}, r={rvalue_orig:.4f}, p={p_orig:.4e}, stderr={stderr_orig:.4f}")

# Scatterplot of original valence vs. original arousal with linear regression line
x_orig = ov[mask_orig]
y_orig = oa[mask_orig]
plt.figure(figsize=(10, 6))
plt.scatter(x_orig, y_orig, alpha=0.5)
if mask_orig.sum() > 1:
    sns.regplot(x=x_orig, y=y_orig, scatter=False, ci=None, color="black", line_kws={"linewidth": 2, "alpha": 0.8})
    txto = f"slope={slope_orig:.3f}\nintercept={intercept_orig:.3f}\nr={rvalue_orig:.3f}\np={p_orig:.2e}"
else:
    txto = "Insufficient points for regression"
plt.xlabel("Original Valence Score")
plt.ylabel("Original Arousal Score")
plt.title("Scatterplot of Original Valence vs Original Arousal (Emobank Corpus)")
plt.text(0.05, 0.95, txto, transform=plt.gca().transAxes,
         ha="left", va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.show()

# %%
### FACEBOOK PLOTS ####
# Redo the plot above but display valence as as deviations from mean, so there is no difference between positive and negative standard deviations.
# For the emobank dataset, project onto arousal and valence and display results in a scatterplot. Compare to a scatterplot with original scores.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

emobank_embeddings = corpora["Facebook"]["embedding"]
# Make valence distance from mean
emobank_valence = np.abs(corpora["Facebook"]["dataframe"]["valence"] - np.mean(corpora["Facebook"]["dataframe"]["valence"]))
emobank_arousal = corpora["Facebook"]["dataframe"]["arousal"]
# Make valence distance from mean
valence_cv = concept_vectors["Facebook"]["negative_to_positive"]
arousal_cv = arousal_concept_vectors["Facebook"]["negative_to_positive"]

valence_projections = np.asarray(valence_cv.project(emobank_embeddings)).ravel()
valence_projections = np.abs(valence_projections - np.mean(valence_projections))
arousal_projections = np.asarray(arousal_cv.project(emobank_embeddings)).ravel()

# Align lengths and mask NaNs for projected arrays
m_proj = min(len(valence_projections), len(arousal_projections))
vp = valence_projections[:m_proj]
ap = arousal_projections[:m_proj]
mask_proj = (~np.isnan(vp)) & (~np.isnan(ap))

slope_proj = intercept_proj = rvalue_proj = p_proj = stderr_proj = np.nan
if mask_proj.sum() > 1:
    # linear regression
    reg_proj = linregress(vp[mask_proj], ap[mask_proj])
    slope_proj, intercept_proj, rvalue_proj, p_proj, stderr_proj = (
        reg_proj.slope,
        reg_proj.intercept,
        reg_proj.rvalue,
        reg_proj.pvalue,
        reg_proj.stderr,
    )

print(f"Regression (projected): slope={slope_proj:.4f}, intercept={intercept_proj:.4f}, r={rvalue_proj:.4f}, p={p_proj:.4e}, stderr={stderr_proj:.4f}")

# Scatterplot of projected valence vs. projected arousal with linear regression line
x_proj = vp[mask_proj]
y_proj = ap[mask_proj]
plt.figure(figsize=(10, 6))
plt.scatter(x_proj, y_proj, alpha=0.5)
# add linear regression line (no scatter from regplot)
if mask_proj.sum() > 1:
    sns.regplot(x=x_proj, y=y_proj, scatter=False, ci=None, color="black", line_kws={"linewidth": 2, "alpha": 0.8})
    txt = f"slope={slope_proj:.3f}\nintercept={intercept_proj:.3f}\nr={rvalue_proj:.3f}\np={p_proj:.2e}"
else:
    txt = "Insufficient points for regression"
plt.xlabel("Projected Valence (|deviation from mean|, Negative→Positive)")
plt.ylabel("Projected Arousal (Negative→Positive)")
plt.title("Scatterplot of Projected Valence vs. Projected Arousal (Emobank Corpus)")
plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes,
         ha="left", va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.show()

# Prepare original scores (coerce to numeric), align lengths and mask NaNs
orig_v = pd.to_numeric(emobank_valence, errors="coerce").reset_index(drop=True).values
orig_a = pd.to_numeric(emobank_arousal, errors="coerce").reset_index(drop=True).values
m_orig = min(len(orig_v), len(orig_a))
ov = orig_v[:m_orig]
oa = orig_a[:m_orig]
mask_orig = (~np.isnan(ov)) & (~np.isnan(oa))

slope_orig = intercept_orig = rvalue_orig = p_orig = stderr_orig = np.nan
if mask_orig.sum() > 1:
    reg_orig = linregress(ov[mask_orig], oa[mask_orig])
    slope_orig, intercept_orig, rvalue_orig, p_orig, stderr_orig = (
        reg_orig.slope,
        reg_orig.intercept,
        reg_orig.rvalue,
        reg_orig.pvalue,
        reg_orig.stderr,
    )

print(f"Regression (original): slope={slope_orig:.4f}, intercept={intercept_orig:.4f}, r={rvalue_orig:.4f}, p={p_orig:.4e}, stderr={stderr_orig:.4f}")

# Scatterplot of original valence vs. original arousal with linear regression line
x_orig = ov[mask_orig]
y_orig = oa[mask_orig]
plt.figure(figsize=(10, 6))
plt.scatter(x_orig, y_orig, alpha=0.5)
if mask_orig.sum() > 1:
    sns.regplot(x=x_orig, y=y_orig, scatter=False, ci=None, color="black", line_kws={"linewidth": 2, "alpha": 0.8})
    txto = f"slope={slope_orig:.3f}\nintercept={intercept_orig:.3f}\nr={rvalue_orig:.3f}\np={p_orig:.2e}"
else:
    txto = "Insufficient points for regression"
plt.xlabel("Original Valence Score")
plt.ylabel("Original Arousal Score")
plt.title("Scatterplot of Original Valence vs Original Arousal (Emobank Corpus)")
plt.text(0.05, 0.95, txto, transform=plt.gca().transAxes,
         ha="left", va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.show()
# %%
# Check interrater correlation of arousal and valence in Emobank


# %%
# Create a dictionary of corpora dictionaries using dominance labels
dominance_corpora = {
    "Emobank": {"dominance_label": None, "embedding": corpora["Emobank"]["embedding"], "dataframe": Emobank},
}

import numpy as np
for corpus_name, corpus_info in dominance_corpora.items():
    df = corpus_info["dataframe"]
    if "dominance" in df.columns:
        mean_dominance = df["dominance"].mean()
        std_dominance = df["dominance"].std()
        positive_threshold = mean_dominance + std_dominance
        negative_threshold = mean_dominance - std_dominance
        def label_dominance(x):
            if x >= positive_threshold:
                return "positive"
            elif x <= negative_threshold:
                return "negative"
            else:
                return "neutral"

        corpus_info["dominance_label"] = df["dominance"].apply(label_dominance)
    else:
        print(f"Dominance column not found in {corpus_name} dataframe.")
    # For debugging, print the distribution of labels
    print(f"{corpus_name} dominance label distribution:\n{corpus_info['dominance_label'].value_counts()}\n")


dominance_concept_vectors = {}
for corpus_name, corpus_info in dominance_corpora.items():
    embeddings = np.asarray(corpus_info["embedding"])
    labels = np.asarray(corpus_info["dominance_label"])

    # helper to get embeddings for a given label
    def get_emb(lbl):
        mask = labels == lbl
        return embeddings[mask]

    pos_embeddings = get_emb("positive")
    neg_embeddings = get_emb("negative")
    neu_embeddings = get_emb("neutral")

    cv_dict = {}
    pairs = {
        "negative_to_positive": (neg_embeddings, pos_embeddings),
        "neutral_to_positive": (neu_embeddings, pos_embeddings),
        "negative_to_neutral": (neg_embeddings, neu_embeddings),
    }

    for pair_name, (src_emb, tgt_emb) in pairs.items():
        src_count = 0 if src_emb.size == 0 else src_emb.shape[0]
        tgt_count = 0 if tgt_emb.size == 0 else tgt_emb.shape[0]
        if src_count == 0 or tgt_count == 0:
            print(f"Skipping {corpus_name} {pair_name}: insufficient examples (src={src_count}, tgt={tgt_count})")
            cv_dict[pair_name] = None
            continue

        cv = ConceptVector(normalize=True)
        cv.fit(src_emb, tgt_emb)
        cv_dict[pair_name] = cv
        print(f"Concept vector for {corpus_name} ({pair_name}) created.")

    dominance_concept_vectors[corpus_name] = cv_dict

# %%
# Count-normalized histogram for dominance corpora using dominance concept vectors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# choose corpus and direction from dominance_concept_vectors / dominance_corpora
train_corpus = "Emobank"
direction = "negative_to_positive"

cv = dominance_concept_vectors.get(train_corpus, {}).get(direction)
if cv is None:
    print(f"No concept vector found for {train_corpus} ({direction}). Skipping plot.")
else:
    projections = cv.project(dominance_corpora[train_corpus]["embedding"])
    labels = dominance_corpora[train_corpus]["dominance_label"]

    df_plot = pd.DataFrame({"projection": np.asarray(projections).ravel(), "dominance_label": labels}).dropna()

    plt.figure(figsize=(10, 6))
    palette = {"negative": "tab:red", "neutral": "tab:blue", "positive": "tab:green"}
    bins = np.linspace(-2, 2, 41)

    # Plot histogram for each dominance label with alpha=0.5 (counts)
    for label in ["negative", "neutral", "positive"]:
        subset = df_plot[df_plot["dominance_label"] == label]["projection"]
        if subset.shape[0] > 1:
            sns.histplot(
                subset,
                bins=bins,
                color=palette[label],
                label=label,
                alpha=0.5,
                stat="count",
                kde=False,
            )
        else:
            sns.rugplot(subset, color=palette[label], height=0.05)

    plt.xlabel("Projection onto Negative → Positive Dominance Concept Vector")
    plt.ylabel("Count")
    plt.title(f"Distribution of Projections by Dominance Label — {train_corpus} ({direction})")
    plt.legend(title="Dominance Label")
    plt.xlim(-2, 2)
    plt.tight_layout()
    plt.show()

# %%
# Scatterplot between dataframe["dominance"] and projections onto the negative_to_positive concept vector
import matplotlib.pyplot as plt
projections = dominance_concept_vectors["Emobank"]["negative_to_positive"].project(dominance_corpora["Emobank"]["embedding"])
dominance_scores = dominance_corpora["Emobank"]["dataframe"]["dominance"].reset_index(drop=True)

# Ensure projections is a 1D numpy array aligned with dominance_scores
projections = np.asarray(projections).ravel()

df_scatter = pd.DataFrame({"dominance": dominance_scores, "projection": projections})

# Compute mean and std and create a hue/category column based on std thresholds
mean_d = df_scatter["dominance"].mean()
std_d = df_scatter["dominance"].std()

def dominance_std_category(x):
    if x >= mean_d + std_d:
        return "Positive (>= mean + 1σ)"
    elif x <= mean_d - std_d:
        return "Negative (<= mean - 1σ)"
    else:
        return "Neutral (Within 1σ)"

df_scatter["dominance_hue"] = df_scatter["dominance"].apply(dominance_std_category)

# Plot with hue and show mean / mean±std lines
plt.figure(figsize=(10, 6))
palette = {"Positive (>= mean + 1σ)": "tab:green", "Neutral (Within 1σ)": "tab:blue", "Negative (<= mean - 1σ)": "tab:red"}
sns.scatterplot(data=df_scatter, x="dominance", y="projection", hue="dominance_hue", palette=palette, alpha=0.7)

plt.xlabel("Dominance Score")
plt.ylabel("Projection onto Negative to Positive Concept Vector")
plt.title("Scatterplot of Dominance Scores vs. Projections (Emobank Vector - Emobank Corpus) — hue by ±1σ from mean")
plt.legend(title="Dominance (std category)", loc="best")
plt.tight_layout()
plt.show()

# Print correlation
mask = (~np.isnan(df_scatter["dominance"])) & (~np.isnan(df_scatter["projection"]))
corr = np.nan
if mask.sum() > 1:
    corr = np.corrcoef(df_scatter["dominance"][mask], df_scatter["projection"][mask])[0, 1]
print(f"Pearson correlation between dominance scores and projections: {np.nan_to_num(corr):.4f}")


# %%
# Plot emobank dominance vs valence
import matplotlib.pyplot as plt
import numpy as np
emobank_dominance = corpora["Emobank"]["dataframe"]["dominance"]
emobank_valence = corpora["Emobank"]["dataframe"]["valence"]
plt.figure(figsize=(10, 6))
plt.scatter(emobank_valence, emobank_dominance, alpha=0.5)
plt.xlabel("Valence Score")
plt.ylabel("Dominance Score")
plt.title("Scatterplot of Valence vs. Dominance Scores (Emobank Corpus)")
# compute correlation
mask = (~np.isnan(emobank_valence)) & (~np.isnan(emobank_dominance))
corr = np.nan
if mask.sum() > 1:
    corr = np.corrcoef(emobank_valence[mask], emobank_dominance[mask])[0, 1]    
plt.text(0.05, 0.95, f"Pearson r = {np.nan_to_num(corr):.3f}", transform=plt.gca().transAxes,
         ha="left", va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.show()
# %%
# Plot projected dominance vs projected valence
import matplotlib.pyplot as plt
import numpy as np
emobank_embeddings = corpora["Emobank"]["embedding"]
valence_cv = concept_vectors["Emobank"]["negative_to_positive"]
dominance_cv = dominance_concept_vectors["Emobank"]["negative_to_positive"]
valence_projections = np.asarray(valence_cv.project(emobank_embeddings)).ravel()
dominance_projections = np.asarray(dominance_cv.project(emobank_embeddings)).ravel()
plt.figure(figsize=(10, 6))
plt.scatter(valence_projections, dominance_projections, alpha=0.5)
plt.xlabel("Projected Valence (Negative to Positive Concept Vector)")
plt.ylabel("Projected Dominance (Negative to Positive Concept Vector)")
plt.title("Scatterplot of Projected Valence vs. Projected Dominance (Emobank Corpus)")
# compute correlation
mask = (~np.isnan(valence_projections)) & (~np.isnan(dominance_projections))
corr = np.nan
if mask.sum() > 1:
    corr = np.corrcoef(valence_projections[mask], dominance_projections[mask])[0, 1]    
plt.text(0.05, 0.95, f"Pearson r = {np.nan_to_num(corr):.3f}", transform=plt.gca().transAxes,
            ha="left", va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.show()
# %%
# Show a distribution of emobank dominance scores.
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
emobank_dominance = corpora["Emobank"]["dataframe"]["dominance"]
plt.figure(figsize=(10, 6))
sns.histplot(emobank_dominance.dropna(), bins=100, kde=True, color="skyblue")
plt.xlabel("Dominance Score")
plt.ylabel("Count")
plt.title("Distribution of Dominance Scores (Emobank Corpus)")
plt.tight_layout()
plt.show()

# %%
# Create a 3d plot of projected valence, arousal, and dominance for Emobank
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
emobank_embeddings = corpora["Emobank"]["embedding"]
valence_cv = concept_vectors["Emobank"]["negative_to_positive"]
arousal_cv = arousal_concept_vectors["Emobank"]["negative_to_positive"]
dominance_cv = dominance_concept_vectors["Emobank"]["negative_to_positive"]
valence_projections = np.asarray(valence_cv.project(emobank_embeddings)).ravel()
arousal_projections = np.asarray(arousal_cv.project(emobank_embeddings)).ravel()
dominance_projections = np.asarray(dominance_cv.project(emobank_embeddings)).ravel()

# Sample 100 random points
sample_size = min(1000, len(valence_projections))
indices = np.random.choice(len(valence_projections), sample_size, replace=False)
valence_sample = valence_projections[indices]
arousal_sample = arousal_projections[indices]
dominance_sample = dominance_projections[indices]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(valence_sample, arousal_sample, dominance_sample, alpha=0.5)
ax.set_xlabel("Projected Valence")
ax.set_ylabel("Projected Arousal")
ax.set_zlabel("Projected Dominance")
ax.set_title("3D Scatterplot of Projected Valence, Arousal, and Dominance (Emobank Corpus, n=1000)")
plt.tight_layout()
plt.show()


# %%
# Create a 3d plot of original valence, arousal, and dominance for Emobank corpus
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
emobank_valence = corpora["Emobank"]["dataframe"]["valence"].values
emobank_arousal = corpora["Emobank"]["dataframe"]["arousal"].values
emobank_dominance = corpora["Emobank"]["dataframe"]["dominance"].values

# Sample 100 random points
sample_size = min(1000, len(emobank_valence))
indices = np.random.choice(len(emobank_valence), sample_size, replace=False)
valence_sample = emobank_valence[indices]
arousal_sample = emobank_arousal[indices]
dominance_sample = emobank_dominance[indices]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(valence_sample, arousal_sample, dominance_sample, alpha=0.5)
ax.set_xlabel("Original Valence")
ax.set_ylabel("Original Arousal")
ax.set_zlabel("Original Dominance")
ax.set_title("3D Scatterplot of Original Valence, Arousal, and Dominance (Emobank Corpus, n=1000)")
plt.tight_layout()
plt.show()


# %%
