import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as C
from .preprocessing import QuantileClipper

WORD_RE = r"[a-z0-9]+(?:'[a-z]+)?"
NEGATION_RE = r"\b(?:not|no|never|nothing|nobody|none|nor|cannot|without)\b|n't\b"
BBCODE_RE = r"\[/?[a-zA-Z0-9*]+(?:=[^\]]*)?\]"
APOSTROPHE_RE = "[’‘`´]"
TOKEN_PATTERN = r"(?u)\b\w[\w']*\w\b"

ENGINEERED_LOG = [
    "word_count",
    "char_count",
    "exclamation_count",
    "question_count",
    "negation_count",
    "reviews_per_game_owned",
    "review_age_days",
]
ENGINEERED_LINEAR = ["avg_word_length", "uppercase_ratio", "negation_ratio"]
ENGINEERED_BINARY = ["has_formatting"]
ENGINEERED = ENGINEERED_LOG + ENGINEERED_LINEAR + ENGINEERED_BINARY

LOG_FEATURES = list(C.NUM_FEATURES) + ENGINEERED_LOG
LINEAR_FEATURES = list(ENGINEERED_LINEAR)
BINARY_FEATURES = list(C.BOOL_FEATURES) + ENGINEERED_BINARY
MISSING_FEATURES = list(C.NUM_FEATURES)

FEATURE_DEFINITIONS = pd.DataFrame(
    [
        ("word_count", "number of words in review_clean", "effort and emotion: complaints tend to be explained, praise is often short"),
        ("char_count", "number of characters in review_clean", "alternative length measure, checked for redundancy with word_count"),
        ("avg_word_length", "letters / words", "writing style: short slang words vs. elaborated text"),
        ("exclamation_count", "number of '!' in the original text", "emotional intensity"),
        ("question_count", "number of '?' in the original text", "rhetorical questions are typical for complaints"),
        ("uppercase_ratio", "upper-case letters / all letters in the original text", "'shouting' in capitals"),
        ("negation_count", "occurrences of not, no, never, nothing, n't, ...", "negations flip the meaning of positive words"),
        ("negation_ratio", "negation_count / word_count", "negation density independent of length"),
        ("has_formatting", "1 if the original text contains Steam BBCode tags", "formatted reviews are structured, detailed reviews"),
        ("reviews_per_game_owned", "author_num_reviews / (author_num_games_owned + 1)", "how critical / active a reviewer is relative to library size"),
        ("review_age_days", "days between publication and the end of data collection", "older vs. recent reviews; game updates change opinions over time"),
    ],
    columns=["feature", "definition", "why it can matter"],
).set_index("feature")

LEAKAGE_CANDIDATES = pd.DataFrame(
    [
        ("playtime_ratio", "author_playtime_at_review / author_playtime_forever", "author_playtime_forever is measured after the review"),
        ("was_edited", "1 if updated_at > created_at", "an edit happens after publication"),
        ("log_votes_helpful", "log1p(votes_helpful)", "votes are given by other users after publication"),
        ("log_comment_count", "log1p(comment_count)", "comments appear after publication"),
        ("days_since_last_played", "end of collection - last_played_at", "describes behaviour after the review"),
    ],
    columns=["candidate", "definition", "why it is rejected"],
).set_index("candidate")


def text_preprocessor(text):
    return text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")


def normalize_apostrophes(series):
    return series.fillna("").astype(str).str.replace(APOSTROPHE_RE, "'", regex=True)


def word_count(series):
    return normalize_apostrophes(series).str.count(WORD_RE).astype("float64")


def add_engineered_features(df):
    out = df.copy()
    clean = normalize_apostrophes(out[C.TEXT_CLEAN])
    raw = out[C.TEXT_RAW].fillna("").astype(str)
    words = clean.str.count(WORD_RE).astype("float64")
    safe_words = words.where(words > 0)
    raw_letters = raw.str.count(r"[A-Za-z]").astype("float64")
    out["word_count"] = words
    out["char_count"] = clean.str.len().astype("float64")
    out["avg_word_length"] = clean.str.count(r"[a-z]") / safe_words
    out["exclamation_count"] = raw.str.count("!").astype("float64")
    out["question_count"] = raw.str.count(r"\?").astype("float64")
    out["uppercase_ratio"] = raw.str.count(r"[A-Z]") / raw_letters.where(raw_letters > 0)
    out["negation_count"] = clean.str.count(NEGATION_RE).astype("float64")
    out["negation_ratio"] = out["negation_count"] / safe_words
    out["has_formatting"] = raw.str.contains(BBCODE_RE, regex=True).astype("float64")
    out["reviews_per_game_owned"] = out["author_num_reviews"] / (out["author_num_games_owned"] + 1)
    snapshot = pd.Timestamp(C.SNAPSHOT_END_UNIX, unit="s")
    out["review_age_days"] = (snapshot - pd.to_datetime(out["created_at"])).dt.total_seconds() / 86400
    return out


def add_leakage_candidates(df):
    out = df.copy()
    forever = out["author_playtime_forever"]
    out["playtime_ratio"] = out["author_playtime_at_review"] / forever.where(forever > 0)
    out["was_edited"] = (out["updated_at"] > out["created_at"]).astype("float64")
    out["log_votes_helpful"] = np.log1p(out["votes_helpful"])
    out["log_comment_count"] = np.log1p(out["comment_count"])
    snapshot = pd.Timestamp(C.SNAPSHOT_END_UNIX, unit="s")
    out["days_since_last_played"] = (snapshot - pd.to_datetime(out["last_played_at"])).dt.total_seconds() / 86400
    return out


def build_text_pipeline(k="all", svd_components=None, min_df=5, max_features=100_000):
    steps = [(
        "tfidf",
        TfidfVectorizer(
            preprocessor=text_preprocessor,
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=0.9,
            sublinear_tf=True,
            max_features=max_features,
            dtype=np.float32,
        ),
    )]
    if svd_components:
        steps.append(("svd", TruncatedSVD(n_components=svd_components, random_state=C.RANDOM_STATE)))
    elif k != "all":
        steps.append(("chi2", SelectKBest(chi2, k=k)))
    return Pipeline(steps)


def numeric_pipeline(log):
    steps = [("clip", QuantileClipper(0.001, 0.999))]
    if log:
        steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    steps += [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    return Pipeline(steps)


def tabular_transformers(log_features, linear_features, binary_features, missing_features):
    transformers = []
    if log_features:
        transformers.append(("log", numeric_pipeline(True), list(log_features)))
    if linear_features:
        transformers.append(("linear", numeric_pipeline(False), list(linear_features)))
    if missing_features:
        transformers.append(("missing", MissingIndicator(features="all"), list(missing_features)))
    if binary_features:
        transformers.append(("binary", SimpleImputer(strategy="most_frequent"), list(binary_features)))
    return transformers


def build_feature_pipeline(
    include_text=True,
    text_k="all",
    text_svd=None,
    log_features=(),
    linear_features=(),
    binary_features=(),
    missing_features=(),
    include_games=True,
    tabular_pca=None,
    min_game_frequency=20,
    text_min_df=5,
    text_max_features=100_000,
):
    transformers = []
    if include_text:
        transformers.append(("text", build_text_pipeline(text_k, text_svd, text_min_df, text_max_features), C.TEXT_CLEAN))
    continuous = tabular_transformers(log_features, linear_features, (), ())
    if continuous and tabular_pca:
        columns = list(log_features) + list(linear_features)
        transformers.append(("continuous_pca", Pipeline([
            ("continuous", ColumnTransformer(continuous, remainder="drop", sparse_threshold=0.0)),
            ("pca", PCA(n_components=tabular_pca, random_state=C.RANDOM_STATE)),
        ]), columns))
    else:
        transformers.extend(continuous)
    transformers.extend(tabular_transformers((), (), binary_features, missing_features))
    if include_games:
        transformers.append(("game", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                   min_frequency=min_game_frequency,
                                                   sparse_output=True), list(C.CAT_FEATURES)))
    if not transformers:
        raise ValueError("The feature pipeline needs at least one feature group.")
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)


def probe_model(sparse=True):
    if sparse:
        return LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=1000)
    return LogisticRegression(class_weight="balanced", max_iter=2000)


def validation_scores(model, X, y):
    y = np.asarray(y)
    classes = list(model.classes_)
    proba = model.predict_proba(X)
    pred = model.predict(X)
    return {
        "f1_macro": f1_score(y, pred, average="macro"),
        "recall_negative": recall_score(y, pred, pos_label=0),
        "pr_auc_negative": average_precision_score(y == 0, proba[:, classes.index(0)]),
        "roc_auc": roc_auc_score(y, proba[:, classes.index(1)]),
    }
