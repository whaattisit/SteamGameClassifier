"""Reusable preprocessing functions for the Steam Reviews project.

All functions are pure: they take a DataFrame, return a new DataFrame
(plus a small log table where useful) and never modify the input.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from . import config as C

_BOOL_MAP = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}
_BBCODE_RE = r"\[/?[a-z0-9*]+(?:=[^\]]*)?\]"
_URL_RE = r"(?:https?://|www\.)\S+"


# ====================================================================== helpers
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the exported pandas index ('Unnamed: 0') and replace dots in names.

    'author.playtime_forever' -> 'author_playtime_forever', because dotted
    names break attribute access and some plotting/statistics libraries.
    """
    keep = [c for c in df.columns if not str(c).startswith("Unnamed")]
    out = df.loc[:, keep]
    return out.rename(columns=lambda c: str(c).strip().replace(".", "_"))


def parse_bool(series: pd.Series) -> pd.Series:
    """'True'/'False'/'1'/'0' (any case) -> 1.0 / 0.0, anything else -> NaN."""
    lowered = series.astype("string").str.strip().str.lower()
    mapped = lowered.map(_BOOL_MAP, na_action="ignore")
    return pd.to_numeric(mapped, errors="coerce").astype("float64")


def _to_builtin(obj):
    """Make numpy / pandas scalars JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


# ====================================================================== stage 0
def scan_full_dataset(
    csv_path: Path = C.RAW_CSV,
    frac: float = C.SAMPLE_FRAC,
    chunksize: int = C.CHUNKSIZE,
    seed: int = C.RANDOM_STATE,
    language: str = C.LANGUAGE,
    verbose: bool = True,
):
    """Read the 8 GB CSV once, in chunks.

    During the single pass it collects statistics of the FULL dataset (rows,
    missing values, languages, class balance, duplicates, per-game aggregates)
    and draws a simple random sample of `frac` of the reviews in `language`
    (every review gets an independent coin flip with probability `frac`).

    Returns (sample_df, stats_dict, game_stats_df). Every column of the sample
    is still a raw string; types are fixed later by `convert_types`.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Download the dataset from {C.DATASET_URL} "
            "and unzip steam_reviews.csv into data/raw/ (see README.md)."
        )

    rng = np.random.default_rng(seed)
    t0 = time.time()
    total_rows = 0
    columns: list[str] | None = None
    n_raw_columns = 0
    missing = None
    lang_counts = pd.Series(dtype="float64")
    target_all = {"recommended": 0, "not_recommended": 0, "missing": 0}
    target_lang = {"recommended": 0, "not_recommended": 0, "missing": 0}
    game_parts = []
    review_ids = []
    author_app_hashes = []
    samples = []

    reader = pd.read_csv(
        csv_path,
        dtype=str,
        chunksize=chunksize,
        on_bad_lines="skip",
        encoding="utf-8",
    )
    for i, chunk in enumerate(reader):
        if columns is None:
            n_raw_columns = chunk.shape[1]
        chunk = normalize_column_names(chunk)
        if columns is None:
            columns = list(chunk.columns)
        total_rows += len(chunk)

        miss = chunk.isna().sum()
        missing = miss if missing is None else missing.add(miss, fill_value=0)

        lang = chunk["language"].fillna("<missing>")
        lang_counts = lang_counts.add(lang.value_counts(), fill_value=0)
        is_lang = (lang == language).to_numpy()

        rec = parse_bool(chunk["recommended"])
        for store, mask in ((target_all, np.ones(len(chunk), bool)), (target_lang, is_lang)):
            r = rec[mask]
            store["recommended"] += int((r == 1).sum())
            store["not_recommended"] += int((r == 0).sum())
            store["missing"] += int(r.isna().sum())

        rec0 = rec.fillna(0).to_numpy()
        game_parts.append(
            pd.DataFrame(
                {
                    "app_name": chunk["app_name"].fillna("<missing>").to_numpy(),
                    "n_all": 1,
                    "pos_all": rec0,
                    "n_lang": is_lang.astype(int),
                    "pos_lang": rec0 * is_lang,
                }
            )
            .groupby("app_name")
            .sum()
        )

        rid = pd.to_numeric(chunk["review_id"], errors="coerce").dropna()
        review_ids.append(rid.to_numpy(dtype="float64"))
        pair = chunk[["author_steamid", "app_id"]].dropna()
        author_app_hashes.append(
            pd.util.hash_pandas_object(pair, index=False).to_numpy()
        )

        lang_chunk = chunk.loc[is_lang]
        keep = rng.random(len(lang_chunk)) < frac
        samples.append(lang_chunk.loc[keep])

        if verbose and (i + 1) % 10 == 0:
            print(f"  {total_rows:>12,} rows read  ({time.time() - t0:,.0f} s)")

    if columns is None:
        raise ValueError(f"{csv_path} is empty.")

    sample = pd.concat(samples, ignore_index=True)
    game_stats = pd.concat(game_parts).groupby(level=0).sum()
    game_stats = game_stats.sort_values("n_all", ascending=False)
    game_stats.index.name = "app_name"

    rid_all = np.concatenate(review_ids) if review_ids else np.array([])
    pair_all = np.concatenate(author_app_hashes) if author_app_hashes else np.array([])
    stats = {
        "total_rows": total_rows,
        "n_columns_original": n_raw_columns,
        "columns": columns,
        "missing_per_column": missing.astype(int).to_dict(),
        "language_counts": lang_counts.sort_values(ascending=False).astype(int).to_dict(),
        "target_all_languages": target_all,
        "target_selected_language": target_lang,
        "language": language,
        "sample_frac": frac,
        "random_state": seed,
        "n_sampled": len(sample),
        "n_games": int(game_stats.shape[0]),
        "duplicate_review_ids": int(rid_all.size - np.unique(rid_all).size),
        "duplicate_author_game_pairs": int(pair_all.size - np.unique(pair_all).size),
        "scan_seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"Done: {total_rows:,} rows, {len(sample):,} sampled, "
              f"{stats['scan_seconds']:,} s")
    return sample, _to_builtin(stats), game_stats


def load_or_scan(force: bool = False, **scan_kwargs):
    """Use the cached sample if it exists, otherwise scan the CSV and cache it."""
    cached = [C.SAMPLE_RAW_PARQUET, C.FULL_STATS_JSON, C.GAME_STATS_CSV]
    if not force and all(p.exists() for p in cached):
        print("Loading cached sample and statistics from data/interim/")
        sample = pd.read_parquet(C.SAMPLE_RAW_PARQUET)
        stats = json.loads(C.FULL_STATS_JSON.read_text(encoding="utf-8"))
        game_stats = pd.read_csv(C.GAME_STATS_CSV, index_col="app_name")
        return sample, stats, game_stats

    print("Scanning the full CSV (first run only, takes several minutes)...")
    sample, stats, game_stats = scan_full_dataset(**scan_kwargs)
    C.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(C.SAMPLE_RAW_PARQUET, index=False)
    C.FULL_STATS_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    game_stats.to_csv(C.GAME_STATS_CSV)
    return sample, stats, game_stats


# ====================================================================== stage 1
def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Raw strings -> proper dtypes. Unparseable values become NaN."""
    out = df.copy()
    for col in C.BOOL_COLS:
        out[col] = parse_bool(out[col])
    for col in C.COUNT_COLS + C.TIMESTAMP_COLS + ["weighted_vote_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in ["review_id", "app_id"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    # Steam IDs (~7.6e16) do not fit into float64 exactly -> keep them as text.
    # pandas >= 3 "str" dtype: missing values stay NaN, same dtype as after a parquet round trip.
    for col in ["author_steamid", "app_name", "language", "review"]:
        out[col] = out[col].astype("str")
    return out


def remove_duplicates(df: pd.DataFrame):
    """Exact copies, repeated review_id, and several reviews of one game by one author.

    Steam allows only one review per (author, game); when the pair repeats we
    keep the most recent version (largest timestamp_created).
    """
    log = []
    n0 = len(df)
    out = df.drop_duplicates()
    log.append(("exact duplicate rows", n0 - len(out)))

    out = out.sort_values("timestamp_created", ascending=False, na_position="last")
    n1 = len(out)
    out = out.drop_duplicates(subset=["review_id"], keep="first")
    log.append(("repeated review_id", n1 - len(out)))

    n2 = len(out)
    out = out.drop_duplicates(subset=["author_steamid", "app_id"], keep="first")
    log.append(("same author reviewed the same game again (older copy)", n2 - len(out)))

    out = out.sort_index()
    report = pd.DataFrame(log, columns=["rule", "rows_removed"])
    return out, report


def fix_invalid_values(df: pd.DataFrame):
    """Impossible values are data errors -> set to NaN (or repaired by a domain rule).

    Returns (clean_df, log_df). Real but extreme values are NOT touched here;
    they are analysed in the outlier section.
    """
    out = df.copy()
    log = []

    def to_nan(col, mask, rule):
        mask = mask.fillna(False) if hasattr(mask, "fillna") else mask
        n = int(mask.sum())
        if n:
            out.loc[mask, col] = np.nan
        log.append((col, rule, n))

    for col in C.COUNT_COLS:
        to_nan(col, out[col] >= C.OVERFLOW_THRESHOLD, "overflow marker (>= 2^31)")
        to_nan(col, out[col] < 0, "negative count")

    to_nan("author_num_games_owned", out["author_num_games_owned"] > C.MAX_PLAUSIBLE_GAMES_OWNED,
           f"> {C.MAX_PLAUSIBLE_GAMES_OWNED:,} games (more than Steam had)")
    to_nan("author_num_reviews", out["author_num_reviews"] > C.MAX_PLAUSIBLE_NUM_REVIEWS,
           f"> {C.MAX_PLAUSIBLE_NUM_REVIEWS:,} reviews")
    to_nan("author_playtime_last_two_weeks",
           out["author_playtime_last_two_weeks"] > C.MAX_PLAYTIME_TWO_WEEKS_MIN,
           "more than 20,160 minutes (=14 days) played in 14 days")
    to_nan("weighted_vote_score",
           (out["weighted_vote_score"] < 0) | (out["weighted_vote_score"] > 1),
           "outside the valid range [0, 1]")

    # playtime at review cannot exceed the total playtime collected later
    both = out["author_playtime_at_review"].notna() & out["author_playtime_forever"].notna()
    bad = both & (out["author_playtime_at_review"] > out["author_playtime_forever"])
    out.loc[bad, "author_playtime_at_review"] = out.loc[bad, "author_playtime_forever"]
    log.append(("author_playtime_at_review", "greater than total playtime -> capped", int(bad.sum())))

    # timestamps must lie between Steam launch and the end of data collection
    created = out["timestamp_created"]
    to_nan("timestamp_created",
           (created < C.STEAM_LAUNCH_UNIX) | (created > C.SNAPSHOT_END_UNIX),
           "outside [Steam launch 2003, data collection Jan 2021]")
    created = out["timestamp_created"]
    upd = out["timestamp_updated"]
    bad_upd = upd.notna() & created.notna() & ((upd < created) | (upd > C.SNAPSHOT_END_UNIX))
    out.loc[bad_upd, "timestamp_updated"] = created[bad_upd]
    log.append(("timestamp_updated", "before creation or in the future -> set to creation time",
                int(bad_upd.sum())))
    last = out["author_last_played"]
    to_nan("author_last_played", last <= 0, "0 = 'never / unknown'")
    to_nan("author_last_played", out["author_last_played"] > C.SNAPSHOT_END_UNIX,
           "after the end of data collection")

    report = pd.DataFrame(log, columns=["column", "rule", "values_fixed"])
    return out, report


def clean_text(series: pd.Series) -> pd.Series:
    """Light, label-independent normalisation of review text.

    lower-case -> remove URLs -> remove Steam BBCode tags ([h1], [b], [spoiler]...)
    -> collapse whitespace. Punctuation and stop words are kept on purpose:
    tokenisation, stop words and n-grams are decided in the TF-IDF step (Part 2).
    """
    s = series.fillna("").astype(str).str.lower()
    s = s.str.replace(_URL_RE, " ", regex=True)
    s = s.str.replace(_BBCODE_RE, " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


# ====================================================================== analysis
def iqr_outlier_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Tukey's rule (Q1 - 1.5*IQR, Q3 + 1.5*IQR) on the raw and on the log1p scale."""
    rows = []
    for col in cols:
        x = df[col].dropna().astype("float64")
        if x.empty:
            continue
        rec = {"feature": col, "non_missing": len(x), "median": x.median(),
               "q99": x.quantile(0.99), "max": x.max(), "skew_raw": x.skew()}
        for name, v in (("raw", x), ("log1p", np.log1p(x))):
            q1, q3 = v.quantile([0.25, 0.75])
            iqr = q3 - q1
            n_out = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum())
            rec[f"outliers_{name}_%"] = round(100 * n_out / len(v), 2)
        rec["skew_log1p"] = np.log1p(x).skew()
        rows.append(rec)
    return pd.DataFrame(rows).set_index("feature").round(2)


# ====================================================================== modelling prep
def stratified_split(df: pd.DataFrame, target: str = C.TARGET,
                     val_size: float = 0.15, test_size: float = 0.15,
                     seed: int = C.RANDOM_STATE):
    """70 / 15 / 15 split that keeps the class ratio identical in every part."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val[target], random_state=seed)
    return train, val, test


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Winsorise each column to the [lower, upper] quantiles learned on TRAIN only.

    NaN values pass through unchanged (they are imputed in a later step).
    """

    def __init__(self, lower: float = 0.001, upper: float = 0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        arr = np.asarray(X, dtype="float64")
        self.n_features_in_ = arr.shape[1]
        self.lower_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_", "upper_"])
        arr = np.asarray(X, dtype="float64")
        return np.clip(arr, self.lower_, self.upper_)


def build_tabular_preprocessor(min_game_frequency: int = 20) -> ColumnTransformer:
    """Encoding + scaling for the tabular features known at review time.

    numeric : clip 0.1/99.9 % -> log1p -> median imputation -> standardise
    missing : one 0/1 flag "value was missing" per numeric feature
    boolean : most-frequent imputation (values are already 0/1)
    category: one-hot for app_name; games with < `min_game_frequency` train
              reviews are merged into one 'infrequent' column
    In Part 2 this object is combined with a TfidfVectorizer on `review_clean`.
    """
    numeric = Pipeline([
        ("clip", QuantileClipper(0.001, 0.999)),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    boolean = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist",
                                min_frequency=min_game_frequency,
                                sparse_output=True)
    return ColumnTransformer(
        [("num", numeric, C.NUM_FEATURES),
         ("missing", MissingIndicator(features="all"), C.NUM_FEATURES),
         ("bool", boolean, C.BOOL_FEATURES),
         ("cat", categorical, C.CAT_FEATURES)],
        remainder="drop",
        sparse_threshold=1.0,
    )
