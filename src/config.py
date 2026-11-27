"""Project-wide settings: paths, constants and feature groups.

Every notebook imports from here, so a value is changed in one place only.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"

RAW_CSV = DATA_RAW / "steam_reviews.csv"
SAMPLE_RAW_PARQUET = DATA_INTERIM / "steam_sample_raw.parquet"
FULL_STATS_JSON = DATA_INTERIM / "full_dataset_stats.json"
GAME_STATS_CSV = DATA_INTERIM / "full_dataset_game_stats.csv"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor_tabular.joblib"

DATASET_URL = "https://www.kaggle.com/datasets/najzeko/steam-reviews-2021"

# ---------------------------------------------------------------- sampling
RANDOM_STATE = 42
LANGUAGE = "english"
# Share of English reviews kept. About 9-10 million English reviews x 0.02
# gives roughly 190-200 thousand rows: far above the 40,000 required,
# and small enough for TF-IDF and SVM on an ordinary laptop.
SAMPLE_FRAC = 0.02
CHUNKSIZE = 250_000  # rows read from the 8 GB CSV at a time

# ---------------------------------------------------------------- domain rules
# Steam stores some counters as unsigned integers. Values such as
# 4294967295 (2**32 - 1) or 4398046511104 (2**42) are overflow / "unknown"
# markers, not real counts.
OVERFLOW_THRESHOLD = 2**31
MAX_PLAUSIBLE_GAMES_OWNED = 50_000   # whole Steam store had ~50k games in 2021
MAX_PLAUSIBLE_NUM_REVIEWS = 50_000
MAX_PLAYTIME_TWO_WEEKS_MIN = 14 * 24 * 60  # 20,160 minutes = 14 full days
STEAM_REVIEW_MAX_CHARS = 8_000       # Steam does not accept longer reviews

# ---------------------------------------------------------------- column groups
TARGET = "recommended"
TEXT_RAW = "review"
TEXT_CLEAN = "review_clean"

ID_COLS = ["review_id", "author_steamid", "app_id"]

BOOL_COLS = [
    "recommended",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
]

COUNT_COLS = [
    "votes_helpful",
    "votes_funny",
    "comment_count",
    "author_num_games_owned",
    "author_num_reviews",
    "author_playtime_forever",
    "author_playtime_last_two_weeks",
    "author_playtime_at_review",
]

TIMESTAMP_COLS = [
    "timestamp_created",
    "timestamp_updated",
    "author_last_played",
]

# Features that exist at the moment a review is written.
# These are the only tabular inputs the model is allowed to use.
NUM_FEATURES = [
    "author_num_games_owned",
    "author_num_reviews",
    "author_playtime_at_review",
]
BOOL_FEATURES = [
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
]
CAT_FEATURES = ["app_name"]

# Information collected only AFTER the review was published
# (votes, comments, later playtime, edits). Kept for EDA, excluded from
# the model to avoid look-ahead leakage.
POST_REVIEW_COLS = [
    "votes_helpful",
    "votes_funny",
    "weighted_vote_score",
    "comment_count",
    "author_playtime_forever",
    "author_playtime_last_two_weeks",
    "author_last_played",
    "timestamp_updated",
]

# ---------------------------------------------------------------- time window
# Steam launched on 12 Sep 2003; the dataset was collected in late January 2021.
STEAM_LAUNCH_UNIX = 1_063_324_800        # 2003-09-12 00:00 UTC
SNAPSHOT_END_UNIX = 1_611_619_200        # 2021-01-26 00:00 UTC (safe upper bound)
