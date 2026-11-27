# data/

Nothing in this folder is stored on GitHub (see `.gitignore`); everything is downloaded or generated locally.

| Folder | Content | Created by |
|---|---|---|
| `raw/` | `steam_reviews.csv` — the original Kaggle file (~8 GB) | you, see the main `README.md` |
| `interim/` | `steam_sample_raw.parquet` (2% random sample of English reviews), `full_dataset_stats.json`, `full_dataset_game_stats.csv` (statistics of the full file) | `notebooks/01_problem_dataset_preprocessing.ipynb`, section 2 |
| `processed/` | `train.parquet`, `val.parquet`, `test.parquet` — cleaned modelling data | same notebook, section 3.12 |
| `processed/` | `train_features.parquet`, `val_features.parquet`, `test_features.parquet` — the same rows plus the 11 engineered features | `notebooks/03_feature_engineering.ipynb`, section 5.5 |

To force a new scan of the raw file (for example after changing `SAMPLE_FRAC` in `src/config.py`), delete `data/interim/`.
