import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

EFFECT_THRESHOLDS = {
    "r": (0.1, 0.3, 0.5),
    "v": (0.1, 0.3, 0.5),
    "eta2": (0.01, 0.06, 0.14),
    "rho": (0.1, 0.3, 0.5),
}


def fmt_p(p):
    if p is None or np.isnan(p):
        return "n/a"
    if p < 1e-300:
        return "< 1e-300"
    return f"{p:.2e}"


def effect_label(value, kind):
    if value is None or pd.isna(value):
        return "n/a"
    small, medium, large = EFFECT_THRESHOLDS[kind]
    a = abs(value)
    if a < small:
        return "negligible"
    if a < medium:
        return "small"
    if a < large:
        return "medium"
    return "large"


def wilson_ci(successes, n, z=1.96):
    successes = np.asarray(successes, dtype="float64")
    n = np.asarray(n, dtype="float64")
    safe_n = np.where(n > 0, n, 1.0)
    p = successes / safe_n
    denom = 1 + z**2 / safe_n
    centre = (p + z**2 / (2 * safe_n)) / denom
    half = z * np.sqrt(p * (1 - p) / safe_n + z**2 / (4 * safe_n**2)) / denom
    return centre - half, centre + half


def rate_table(df, by, flag="is_negative"):
    g = df.groupby(by, observed=True)[flag].agg(["sum", "count"])
    g = g.rename(columns={"sum": "negatives", "count": "reviews"})
    g["rate"] = g["negatives"] / g["reviews"]
    g["ci_low"], g["ci_high"] = wilson_ci(g["negatives"], g["reviews"])
    return g


def cramers_v(x, y):
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan, 0, np.nan
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return chi2, p, dof, v


def mann_whitney(a, b):
    a = pd.Series(a).dropna().to_numpy()
    b = pd.Series(b).dropna().to_numpy()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    cles = u / (len(a) * len(b))
    return {"U": u, "p": p, "cles": cles, "r_rb": 2 * cles - 1, "n_a": len(a), "n_b": len(b)}


def eta_squared(groups):
    groups = [np.asarray(g, dtype="float64") for g in groups]
    all_values = np.concatenate(groups)
    grand = all_values.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((all_values - grand) ** 2).sum()
    return ss_between / ss_total if ss_total > 0 else np.nan


def bonferroni(p_values):
    p = np.asarray(p_values, dtype="float64")
    return np.minimum(p * len(p), 1.0)


def strong_pairs(corr, threshold):
    cols = corr.columns
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append((cols[i], cols[j], value))
    out = pd.DataFrame(rows, columns=["feature_a", "feature_b", "rho"])
    return out.reindex(out["rho"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def log_odds_words(texts, is_negative, min_df=20, token_pattern=r"(?u)\b[a-z][a-z']+\b"):
    vectorizer = CountVectorizer(token_pattern=token_pattern, min_df=min_df)
    counts = vectorizer.fit_transform(texts)
    mask = np.asarray(is_negative, dtype=bool)
    y_neg = np.asarray(counts[mask].sum(axis=0)).ravel().astype("float64")
    y_pos = np.asarray(counts[~mask].sum(axis=0)).ravel().astype("float64")
    prior = y_neg + y_pos
    a0 = prior.sum()
    n_neg, n_pos = y_neg.sum(), y_pos.sum()
    delta = (np.log((y_neg + prior) / (n_neg + a0 - y_neg - prior))
             - np.log((y_pos + prior) / (n_pos + a0 - y_pos - prior)))
    z = delta / np.sqrt(1 / (y_neg + prior) + 1 / (y_pos + prior))
    return (pd.DataFrame({"word": vectorizer.get_feature_names_out(), "z": z,
                          "count_negative": y_neg.astype(int), "count_positive": y_pos.astype(int)})
            .sort_values("z", ascending=False)
            .reset_index(drop=True))
