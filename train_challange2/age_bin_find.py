# Loads a CSV, cleans age/externalizing columns, searches for the best
# binning scheme via cross-validated MSE, then prints the mapping.

import math
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# File path
FILE_PATH = r"full_meta_data.csv"

# Binning settings
TARGET_MIN_SAMPLES_PER_BIN = 30
MAX_BINS_CAP = 12
FORCE_METHOD = "quantile"   # choose "quantile" or "equal_width"

# ------------------------------------------------------------------ #

def _find_columns(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    print(cols_lower)

    # age column
    if "age" in cols_lower:
        age_col = cols_lower["age"]
    else:
        matches = [c for c in df.columns if "age" in c.lower()]
        if not matches:
            raise ValueError("No 'age' column found.")
        age_col = matches[0]

    # externalizing column
    matches = [c for c in df.columns if "externalizing" in c.lower()]
    if not matches:
        raise ValueError("No 'externalizing' column found.")
    ext_col = cols_lower["externalizing"]

    return age_col, ext_col


def _clean(df: pd.DataFrame, age_col: str, ext_col: str):
    out = df.copy()
    out[age_col] = pd.to_numeric(out[age_col], errors="coerce")
    out[ext_col] = pd.to_numeric(out[ext_col], errors="coerce")
    out = out.dropna(subset=[age_col, ext_col])
    if len(out) < 10:
        raise ValueError("Too few valid rows after cleaning.")
    return out


def _kfold_indices(n, k=5, seed=42):
    k = min(k, n) if n > 1 else 1
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    splits = np.array_split(idx, k)
    for i in range(k):
        val = splits[i]
        train = np.concatenate([splits[j] for j in range(k) if j != i]) if k > 1 else val
        yield train, val


def _edges_equal_width(values, k):
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmin == vmax:
        return np.array([vmin, vmax])
    return np.unique(np.linspace(vmin, vmax, k + 1))


def _edges_quantile(values, k):
    try:
        _, edges = pd.qcut(values, q=k, retbins=True, duplicates="drop")
        return np.array(edges, float)
    except Exception:
        qs = np.linspace(0, 1, k + 1)
        return np.unique(np.quantile(values, qs))


def _compute_bin_means(a, y, edges):
    if len(edges) < 2:
        return pd.Series([float(y.mean())],
                         index=pd.IntervalIndex.from_breaks(edges, closed="right"))

    intervals = pd.IntervalIndex.from_breaks(edges, closed="right")
    binned = pd.cut(a, bins=intervals, include_lowest=True)
    means = y.groupby(binned).mean().reindex(intervals)

    if means.isna().any():
        g = float(y.mean())
        means = means.fillna(method="ffill").fillna(method="bfill").fillna(g)
    return means


def _predict_from_edges_means(a, edges, means):
    intervals = pd.IntervalIndex.from_breaks(edges, closed="right")
    binned = pd.cut(a, bins=intervals, include_lowest=True)
    preds = binned.map(means).astype(float).to_numpy()

    if np.isnan(preds).any():
        arr = np.asarray(a, float)
        left, right = edges[0], edges[-1]
        preds = preds.copy()
        preds[np.isnan(preds) & (arr <= left)] = float(means.iloc[0])
        preds[np.isnan(preds) & (arr >= right)] = float(means.iloc[-1])
        preds = np.where(np.isnan(preds), float(means.mean()), preds)
    return preds


def _cv_score(ages, ys, method, k_bins, n_splits=5):
    out = []
    for tr, va in _kfold_indices(len(ages), k=n_splits, seed=42):
        a_tr, y_tr = ages[tr], ys[tr]
        a_va, y_va = ages[va], ys[va]

        if method == "quantile":
            edges = _edges_quantile(a_tr, k_bins)
        else:
            edges = _edges_equal_width(a_tr, k_bins)

        if len(edges) < 2:
            pred = np.full(len(y_va), y_tr.mean())
        else:
            means = _compute_bin_means(pd.Series(a_tr), pd.Series(y_tr), edges)
            pred = _predict_from_edges_means(pd.Series(a_va), edges, means)

        mse = float(np.mean((pred - y_va) ** 2))
        out.append(mse)
    return float(np.mean(out))


def _pearsonr(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearmanr(x, y):
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    return _pearsonr(xr, yr)


# ------------------------------------------------------------------ #

def select_optimal_binning(ages, ys):
    n = len(ages)
    max_bins = max(2, min(MAX_BINS_CAP, n // max(1, TARGET_MIN_SAMPLES_PER_BIN)))

    methods = (FORCE_METHOD,) if FORCE_METHOD in ("quantile", "equal_width") else ("quantile", "equal_width")
    k_min = 3 if max_bins >= 3 else 2

    candidates = [(m, k) for m in methods for k in range(k_min, max_bins + 1)]
    scores = []

    n_splits = min(5, max(2, n // 20))
    for method, k_bins in tqdm(candidates, desc="Evaluating", leave=False):
        s = _cv_score(ages, ys, method, k_bins, n_splits)
        scores.append((s, method, k_bins))

    scores.sort(key=lambda t: t[0])
    return scores[0][1], scores[0][2], scores[0][0], scores


def fit_final_mapping(ages, ys, method, k_bins):
    edges = _edges_quantile(ages, k_bins) if method == "quantile" else _edges_equal_width(ages, k_bins)
    if len(edges) < 2:
        edges = np.array([float(np.min(ages)), float(np.max(ages))])
    means = _compute_bin_means(pd.Series(ages), pd.Series(ys), edges)
    return edges, means


def print_mapping(edges, means):
    intervals = pd.IntervalIndex.from_breaks(edges, closed="right")
    df_map = pd.DataFrame({
        "bin": [f"({iv.left:.6g}, {iv.right:.6g}]" for iv in intervals],
        "mean_externalizing": means.to_numpy()
    })
    print("\nFinal mapping:")
    print(df_map.to_string(index=False))


class AgeToExternalizingModel:
    def __init__(self, edges, means):
        self.edges = np.array(edges, float)
        self.means = means.copy()
        self.intervals = pd.IntervalIndex.from_breaks(self.edges, closed="right")

    def predict(self, age_value):
        s = pd.Series([age_value]) if np.isscalar(age_value) else pd.Series(age_value)
        binned = pd.cut(s, bins=self.intervals, include_lowest=True)
        preds = binned.map(self.means).astype(float).to_numpy()

        if np.isnan(preds).any():
            arr = s.to_numpy(float)
            preds = preds.copy()
            preds[np.isnan(preds) & (arr <= self.edges[0])] = float(self.means.iloc[0])
            preds[np.isnan(preds) & (arr >= self.edges[-1])] = float(self.means.iloc[-1])
            preds = np.where(np.isnan(preds), float(self.means.mean()), preds)

        return float(preds[0]) if np.isscalar(age_value) else preds


def main():
    df = pd.read_csv(FILE_PATH)
    age_col = "age"
    ext_col = "externalizing"

    df = _clean(df, age_col, ext_col)
    ages = df[age_col].to_numpy(float)
    ys = df[ext_col].to_numpy(float)
    n = len(df)

    pear = _pearsonr(ages, ys)
    spear = _spearmanr(ages, ys)

    best_method, best_k, cv_mse, _ = select_optimal_binning(ages, ys)

    edges, means = fit_final_mapping(ages, ys, best_method, best_k)
    model = AgeToExternalizingModel(edges, means)

    preds = model.predict(ages)
    rmse_in = math.sqrt(np.mean((preds - ys) ** 2))

    print("=== Age → Externalizing ===")
    print(f"Rows: {n}")
    print(f"Pearson:  {pear:.6f}")
    print(f"Spearman: {spear:.6f}")
    print("\nSelected binning:")
    print(f"  Method: {best_method}")
    print(f"  Bins:   {best_k}")
    print(f"  CV MSE: {cv_mse:.6f}")
    print(f"  CV RMSE: {math.sqrt(cv_mse):.6f}")
    print(f"\nIn-sample RMSE: {rmse_in:.6f}")

    print_mapping(edges, means)


if __name__ == "__main__":
    main()
