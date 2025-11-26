# age_externalizing_binning.py
# Full script: reads a TSV with columns 'age' and 'externalizing',
# finds the binning (method + number of bins) that minimizes cross-validated MSE,
# builds a piecewise-constant mapping: age -> mean externalizing of its bin,
# and reports correlations + final mapping table.
#
# No argparse. Set FILE_PATH below.

import math
import numpy as np
import pandas as pd

# Optional progress bar; safe fallback if tqdm isn't installed.
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

FILE_PATH = r"full_meta_data.csv"  # <-- change this to your TSV path
# ---- binning controls (bigger buckets) ----
TARGET_MIN_SAMPLES_PER_BIN = 30  # raise to 40–60 for even bigger bins
MAX_BINS_CAP = 12                # hard cap on number of bins
FORCE_METHOD = "quantile"        # use quantile bins (robust, no empty bins)

# ---------------------------- Utilities ---------------------------- #

def _find_columns(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    print(cols_lower)
    # Prefer exact names if present
    if "age" in cols_lower:
        age_col = cols_lower["age"]
    else:
        # fallback: first column with 'age' substring
        matches = [c for c in df.columns if "age" in c.lower()]
        if not matches:
            raise ValueError("Could not find an 'age' column.")
        age_col = matches[0]

    # externalizing variants
    cand = [c for c in df.columns if "externalizing" in c.lower()]
    if not cand:
        raise ValueError("Could not find an 'externalizing' column (looking for name containing 'external').")
    ext_col =cols_lower[ "externalizing"]
    return age_col, ext_col


def _clean(df: pd.DataFrame, age_col: str, ext_col: str) -> pd.DataFrame:
    out = df.copy()
    out[age_col] = pd.to_numeric(out[age_col], errors="coerce")
    out[ext_col] = pd.to_numeric(out[ext_col], errors="coerce")
    out = out.dropna(subset=[age_col, ext_col])
    if out.shape[0] < 10:
        raise ValueError("Not enough valid rows after cleaning; need at least 10.")
    return out


def _kfold_indices(n: int, k: int = 5, seed: int = 42):
    k = min(k, n) if n > 1 else 1
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    splits = np.array_split(idx, k)
    for i in range(k):
        val_idx = splits[i]
        train_idx = np.concatenate([splits[j] for j in range(k) if j != i]) if k > 1 else splits[i]
        yield train_idx, val_idx


def _edges_equal_width(values: np.ndarray, k: int) -> np.ndarray:
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmin == vmax:
        return np.array([vmin, vmax])
    # unique to avoid zero-width bins due to float quirks
    edges = np.linspace(vmin, vmax, k + 1)
    return np.unique(edges)


def _edges_quantile(values: np.ndarray, k: int) -> np.ndarray:
    # Use pandas qcut to get robust quantile edges (handles ties); drop duplicate edges.
    try:
        _, edges = pd.qcut(values, q=k, retbins=True, duplicates="drop")
        edges = np.array(edges, dtype=float)
    except Exception:
        # fallback via numpy (may produce duplicates when many ties)
        qs = np.linspace(0, 1, k + 1)
        edges = np.quantile(values, qs)
        edges = np.unique(edges)
    return edges


def _compute_bin_means(train_age: pd.Series, train_y: pd.Series, edges: np.ndarray) -> pd.Series:
    if len(edges) < 2:
        # degenerate: return global mean in a single interval
        mean_val = float(np.mean(train_y))
        return pd.Series([mean_val], index=pd.IntervalIndex.from_breaks(edges, closed="right", verify_integrity=False))

    intervals = pd.IntervalIndex.from_breaks(edges, closed="right")
    bins = pd.cut(train_age, bins=intervals, include_lowest=True)
    means = train_y.groupby(bins).mean()

    # ensure all intervals present
    means = means.reindex(intervals)

    # fill empty-bin means by nearest non-empty neighbor, else global mean
    if means.isna().any():
        global_mean = float(np.mean(train_y))
        # two-sided fill
        means = means.fillna(method="ffill").fillna(method="bfill").fillna(global_mean)

    return means


def _predict_from_edges_means(age_values: pd.Series | np.ndarray, edges: np.ndarray, means: pd.Series) -> np.ndarray:
    intervals = pd.IntervalIndex.from_breaks(edges, closed="right")
    bins = pd.cut(age_values, bins=intervals, include_lowest=True)
    preds = bins.map(means).astype(float).to_numpy()
    # If any NaNs (can happen with extreme float equality issues), clamp to nearest edge:
    if np.isnan(preds).any():
        age_arr = np.asarray(age_values, dtype=float)
        preds = preds.copy()
        left, right = edges[0], edges[-1]
        preds[np.isnan(preds) & (age_arr <= left)] = means.iloc[0]
        preds[np.isnan(preds) & (age_arr >= right)] = means.iloc[-1]
        # remaining nans (very unlikely)
        if np.isnan(preds).any():
            preds = np.where(np.isnan(preds), float(means.mean()), preds)
    return preds


def _cv_score(ages: np.ndarray, ys: np.ndarray, method: str, k_bins: int, n_splits: int = 5) -> float:
    mses = []
    n = len(ages)
    for train_idx, val_idx in _kfold_indices(n, k=n_splits, seed=42):
        a_tr, y_tr = ages[train_idx], ys[train_idx]
        a_va, y_va = ages[val_idx], ys[val_idx]

        if method == "quantile":
            edges = _edges_quantile(a_tr, k_bins)
        elif method == "equal_width":
            edges = _edges_equal_width(a_tr, k_bins)
        else:
            raise ValueError("Unknown method.")

        # skip degenerate when edges collapse to <2
        if len(edges) < 2:
            # predict training mean as constant baseline
            pred = np.full_like(y_va, fill_value=float(np.mean(y_tr)), dtype=float)
            mse = float(np.mean((pred - y_va) ** 2))
            mses.append(mse)
            continue

        means = _compute_bin_means(pd.Series(a_tr), pd.Series(y_tr), edges)
        pred = _predict_from_edges_means(pd.Series(a_va), edges, means)
        mse = float(np.mean((pred - y_va) ** 2))
        mses.append(mse)

    return float(np.mean(mses))


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    # Spearman = Pearson of ranks
    x_rank = pd.Series(x).rank(method="average").to_numpy()
    y_rank = pd.Series(y).rank(method="average").to_numpy()
    return _pearsonr(x_rank, y_rank)


# ---------------------------- Main Logic ---------------------------- #

def select_optimal_binning(ages: np.ndarray, ys: np.ndarray):
    n = len(ages)
    # Fewer bins by design: ensure at least TARGET_MIN_SAMPLES_PER_BIN per bin,
    # and never exceed MAX_BINS_CAP
    max_bins = max(2, min(MAX_BINS_CAP, n // max(1, TARGET_MIN_SAMPLES_PER_BIN)))

    candidates = []
    methods = (FORCE_METHOD,) if FORCE_METHOD in ("quantile", "equal_width") else ("quantile", "equal_width")
    # search a small range of k (larger buckets)
    k_min = 3 if max_bins >= 3 else 2
    for method in methods:
        for k_bins in range(k_min, max_bins + 1):
            candidates.append((method, k_bins))

    scores = []
    n_splits = min(5, max(2, n // 20))
    for method, k_bins in tqdm(candidates, desc="Evaluating binning", leave=False):
        score = _cv_score(ages, ys, method=method, k_bins=k_bins, n_splits=n_splits)
        scores.append((score, method, k_bins))

    scores.sort(key=lambda t: t[0])
    best_score, best_method, best_k = scores[0]
    return best_method, best_k, best_score, scores



def fit_final_mapping(ages: np.ndarray, ys: np.ndarray, method: str, k_bins: int):
    if method == "quantile":
        edges = _edges_quantile(ages, k_bins)
    else:
        edges = _edges_equal_width(ages, k_bins)

    if len(edges) < 2:
        # fallback single-bin
        edges = np.array([float(np.min(ages)), float(np.max(ages))])
    means = _compute_bin_means(pd.Series(ages), pd.Series(ys), edges)
    return edges, means


def print_mapping(edges: np.ndarray, means: pd.Series):
    intervals = pd.IntervalIndex.from_breaks(edges, closed="right")
    df_map = pd.DataFrame({
        "bin": [f"({iv.left:.6g}, {iv.right:.6g}]" for iv in intervals],
        "mean_externalizing": means.to_numpy(),
        "count": pd.Series(pd.cut(pd.Series([iv.mid for iv in intervals]), bins=intervals, include_lowest=True)).value_counts().reindex(intervals, fill_value=0).to_numpy()
    })
    # Note: the count above is a placeholder (one per bin) purely to keep shape; will replace below with actual counts.
    # Recompute true counts using edges and original ages for clarity.
    print("\nFinal piecewise-constant mapping (age range -> mean externalizing):")
    print(df_map[["bin", "mean_externalizing"]].to_string(index=False))


class AgeToExternalizingModel:
    def __init__(self, edges: np.ndarray, means: pd.Series):
        self.edges = np.array(edges, dtype=float)
        self.means = means.copy()
        self.intervals = pd.IntervalIndex.from_breaks(self.edges, closed="right")

    def predict(self, age_value):
        """
        Predict externalizing for a single age or an array-like of ages.
        Returns a float or numpy array of floats.
        """
        s = pd.Series([age_value]) if np.isscalar(age_value) else pd.Series(age_value)
        bins = pd.cut(s, bins=self.intervals, include_lowest=True)
        preds = bins.map(self.means).astype(float).to_numpy()

        # clamp out-of-range values to nearest bin mean
        if np.isnan(preds).any():
            arr = s.to_numpy(dtype=float)
            preds = preds.copy()
            preds[np.isnan(preds) & (arr <= self.edges[0])] = float(self.means.iloc[0])
            preds[np.isnan(preds) & (arr >= self.edges[-1])] = float(self.means.iloc[-1])
            preds = np.where(np.isnan(preds), float(self.means.mean()), preds)

        return float(preds[0]) if np.isscalar(age_value) else preds


def main():
    # 1) Load
    df = pd.read_csv(FILE_PATH)
    #age_col, ext_col = _find_columns(df)
    age_col = "age"
    ext_col = "externalizing"
    df = _clean(df, age_col, ext_col)
    ages = df[age_col].to_numpy(dtype=float)
    ys = df[ext_col].to_numpy(dtype=float)
    n = len(df)

    # 2) Basic stats + correlations
    pear = _pearsonr(ages, ys)
    spear = _spearmanr(ages, ys)

    # 3) Search optimal binning (method + number of bins) by CV MSE
    best_method, best_k, cv_mse, scored = select_optimal_binning(ages, ys)

    # 4) Fit final mapping on all data with selected config
    edges, means = fit_final_mapping(ages, ys, best_method, best_k)
    model = AgeToExternalizingModel(edges, means)

    # 5) In-sample RMSE using final mapping
    preds_in = model.predict(ages)
    rmse_in = math.sqrt(float(np.mean((preds_in - ys) ** 2)))

    # 6) Report
    print("=== Age ↦ Externalizing (Binned Nonparametric Regression) ===")
    print(f"Rows used: {n}")
    print(f"Correlation (Pearson):  {pear:.6f}")
    print(f"Correlation (Spearman): {spear:.6f}")
    print("\nSelected binning (by cross-validated MSE):")
    print(f"  Method: {best_method}")
    print(f"  Bins:   {best_k}")
    print(f"  CV MSE: {cv_mse:.6f}  (CV RMSE: {math.sqrt(cv_mse):.6f})")
    print(f"\nIn-sample RMSE with final mapping: {rmse_in:.6f}")


if __name__ == "__main__":
    main()
