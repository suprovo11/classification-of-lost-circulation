"""
Improved lost-circulation severity classification on the Azadegan dataset.
Re-implementation of Mardanirad, Wood & Zakeri (2021) with efficiency-focused models:
  A. HistGradientBoosting (LightGBM-style GBDT) with balanced class weights
  B. Same GBDT + depth-window (sequence) features  -- TCN-inspired receptive field
  C. Compact MLP neural network (paper-comparable neural baseline)
  D. Replica-style MLP without class weighting (proxy for the paper's setup)
----------------------------------------------------------------------------------------------------------------------------------------------------------------
Created By: Salman Shakib Suprova
"""
import time, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight

RNG = 42
df = pd.read_csv("classify subsurface drilling lost circulation severity in large oil field dataset.csv")

# ---------- 1. Preprocessing ----------
label_col = "LossesSeverity"
order = ["No-Loss", "Seepage Loss", "Partial Loss", "Severe Loss", "Complete Loss"]
short = ["No loss", "Seepage", "Partial", "Severe", "Complete"]
y = df[label_col].map({c: i for i, c in enumerate(order)}).values
X_raw = df.drop(columns=[label_col]).copy()

# Derive well segments from large negative depth resets (paper: 20 wells)
depth = df["M.Depth"].values
well_id = np.zeros(len(df), dtype=int)
well_id[1:] = (np.diff(depth) < -50).cumsum()
n_wells = well_id.max() + 1
print(f"Derived well segments: {n_wells}")

# Physics-guided anomaly filter (paper removed 1,240 rows for pump/flow anomalies).
# Flag records where pumps are on but flow-out is zero (sensor failure) outside loss events, etc.
anom = ((X_raw["FlowIn"] > 50) & (X_raw["FlowOut"] <= 0) & (y == 0)) | \
       ((X_raw["PumpStroke"] > 20) & (X_raw["StandpipePressure"] <= 20))
print(f"Anomalous rows flagged: {anom.sum()}")
keep = ~anom.values if hasattr(anom, 'values') else ~anom
X_raw, y, well_id = X_raw[keep].reset_index(drop=True), y[keep], well_id[keep]

# ---------- 2. Sequence (depth-window) features: TCN-inspired ----------
DYN = ["RateofPenetration", "WeightonBit", "Rotation", "Torque",
       "StandpipePressure", "FlowIn", "FlowOut", "PumpStroke"]
def add_window_features(X, wells, window=10):
    X = X.copy()
    g = X.groupby(wells)
    for c in DYN:
        X[f"{c}_rmean{window}"] = g[c].transform(lambda s: s.rolling(window, min_periods=1).mean())
        X[f"{c}_rstd{window}"]  = g[c].transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0))
        X[f"{c}_delta"]         = g[c].transform(lambda s: s.diff().fillna(0))
    # flow balance: the physically decisive signal for losses
    X["FlowBalance"] = X["FlowOut"] - (X["FlowIn"] / X["FlowIn"].max() * 100)
    X["FlowBalance_rmean"] = X.groupby(wells)["FlowBalance"].transform(
        lambda s: s.rolling(window, min_periods=1).mean())
    return X

X_seq = add_window_features(X_raw, well_id, window=10)
print(f"Raw features: {X_raw.shape[1]} | With window features: {X_seq.shape[1]}")

# ---------- 3. Split (paper-replicating stratified 75/25) ----------
idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.25, random_state=RNG, stratify=y)

results, cms, timings = {}, {}, {}

def evaluate(name, model, Xtr, Xte, sample_weight=None):
    t0 = time.time()
    if sample_weight is not None:
        model.fit(Xtr, y[tr], sample_weight=sample_weight)
    else:
        model.fit(Xtr, y[tr])
    t_train = time.time() - t0
    t0 = time.time(); pred = model.predict(Xte); t_inf = time.time() - t0
    acc = accuracy_score(y[te], pred)
    mf1 = f1_score(y[te], pred, average="macro")
    wf1 = f1_score(y[te], pred, average="weighted")
    rep = classification_report(y[te], pred, target_names=short, output_dict=True, zero_division=0)
    cm = confusion_matrix(y[te], pred)
    results[name] = dict(accuracy=acc, macro_f1=mf1, weighted_f1=wf1, report=rep,
                         train_s=t_train, infer_s=t_inf)
    cms[name] = cm.tolist()
    print(f"\n=== {name} ===  train {t_train:.1f}s | infer {t_inf:.3f}s")
    print(f"Accuracy {acc:.4f} | Macro-F1 {mf1:.4f} | Weighted-F1 {wf1:.4f}")
    print(classification_report(y[te], pred, target_names=short, zero_division=0))
    return model

sw = compute_sample_weight("balanced", y[tr])

# Model A: GBDT on raw 17 features, class-balanced
A = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                   early_stopping=True, random_state=RNG)
evaluate("GBDT (raw features, balanced)", A, X_raw.iloc[tr], X_raw.iloc[te], sample_weight=sw)

# Model B: GBDT + depth-window features (sequence-aware)
B = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                   early_stopping=True, random_state=RNG)
evaluate("GBDT + depth-window (seq-aware)", B, X_seq.iloc[tr], X_seq.iloc[te], sample_weight=sw)

# Model C: compact MLP with balanced-ish handling via scaled data + early stopping
scaler = MinMaxScaler(feature_range=(-1, 1))
Xtr_s, Xte_s = scaler.fit_transform(X_seq.iloc[tr]), scaler.transform(X_seq.iloc[te])
C = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
                  early_stopping=True, n_iter_no_change=10, max_iter=300, random_state=RNG)
evaluate("Compact MLP (seq features)", C, Xtr_s, Xte_s)

# Model D: paper-style replica — plain net on raw normalized features, no imbalance handling
scaler2 = MinMaxScaler(feature_range=(-1, 1))
Xtr_r, Xte_r = scaler2.fit_transform(X_raw.iloc[tr]), scaler2.transform(X_raw.iloc[te])
D = MLPClassifier(hidden_layer_sizes=(100,), activation="relu",
                  early_stopping=True, n_iter_no_change=10, max_iter=300, random_state=RNG)
evaluate("Paper-style NN replica (raw)", D, Xtr_r, Xte_r)

# ---------- 4. Robustness: grouped-by-well split for the best model ----------
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RNG)
gtr, gte = next(gss.split(X_seq, y, groups=well_id))
Bg = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                    early_stopping=True, random_state=RNG)
swg = compute_sample_weight("balanced", y[gtr])
t0 = time.time(); Bg.fit(X_seq.iloc[gtr], y[gtr], sample_weight=swg); tg = time.time()-t0
pg = Bg.predict(X_seq.iloc[gte])
gacc, gmf1 = accuracy_score(y[gte], pg), f1_score(y[gte], pg, average="macro")
print(f"\n=== GBDT+window, WELL-GROUPED split ===")
print(f"Accuracy {gacc:.4f} | Macro-F1 {gmf1:.4f} | train {tg:.1f}s")
print(classification_report(y[gte], pg, target_names=short, zero_division=0))
results["GBDT + window (well-grouped split)"] = dict(
    accuracy=gacc, macro_f1=gmf1,
    weighted_f1=f1_score(y[gte], pg, average="weighted"),
    report=classification_report(y[gte], pg, target_names=short, output_dict=True, zero_division=0),
    train_s=tg, infer_s=None)
cms["GBDT + window (well-grouped split)"] = confusion_matrix(y[gte], pg).tolist()

# Feature importances (permutation on a sample would be slow; use built-in via a tree proxy)
import os
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
with open(output_path, "w") as f:
    json.dump(dict(results=results, cms=cms, classes=short,
                   n_records=int(len(y)), n_wells=int(n_wells),
                   class_counts={short[i]: int((y==i).sum()) for i in range(5)}), f, indent=1)
print(f"\nSaved results.json to {output_path}")