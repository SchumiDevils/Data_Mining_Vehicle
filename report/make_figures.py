"""Larger-label versions of six figures for the printed report (same data and results as the notebooks, only bigger text).

Light: single fits of the Module 3 configurations and one Apriori run. Writes figures/report_*.pdf|png.
Run with one thread, e.g.  OMP_NUM_THREADS=1 python report/make_figures.py
"""
import json
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
import plot_style as ps
from m3_utils import PriorCorrectedKNN
from m5_utils import Prep

from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier

ps.setup()
CLASSES = ps.CLASS_ORDER
seq = LinearSegmentedColormap.from_list("seq_blue", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
df = pd.read_csv(ROOT / "data/processed/clean_base.csv", index_col="sp_id")
fs = json.load(open(ROOT / "data/processed/feature_sets.json"))
NUM, CAT, TARGET = fs["numeric_model"], fs["categorical_model"], fs["target"]
T = lambda n, **k: pd.read_csv(ps.TAB_DIR / f"{n}.csv", **k)

# ---- 1. Spearman heatmap
num = [c for c in df.select_dtypes("number").columns]
sp = df[num].corr("spearman")
short = [c.replace("_median", "").replace("_fuel_economy", "_mpg").replace("engine_", "").replace("_volume", "").replace("savings_amount", "savings").replace("listing_span_days", "listing_span") for c in num]
fig, ax = plt.subplots(figsize=(ps.FULL_W, 5.9))
im = ax.imshow(np.ma.masked_where(np.triu(np.ones_like(sp, bool)), sp.values), cmap=ps.DIVERGING, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(num))); ax.set_xticklabels(short, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(num))); ax.set_yticklabels(short, fontsize=8)
for i in range(len(num)):
    for j in range(i):
        v = sp.values[i, j]
        if abs(v) >= 0.5:
            ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=6.4, color="white" if abs(v) >= 0.62 else ps.INK)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02); cb.set_label("Spearman ρ"); cb.outline.set_visible(False)
fig.tight_layout(); ps.save(fig, "report_spearman")

# ---- 2. categorical attributes
fig, axes = plt.subplots(1, 4, figsize=(ps.FULL_W, 2.1))
for ax, c in zip(axes, ["dominant_body_type", "dominant_listing_color", "dominant_transmission", "new_used_profile"]):
    vc = df[c].value_counts(dropna=False).sort_values()
    lab = ["NaN" if pd.isna(i) else str(i).replace("coupe_convertible", "coupe/conv.").replace("suv_crossover", "SUV").replace("pickup_truck", "pickup").replace("hatchback_wagon", "hatch/wagon").replace("van_minivan", "van/minivan") for i in vc.index]
    pct = vc.values / len(df) * 100
    ax.barh(lab, pct, color=ps.CATEGORICAL[0], height=0.6)
    for y, (p, k) in enumerate(zip(pct, vc.values)):
        ax.text(p + pct.max() * 0.03, y, f"{k:,}", va="center", fontsize=6.8, color=ps.INK_2)
    ax.set_xlim(0, pct.max() * 1.5); ax.set_title(c.replace("dominant_", "").replace("_", " ")); ax.set_xlabel("% of sellers"); ax.grid(axis="y", visible=False)
fig.tight_layout(); ps.save(fig, "report_categorical")

# ---- 3-4. confusion matrices and ROC curves of the Module 3 models (refitted with the chosen configurations)
a, b = train_test_split(df.index, test_size=.2, stratify=df[TARGET], random_state=42)
tr, te = df.loc[a], df.loc[b]
ytr, yte = tr[TARGET].to_numpy(), te[TARGET].to_numpy()
pr = lambda **k: Prep(NUM, CAT, fs["log1p_vars"], **k)
cats = [sorted(df[c].dropna().unique()) for c in CAT]
MODELS = {
    "Decision Tree": Pipeline([("p", pr()), ("c", DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, class_weight="balanced", random_state=42))]),
    "KNN": Pipeline([("p", pr(log=True, scaler="standard")), ("c", PriorCorrectedKNN(151, p=1, weights="distance", prior_correction=True))]),
    "GaussianNB": Pipeline([("p", pr(log=True, scaler="standard", drop=tuple(CAT))), ("c", GaussianNB(var_smoothing=.1, priors=[1 / 3] * 3))]),
    "CategoricalNB": Pipeline([("p", ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("b", KBinsDiscretizer(3, encode="ordinal", strategy="quantile"))]), NUM),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OrdinalEncoder(categories=cats))]), CAT)])),
        ("c", CategoricalNB(alpha=.01, min_categories=max(3, max(map(len, cats)))))])}
m3 = T("m3_test_metrics", index_col=0)
fc, ac = plt.subplots(1, 4, figsize=(ps.FULL_W, 1.95))
fr, ar_ = plt.subplots(1, 4, figsize=(ps.FULL_W, 2.1), sharey=True)
Y = label_binarize(yte, classes=CLASSES)
for j, (n, m) in enumerate(MODELS.items()):
    m.fit(tr, ytr)
    p = m.predict(te)
    P = m.predict_proba(te)[:, [list(m.classes_).index(c) for c in CLASSES]]
    f1 = f1_score(yte, p, average="macro")
    assert abs(f1 - m3.loc[n, "macro-F1"]) < 1e-4, n           # identical to the Module 3 results
    cm = confusion_matrix(yte, p, labels=CLASSES); cn = cm / cm.sum(1, keepdims=True)
    ax = ac[j]
    ax.imshow(cn, cmap=seq, vmin=0, vmax=1, aspect="auto")
    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{cm[r, c]}\n{cn[r, c]:.0%}", ha="center", va="center", fontsize=7, color="white" if cn[r, c] > 0.45 else ps.INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES if j == 0 else [])
    ax.set_title(f"{n} ({f1:.3f})"); ax.set_xlabel("predicted"); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax = ar_[j]
    fp, tp = [], []
    for i, c in enumerate(CLASSES):
        u, v, _ = roc_curve(Y[:, i], P[:, i]); fp.append(u); tp.append(v)
        ax.plot(u, v, color=ps.CLASS_COLORS[c], linewidth=1.5, label=f"{c} ({auc(u, v):.2f})")
    u, v, _ = roc_curve(Y.ravel(), P.ravel())
    ax.plot(u, v, color=ps.CATEGORICAL[1], linewidth=1.2, linestyle="--", label=f"micro ({auc(u, v):.2f})")
    gx = np.linspace(0, 1, 201); gm = np.mean([np.interp(gx, x, y) for x, y in zip(fp, tp)], axis=0)
    ax.plot(gx, gm, color=ps.INK, linewidth=1.2, linestyle=":", label=f"macro ({auc(gx, gm):.2f})")
    ax.plot([0, 1], [0, 1], color=ps.GRID, linewidth=1); ax.set_title(n); ax.set_xlabel("FPR")
    ax.legend(frameon=False, fontsize=6.3, loc="lower right", handlelength=1.2, borderaxespad=0.1, labelspacing=0.25)
ac[0].set_ylabel("true"); ar_[0].set_ylabel("TPR")
fc.tight_layout(); ps.save(fc, "report_confusion")
fr.tight_layout(); ps.save(fr, "report_roc")

# ---- 5. association rules: counts, histograms, scatter (one Apriori run at min_sup 0.05 with the Module 4 encoding)
nb = json.load(open(ROOT / "notebooks/04_pattern_mining.ipynb", encoding="utf8"))
code = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "".join(c["source"]).startswith("TERTILE"))
ns = {"np": np, "pd": pd, "TARGET": TARGET}
exec(code.split("enc_all = ")[0], ns)
B = ns["encode_items"](df, ns["fit_encoder"](df))
fi = apriori(B, min_support=0.05, use_colnames=True, low_memory=True)
R = association_rules(fi, num_itemsets=len(B), metric="confidence", min_threshold=0.6, return_metrics=["support", "confidence", "lift"])
sw = T("m4_rule_sweep", index_col=0)
assert len(fi) == 6835 and len(R) == sw.loc[0.6, "total"]
ks = [c for c in sw.columns if c.isdigit()]; H = sw[ks].T.to_numpy()
fig, axes = plt.subplots(1, 4, figsize=(ps.FULL_W, 2.4), gridspec_kw={"width_ratios": [2.3, 1, 1, 1.15]})
ax = axes[0]
ax.imshow(np.maximum(H, 1), cmap=seq, norm=LogNorm(vmin=1, vmax=H.max()), aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H[i, j]; lab = f"{v / 1000:.0f}k" if v >= 10000 else (f"{v / 1000:.1f}k" if v >= 1000 else str(v))
        ax.text(j, i, lab, ha="center", va="center", fontsize=6.6, color="white" if v > 1500 else ps.INK)
ax.set_xticks(range(len(sw))); ax.set_xticklabels([f"{x:g}" for x in sw.index], fontsize=6.8)
ax.set_yticks(range(len(ks))); ax.set_yticklabels(ks); ax.set_xlabel("min_conf"); ax.set_ylabel("rule length k"); ax.set_title("Rules by min_conf and k"); ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
k_fmt = plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")
axes[1].hist(R.confidence, bins=30, color=ps.CATEGORICAL[0]); axes[1].set_xlabel("confidence"); axes[1].set_ylabel("rules"); axes[1].set_title("Confidence"); axes[1].yaxis.set_major_formatter(k_fmt)
axes[2].hist(R.lift, bins=40, color=ps.CATEGORICAL[1]); axes[2].axvline(1, color=ps.INK_2, linewidth=0.8, linestyle="--"); axes[2].set_xlabel("lift"); axes[2].set_title("Lift"); axes[2].yaxis.set_major_formatter(k_fmt)
q = R.sample(6000, random_state=42)
sc = axes[3].scatter(q.support, q.confidence, c=q.lift, s=2, cmap=seq, vmin=1, vmax=4, rasterized=True)
cb = fig.colorbar(sc, ax=axes[3], pad=0.02); cb.set_label("lift", fontsize=7); cb.outline.set_visible(False)
axes[3].set_xlabel("support"); axes[3].set_ylabel("confidence"); axes[3].set_title("Support–confidence")
fig.tight_layout(); ps.save(fig, "report_rules_quant")

# ---- 6. rule-based classifier
cmp_, cv = T("m4_classifier_comparison", index_col=0), T("m4_rule_classifier_cv")
fig, axes = plt.subplots(1, 2, figsize=(ps.FULL_W, 2.2), gridspec_kw={"width_ratios": [1, 1.15]})
o = cmp_["macro-F1"].sort_values().index
axes[0].barh(o, cmp_.loc[o, "macro-F1"], height=0.6, color=[ps.CATEGORICAL[1] if n.startswith("Rule") else (ps.NEUTRAL if n.startswith("Base") else ps.CATEGORICAL[0]) for n in o])
for y, v in enumerate(cmp_.loc[o, "macro-F1"]):
    axes[0].text(v + 0.006, y, f"{v:.3f}", va="center", fontsize=7, color=ps.INK_2)
axes[0].set_xlim(0, 0.54); axes[0].set_xlabel("test macro-F1"); axes[0].grid(axis="y", visible=False)
ax = axes[1]
for i, g in enumerate(["best-antecedent", "top10", "mean + prior", "class-best-conf", "class-best-lift"]):
    d = cv[(cv.aggregation == g) & (cv.min_sup == 0.005)].sort_values("max antecedent")
    ax.plot(d["max antecedent"], d["CV macro-F1"], marker="os^Dv"[i], markersize=4, color=ps.CATEGORICAL[i], linestyle="-" if i < 3 else "--", label=g, linewidth=1.6)
ax.axhline(0.236, color=ps.NEUTRAL, linestyle=":", linewidth=1, label="majority class")
ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("max antecedent length L (s = 0.005)"); ax.set_ylabel("CV macro-F1")
ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.tight_layout(); ps.save(fig, "report_rule_classifier")
print("report figures written")
