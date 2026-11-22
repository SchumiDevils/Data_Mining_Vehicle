# executed by make_tables.py (shares its helpers: read, write, num, f3, esc, T)

# ---- ablation: one row per variant, one column per model (delta vs reference, folds better)
a = read("m5_ablation")
ref = a[a.ablation == "reference"].set_index("model")["CV macro-F1"]
ORDER = [("scale", "raw, unscaled", "raw scale, no scaling"), ("scale", "raw, standardized", "raw scale, standardized"),
         ("scale", "log1p, min-max", "log1p, min-max"), ("scale", "log1p, robust", "log1p, robust scaler"),
         ("outliers", "winsorized at p1/p99 (per fold)", "winsorized at $p_1/p_{99}$"),
         ("outliers", "DBSCAN noise removed from training folds", "DBSCAN noise removed from training"),
         ("features removed", "without seller size", r"$-$ seller size"), ("features removed", "without price/turnover/wear", r"$-$ price, turnover, wear (6)"),
         ("features removed", "without vehicle specs", r"$-$ vehicle specifications (8)"), ("features removed", "without vehicle history", r"$-$ vehicle history (4)"),
         ("features removed", "without categoricals", r"$-$ categoricals (5)"), ("features removed", "only seller size (n_listings)", "only seller size"),
         ("features added", "12 engineered features", r"$+$ 12 engineered attributes"), ("features added", "K-Means segment + centroid distances", r"$+$ K-Means segment and distances")]
rows = [["reference (CV macro-F1)"] + [f"{ref[m]:.3f}" for m in ("DT", "KNN", "GNB")]]
for g, v, lab in ORDER:
    r = [lab]
    for m in ("DT", "KNN", "GNB"):
        q = a[(a.model == m) & (a.ablation == g) & (a.variant == v)]
        if q.empty:
            r.append("--")
        else:
            d, fb = q.delta.iloc[0], int(q["folds better"].iloc[0])
            s = ("+" if d >= 0.0005 else ("$-$" if d <= -0.0005 else "")) + f"{abs(d):.3f}"
            r.append(rf"\textbf{{{s}}}" if fb in (0, 5) and abs(d) >= 0.0005 else s)
    rows.append(r)
write("m5_ablation", "@{}lrrr@{}", r"Variant & DT & KNN & GNB", rows, (1, 5, 7, 13))

# ---- nested CV
q = read("m5_nested_cv").set_index(["model", "space"])
SP = ["M3 space (tau 0/1)", "+ tau grid", "+ tau grid + engineered"]
rows = []
for m, lab in [("DT", "Decision Tree"), ("KNN", "KNN"), ("GNB", "GaussianNB"), ("CNB", "CategoricalNB")]:
    cells = []
    for s in SP:
        x = q.loc[(m, s)]
        fb = "" if pd.isna(x["folds better than M3 space"]) else f" ({int(x['folds better than M3 space'])}/5)"
        cells.append(f"{x['mean']:.3f}$\\pm${x['std']:.3f}{fb}".replace("0.", "."))
    rows.append([lab] + cells)
write("m5_nested", "@{}llll@{}", r"Model & baseline space & $+\,\tau$ grid & $+\,\tau$ and engineered", rows)

# ---- test set
t = read("m5_test_metrics", index_col=0)
ROWS = [("DT (M3)", "Decision Tree, baseline"), ("KNN (M3)", "KNN, baseline"), ("GNB (M3)", "GaussianNB, baseline"), ("DT (τ)", r"Decision Tree, extended"), ("KNN (τ)", r"KNN, extended"),
        ("GNB (τ)", r"GaussianNB, extended"), ("CNB (τ)", r"CategoricalNB, extended"), ("Rules (M4)", r"Rules (\cref{sec:exploit})"),
        ("Random forest (ref.)", "Random forest (reference)"), ("Gradient boosting (ref.)", "Gradient boosting (reference)"), ("Majority", "Majority class")]
m4cv = read("m4_rule_classifier_cv")["CV macro-F1"].max()
rows = []
for n, lab in ROWS:
    r = t.loc[n]
    cv = m4cv if n == "Rules (M4)" else r["CV macro-F1"]
    b = lambda s: rf"\textbf{{{s}}}" if n == "KNN (τ)" else s
    rows.append([lab, f3(cv), b(f3(r["macro-F1"])), f"[{r['F1 lo']:.3f}, {r['F1 hi']:.3f}]", f3(r["accuracy"]), f3(r["weighted-F1"]), f3(r["AUC macro"]),
                 num(r["R low"], 2), num(r["R medium"], 2), num(r["R high"], 2), num(r["QWK"], 2), num(r["MAE"], 2)])
write("m5_test", "@{}lrrlrrrrrrrr@{}",
      r"Model & CV \Fm & Test \Fm & 95\% CI & Acc. & F1$_\mathrm{wei}$ & AUC$_\mathrm{mac}$ & R$_\mathrm{low}$ & R$_\mathrm{med}$ & R$_\mathrm{high}$ & QWK & MAE", rows, (3, 7, 8, 10))

# ---- size effect
s = read("m5_size_effect", index_col=0)
rows = [[{"all": "all sellers"}.get(k, k), num(r["n"], 0), num(r["rho(size, rating)"], 2), f"{r['high % small']:.0f} / {r['high % large']:.0f}", f"{r['low % small']:.0f} / {r['low % large']:.0f}"] for k, r in s.iterrows()]
write("m5_size", "@{}lrrrr@{}", r"Group & $n$ & $\rho_s$ & \% \emph{high} (S/L) & \% \emph{low} (S/L)", rows, (1, 5))
x = read("m5_size_by_dealer", header=[0, 1], index_col=0)
lab = {"small": "small", "mid": "mid", "large": "large"}
rows = [[lab[k], num(x.loc[k, ("n", "franchise")], 0), num(x.loc[k, ("n", "independent")], 0)] + [num(x.loc[k, (c, d)], 1) for c in ("high", "low") for d in ("franchise", "independent")] for k in ("small", "mid", "large")]
write("m5_dealer", "@{}lrr|rr|rr@{}", r" & \multicolumn{2}{c|}{Sellers} & \multicolumn{2}{c|}{\% \emph{high}} & \multicolumn{2}{c}{\% \emph{low}} \\ Size tertile & fra. & ind. & fra. & ind. & fra. & ind.", rows)

# ---- binning sensitivity
b = read("m5_binning")
rows = [[r.binning.replace("equal-frequency", "equal freq.").replace("equal-width", "equal width"), num(r["items"], 0), num(r["frequent itemsets"], 0), num(r["rules"], 0), num(r["median lift"], 2), num(r["max lift"], 1),
         num(r["largest numeric-bin support"], 2), f3(r["rule classifier CV macro-F1"])] for _, r in b.iterrows()]
write("m5_binning", "@{}lrrrrrrr@{}", r"Bins & Items & Itemsets & Rules & lift$_\mathrm{med}$ & lift$_\mathrm{max}$ & sup$_\mathrm{max}$ & CV \Fm", rows)

# ---- rule generalization
g = read("m5_rule_generalization")
rows = []
for r in ("low", "medium", "high"):
    q = g[g.rating == r].set_index("support")
    rows.append([rf"\emph{{{r}}}"] + [f"{q.loc[k, 'train_lift']:.2f} $\\to$ {q.loc[k, 'test_lift']:.2f}" for k in ("0.5-1%", "1-2%", "2-5%", ">5%")])
write("m5_gen", "@{}lllll@{}", r"Rating & sup 0.5--1\% & 1--2\% & 2--5\% & $>5\%$", rows)

# ---- regression ablation
r = read("m5_regression_ablation")
rows = []
for p_ in ["1 (mileage)", "4 (Module 3)", "18 (all numeric)"]:
    for m in ("OLS", "DT", "KNN"):
        A, B = (r[(r.predictors == p_) & (r.model == m) & (r.target == k)].iloc[0] for k in ("raw", "log1p"))
        rows.append([p_.split(" ")[0] if m == "OLS" else "", m, num(A["CV R2 ($)"], 3), num(A["median abs. % error"], 1), num(A["negative predictions"], 0), num(B["CV R2 (log)"], 3), num(B["CV R2 ($)"], 3), num(B["median abs. % error"], 1)])
write("m5_reg", "@{}ll|rrr|rrr@{}", r" & & \multicolumn{3}{c|}{raw price} & \multicolumn{3}{c}{log price} \\ Pred. & Model & $R^2$(\$) & APE\% & neg. & $R^2$ & $R^2$(\$) & APE\%", rows, (3, 6))
print("extension tables written")
