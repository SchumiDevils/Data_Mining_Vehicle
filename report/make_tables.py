"""Generate the LaTeX tables of the report from tables/*.csv, so that no number is typed by hand.

Run from anywhere:  python report/make_tables.py     (writes report/tables/*.tex)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "tables"
OUT = ROOT / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
read = lambda n, **k: pd.read_csv(T / f"{n}.csv", **k)


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def num(v, d=2):
    if pd.isna(v):
        return "--"
    s = f"{v:,.{d}f}".replace(",", r"\,")
    return ("$-$" + s[1:]) if s.startswith("-") else s


def f3(v, bold=False):
    s = "--" if pd.isna(v) else (f"{v:.3f}" if v >= 0 else f"$-${-v:.3f}")
    return rf"\textbf{{{s}}}" if bold else s


def write(name, spec, head, rows, rules=()):
    """rows: list of lists of cells; rules: row indices before which a midrule is drawn."""
    body = []
    for i, r in enumerate(rows):
        if i in rules:
            body.append(r"\midrule")
        body.append(" & ".join(r) + r" \\")
    tex = (rf"\begin{{tabular}}{{{spec}}}" + "\n\\toprule\n" + head + " \\\\\n\\midrule\n" + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n")
    (OUT / f"{name}.tex").write_text(tex, encoding="utf8")


# ------------------------------------------------------------------ Module 1
num_tab = read("m1_numeric_summary", index_col=0)
log = read("m1_log_transform", index_col=0)
kw = read("m1_kruskal_target", index_col=0)
G = [("Listing activity", {"n_listings": ("listings", "count"), "n_listing_days": ("days with listings", "days"),
                           "listing_span_days": ("activity span", "days"), "daysonmarket_median": ("days on market", "days")}),
     ("Price and wear", {"price_median": ("advertised price", "USD"), "savings_amount_median": ("adv.\ savings", "USD"),
                         "mileage_median": ("mileage", "mi"), "year_median": ("model year", "year"),
                         "vehicle_age_median": ("age at listing", "years"), "owner_count_median": ("previous owners", "count")}),
     ("Engine, economy", {"horsepower_median": ("horsepower", "hp"), "engine_displacement_median": ("displacement", "cc"),
                          "city_fuel_economy_median": ("city economy", "MPG"), "highway_fuel_economy_median": ("highway economy", "MPG")}),
     ("Dimensions", {"back_legroom_median": ("rear legroom", "in"), "front_legroom_median": ("front legroom", "in"),
                     "fuel_tank_volume_median": ("fuel tank", "gal"), "height_median": ("height", "in"),
                     "length_median": ("length", "in"), "width_median": ("width", "in")}),
     ("Inventory mix", {"pct_isCab": ("share ex-taxi", "[0,1]"), "pct_is_new": ("share new", "[0,1]"),
                        "pct_salvage": ("share salvage", "[0,1]"), "pct_theft_title": ("share theft title", "[0,1]")})]
DROPPED = {"n_listing_days": "no", "year_median": "no", "vehicle_age_median": "no", "engine_displacement_median": "no",
           "highway_fuel_economy_median": "no"}
rows, rules = [], []
for g, vs in G:
    rows.append([rf"\multicolumn{{14}}{{@{{}}l}}{{\textit{{{g}}}}}"])
    for i, (v, (meaning, unit)) in enumerate(vs.items()):
        r = num_tab.loc[v]
        d = 0 if abs(r["max"]) >= 1000 and v != "year_median" else (0 if v == "year_median" else (2 if r["max"] <= 1 else 1))
        fmt = lambda x: (f"{x:.0f}" if v == "year_median" else num(x, d))
        use = "dropped" if v in DROPPED else (r"log1p" if bool(log.log1p.get(v, False)) else "raw")
        rows.append([rf"\texttt{{{esc(v.replace('_median', ''))}}}", meaning, unit,
                     num(100 * r["missing"] / 13843, 2) if r["missing"] else "0", fmt(r["min"]), fmt(r["q25"]), fmt(r["median"]), fmt(r["q75"]), fmt(r["max"]),
                     fmt(r["mean"]), fmt(r["std"]), num(r["skew"], 1), num(kw.epsilon2[v], 3), use])
write("m1_numeric", "@{}llrrrrrrrrrrrl@{}",
      r"Attribute & Meaning & Unit & NaN \% & min & Q1 & median & Q3 & max & mean & std & skew & $\varepsilon^2$ & Use", rows, rules)

pairs = read("m1_top_correlations")
sh = lambda v: esc(v.replace("_median", "").replace("engine_", "").replace("_volume", "").replace("_fuel_economy", "_mpg"))
rows = [[sh(a), sh(b), num(p, 2), num(s, 2)]
        for a, b, p, s in pairs[["var_a", "var_b", "pearson", "spearman"]].itertuples(index=False)]
write("m1_corr", "@{}llrr@{}", r"Attribute A & Attribute B & $r$ & $\rho_s$", rows)

# ------------------------------------------------------------------ Module 2
s = read("m2_feature_set_sensitivity")
w = s.pivot_table(index=["feature set", "scale"], columns="k", values="silhouette")
m = s.pivot_table(index=["feature set", "scale"], columns="k", values="min size")
NAME = {"ALL19 (all modelling features)": "All 19", "PROFILE7 (chosen)": "Profile 7", "PROFILE7 + 3 vehicle specs": "Profile 7 + 3 specs"}
rows = []
for fset in ["PROFILE7 (chosen)", "PROFILE7 + 3 vehicle specs", "ALL19 (all modelling features)"]:
    for sc in ["log1p", "raw"]:
        rows.append([NAME[fset], sc] + [f3(w.loc[(fset, sc), k])[0:5] for k in (2, 3, 4, 5, 6)] + [num(m.loc[(fset, sc), 4], 0)])
write("m2_sens", "@{}llrrrrrr@{}", r"Feature set & Scale & $k{=}2$ & 3 & 4 & 5 & 6 & min$_{4}$", rows, (2, 4))

p = read("m2_kmeans4_profiles", index_col=0)
cols = [("size", "Sellers", 0), ("n_listings", r"Listings", 0), ("price_median", r"Price (\$)", 0), ("savings_amount_median", r"Savings (\$)", 0),
        ("daysonmarket_median", "Days on market", 1), ("mileage_median", "Mileage", 0), ("vehicle_age_median", "Vehicle age", 1),
        ("pct_is_new", "Share new", 2), ("listing_span_days", "Span (days)", 0), ("horsepower_median", "Horsepower", 0), ("franchise %", r"Franchise \%", 1), ("suv %", r"SUV \%", 1),
        ("low %", r"Rated low \%", 1), ("medium %", r"Rated medium \%", 1), ("high %", r"Rated high \%", 1)]
rows = [[lab] + [num(p.loc[k, c], d) for k in range(4)] for c, lab, d in cols]
write("m2_profiles", "@{}lrrrr@{}", r" & C0 & C1 & C2 & C3", rows, (1, 12))

g = read("m2_dbscan_grid")
EPS = sorted(g.eps.unique())
rows = []
for mp in sorted(g.MinPts.unique()):
    q = g[g.MinPts == mp].set_index("eps")
    rows.append([str(mp)] + [("\\textbf{%s}" if (mp == 15 and e == 0.6) else "%s") % f"{int(q.loc[e, 'clusters'])}\\,({int(q.loc[e, 'clusters >= 1%'])})" for e in EPS])
    rows.append([""] + [f"{q.loc[e, 'noise %']:.0f}\\%" for e in EPS])
    rows.append([""] + [num(q.loc[e, "silhouette (non-noise)"], 2) for e in EPS])
write("m2_dbscan", "@{}l" + "r" * len(EPS) + "@{}", "MinPts & " + " & ".join(f"$\\varepsilon$={e:g}" for e in EPS), rows, (3, 6, 9, 12))

t = read("m2_dbscan_noise_traits", index_col=0)
SEL = ["days on market > 120", "listing span > p95", "price < p5", "n_listings <= 10", "price > p95", "mixed seller with mileage > 20k", "n_listings > p95"]
LAB = {"days on market > 120": "days on market $>120$", "listing span > p95": "listing span $>p_{95}$", "price < p5": "price $<p_5$", "n_listings <= 10": r"$\leq 10$ listings",
       "price > p95": "price $>p_{95}$", "mixed seller with mileage > 20k": r"mixed seller, mileage $>$20k", "n_listings > p95": "listings $>p_{95}$"}
write("m2_noise", "@{}lrrr@{}", r"Trait & Noise \% & Core \% & Ratio", [[LAB[k], num(t.loc[k, "noise %"], 1), num(t.loc[k, "core %"], 1), num(t.loc[k, "ratio"], 1)] for k in SEL])

h = read("m2_hierarchical")
rows = []
for (l, me), q in h.groupby(["linkage", "metric"], sort=False):
    q = q.set_index("k")
    rows.append([l.capitalize(), {"cityblock": "Manhattan"}.get(me, me.capitalize()), num(q["cophenetic r"].iloc[0], 2)]
                + [num(q.loc[k, "silhouette"], 2) for k in (2, 3, 4, 5, 6)] + [num(q.loc[4, "min size"], 0), num(q.loc[4, "max size"], 0)])
write("m2_hier", "@{}llrrrrrrrr@{}", r"Linkage & Distance & $r_c$ & $k{=}2$ & 3 & 4 & 5 & 6 & min$_4$ & max$_4$", rows)

c = read("m2_comparison")
rows = []
for r in c.itertuples(index=False):
    r = r._asdict() if hasattr(r, "_asdict") else r
    rows.append(None)
rows = []
for _, r in c.iterrows():
    par = r["parameters"].replace("eps=0.6, MinPts=15", r"$\varepsilon{=}0.6$, MinPts$=$15").replace("/Euclidean", "").replace("k=", "$k$=")
    rows.append([r["method"], par, num(r["clusters"], 0), num(r["noise %"], 0), num(r["silhouette"], 3), num(r["Davies-Bouldin"], 2), num(r["SSE"], 0),
                 num(r["min size %"], 1), num(r["max size %"], 1), num(r["ARI vs K-Means4"], 2), num(r["ARI rating"], 3), num(r["ARI franchise"], 2), num(r["ARI new_used_profile"], 2)])
write("m2_comparison", "@{}llrrrrrrrrrrr@{}",
      r"Method & Parameters & Clusters & Noise \% & Silh. & DB & SSE & min \% & max \% & ARI$_\mathrm{KM4}$ & ARI$_\mathrm{rating}$ & ARI$_\mathrm{franchise}$ & ARI$_\mathrm{new/used}$", rows)

# ------------------------------------------------------------------ Module 3
q = read("m3_test_metrics", index_col=0)
rows = []
for n, lab in [("Decision Tree", "Decision Tree"), ("KNN", "KNN"), ("GaussianNB", "GaussianNB"), ("CategoricalNB", "CategoricalNB"), ("Baseline: majority", "Majority class")]:
    r = q.loc[n]
    rows.append([lab] + [num(r[k], 3) for k in ("accuracy", "macro-F1", "weighted-F1", "AUC macro (OvR)", "AUC micro")]
                + [num(r[f"{a} {cl}"], 2) for cl in ("low", "medium", "high") for a in ("P", "R", "F1")])
write("m3_test", "@{}lrrrrr|rrr|rrr|rrr@{}",
      r" & & & & & & \multicolumn{3}{c|}{\textit{low}} & \multicolumn{3}{c|}{\textit{medium}} & \multicolumn{3}{c}{\textit{high}} \\ Model & Acc. & F1$_\mathrm{mac}$ & F1$_\mathrm{wei}$ & AUC$_\mathrm{mac}$ & AUC$_\mathrm{mic}$ & P & R & F1 & P & R & F1 & P & R & F1", rows, (4,))

d = read("m3_dt_pruning_comparison", index_col=0)
rows = [[n.split(" (")[0].capitalize(), num(r["train macro-F1"], 3), num(r["CV macro-F1"], 3), num(r["CV accuracy"], 3), num(r["leaves"], 0)] for n, r in d.iterrows()]
write("m3_prune", "@{}lrrrr@{}", r"Tree & Train F1 & CV F1 & CV acc. & Leaves", rows)

# best CV score per value of every Module 3 hyperparameter (marginal view of the full grids)
def marg(tab, col, f=str):
    x = tab.copy()
    x[col] = x[col].fillna("None").map(str)
    g_ = x.groupby(col)["mean_test_f1_macro"].max()
    return ", ".join(f"{f(k)} {v:.3f}" for k, v in g_.items())

r4 = read("m3_regression_results", index_col=0)
rows = [[n.replace("Single linear (OLS)", "Single linear (mileage)").replace("Multiple linear (OLS)", "Multiple linear"), num(r["train MSE"], 4), num(r["test MSE"], 4), num(r["train R2"], 3), num(r["test R2"], 3), num(r["test R2 ($ scale)"], 3)]
        for n, r in r4.iterrows()]
write("m3_reg", "@{}lrrrrr@{}", r"Model & MSE$_\mathrm{tr}$ & MSE$_\mathrm{te}$ & $R^2_\mathrm{tr}$ & $R^2_\mathrm{te}$ & $R^2_\mathrm{te}$(\$)", rows)

# ------------------------------------------------------------------ Module 4
q = read("m4_itemset_sweep")
rows = [[f"{r.min_sup:g}", num(r.frequent, 0), num(r.closed, 0), num(r.maximal, 0), num(r["max k"], 0)] for _, r in q.iterrows()]
write("m4_sweep", "@{}lrrrr@{}", r"min\_sup & Frequent & Closed & Maximal & max $k$", rows)

def rule(x):
    x = x.replace("suv_crossover", "SUV").replace("pickup_truck", "pickup").replace("used_only", "used")
    a, b = [p.strip() for p in x.split("→")]
    f = lambda s: r" $\wedge$ ".join(esc(i.strip()) for i in s.split("∧"))
    return f(a) + r" $\rightarrow$ " + f(b)

q = read("m4_top_rules")
write("m4_rules", r"@{}>{\raggedright\arraybackslash}p{0.72\linewidth}rrr@{}", "Rule & sup & conf & lift", [[rule(r.rule), num(r.support, 3), num(r.confidence, 2), num(r.lift, 2)] for r in q.itertuples()])
q = read("m4_rating_rules")
write("m4_rating", r"@{}>{\raggedright\arraybackslash}p{0.72\linewidth}rrr@{}", "Rule & sup & conf & lift", [[rule(r.rule), num(r.support, 3), num(r.confidence, 2), num(r.lift, 2)] for r in q.itertuples()], (4, 8))
q = read("m4_segment_rules")
q = q.groupby(q.rule.str[-2:], group_keys=False).head(2)
write("m4_segments", r"@{}>{\raggedright\arraybackslash}p{0.72\linewidth}rrr@{}", "Rule & sup & conf & lift", [[rule(r.rule), num(r.support, 3), num(r.confidence, 2), num(r.lift, 2)] for r in q.itertuples()], (2, 4, 6))
q = read("m4_top_itemsets").head(4)
fi = lambda x: r" $\wedge$ ".join(esc(i.strip().replace("suv_crossover", "SUV").replace("used_only", "used")) for i in x.split("∧"))
write("m4_itemsets", r"@{}>{\raggedright\arraybackslash}p{0.84\linewidth}r@{}", "Itemset & sup", [[fi(r.itemset), num(r.support, 3)] for r in q.itertuples()]
      + [[fi(r.itemset), num(r.support, 3)] for r in read("m4_top_rating_itemsets").groupby("rating").head(1).itertuples()], (4,))

q = read("m4_rule_classifier_cv")
w = q.pivot_table(index="aggregation", columns=["min_sup", "max antecedent"], values="CV macro-F1")
AG = ["best-antecedent", "top10", "mean + prior", "top10 + prior", "best-antecedent + prior", "class-best-lift", "class-best-conf", "mean"]
rows = [[a] + [(r"\textbf{%s}" if (a == "best-antecedent" and s_ == 0.005 and L == 4) else "%s") % f"{w.loc[a, (s_, L)]:.3f}"[1:] for s_ in (0.005, 0.01, 0.02) for L in (1, 2, 3, 4)] for a in AG]
write("m4_agg", "@{}l|rrrr|rrrr|rrrr@{}",
      r" & \multicolumn{4}{c|}{$s=0.005$, $L$} & \multicolumn{4}{c|}{$s=0.01$, $L$} & \multicolumn{4}{c}{$s=0.02$, $L$} \\ Aggregation & 1 & 2 & 3 & 4 & 1 & 2 & 3 & 4 & 1 & 2 & 3 & 4", rows)

q = read("m4_imputation")
rows = [[{"price": "Price tertile", "body": "Body type"}[r.attribute], num(r.levels, 0), num(r.rules, 0), num(100 * r["coverage"], 0),
         num(100 * r["accuracy (rules)"], 1), num(100 * r["accuracy (mode baseline)"], 1), num(r["macro-F1 (rules)"], 2), num(r["macro-F1 (mode baseline)"], 2)] for _, r in q.iterrows()]
write("m4_imp", "@{}lrrrrrrr@{}", r"Attribute & Levels & Rules & Cov.\,\% & Acc.\,\% & (mode) & F1$_\mathrm{mac}$ & (mode)", rows)
print("tables written to", OUT)

# ------------------------------------------------------------------ Notebook 05 (extensions)
M5 = ROOT / "report" / "make_tables_m5.py"
if M5.exists():
    exec(compile(M5.read_text(encoding="utf8"), str(M5), "exec"))
