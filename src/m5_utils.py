"""Shared code for notebook 05 (quality extensions): row-local feature engineering, a configurable
leak-free preprocessing step, classifiers with a tunable prior-correction exponent, X-Means, and a
parametrised copy of the Module 4 rule-based classifier (bins and binning strategy as arguments).

Kept in a module so that parallel CV workers can import it.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

CLUSTER_COLS = ["n_listings", "price_median", "savings_amount_median", "daysonmarket_median",
                "mileage_median", "pct_is_new", "listing_span_days"]

# ----------------------------------------------------------------- feature engineering (row-local, so leak-free)
ENGINEERED = {
    "listings_per_day": "n_listings / n_listing_days",
    "active_share": "n_listing_days / (listing_span_days + 1)",
    "listing_rate": "n_listings / (listing_span_days + 1)",
    "savings_ratio": "savings / price",
    "n_new": "n_listings * pct_is_new",
    "n_used": "n_listings * (1 - pct_is_new)",
    "price_per_hp": "price / horsepower",
    "miles_per_year": "mileage / (vehicle_age + 1)",
    "dom_over_span": "daysonmarket / (listing_span_days + 1)",
    "n_theft": "n_listings * pct_theft_title",
    "n_salvage": "n_listings * pct_salvage",
    "n_cab": "n_listings * pct_isCab",
}
ENGINEERED_LOG = [c for c in ENGINEERED if c not in ("active_share", "savings_ratio")]


def engineer(d):
    """Domain features computed from a single seller's own row (no dataset statistics)."""
    return pd.DataFrame({
        "listings_per_day": d.n_listings / d.n_listing_days,
        "active_share": d.n_listing_days / (d.listing_span_days + 1),
        "listing_rate": d.n_listings / (d.listing_span_days + 1),
        "savings_ratio": d.savings_amount_median / d.price_median,
        "n_new": d.n_listings * d.pct_is_new,
        "n_used": d.n_listings * (1 - d.pct_is_new),
        "price_per_hp": d.price_median / d.horsepower_median,
        "miles_per_year": d.mileage_median / (d.vehicle_age_median + 1),
        "dom_over_span": d.daysonmarket_median / (d.listing_span_days + 1),
        "n_theft": d.n_listings * d.pct_theft_title,
        "n_salvage": d.n_listings * d.pct_salvage,
        "n_cab": d.n_listings * d.pct_isCab,
    }, index=d.index)


class Prep(BaseEstimator, TransformerMixin):
    """Select -> (engineer) -> impute -> (winsorize) -> (log1p) -> (scale) -> one-hot. All statistics come from fit()."""

    def __init__(self, num, cat, log_vars, engineered=False, log=False, scaler=None, winsor=False, drop=(), clusters=False):
        self.num, self.cat, self.log_vars = num, cat, log_vars
        self.engineered, self.log, self.scaler = engineered, log, scaler
        self.winsor, self.drop, self.clusters = winsor, drop, clusters

    def _numeric(self, d):
        X = d[[c for c in self.num if c not in self.drop]]
        return pd.concat([X, engineer(d)], axis=1) if self.engineered else X.copy()

    def _transform_num(self, X):
        X = X.fillna(self.med_)
        if self.winsor:
            X = X.clip(self.lo_, self.hi_, axis=1)
        if self.log:
            lv = [c for c in X.columns if c in self.log_vars or c in ENGINEERED_LOG]
            X[lv] = np.log1p(X[lv])
        return X

    def _cluster_view(self, d):
        X = d[CLUSTER_COLS].fillna(self.cmed_)
        lv = [c for c in CLUSTER_COLS if c in self.log_vars]
        X[lv] = np.log1p(X[lv])
        return X

    def fit(self, d, y=None):
        X = self._numeric(d)
        self.med_ = X.median()
        self.lo_, self.hi_ = X.quantile(0.01), X.quantile(0.99)
        X = self._transform_num(X)
        self.num_names_ = list(X.columns)
        sc = {None: None, "standard": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}[self.scaler]
        self.sc_ = sc.fit(X) if sc is not None else None
        self.cat_ = [c for c in self.cat if c not in self.drop]
        if self.cat_:
            self.mode_ = d[self.cat_].mode().iloc[0]
            self.oh_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(d[self.cat_].fillna(self.mode_))
        if self.clusters:
            self.cmed_ = d[CLUSTER_COLS].median()
            V = self._cluster_view(d)
            self.csc_ = StandardScaler().fit(V)
            self.km_ = KMeans(4, n_init=10, random_state=42).fit(self.csc_.transform(V))
        return self

    def transform(self, d):
        X = self._transform_num(self._numeric(d))
        parts = [self.sc_.transform(X) if self.sc_ is not None else X.to_numpy(float)]
        if self.cat_:
            parts.append(self.oh_.transform(d[self.cat_].fillna(self.mode_)))
        if self.clusters:
            Dm = self.km_.transform(self.csc_.transform(self._cluster_view(d)))
            parts += [Dm, np.eye(4)[Dm.argmin(1)]]
        return np.hstack(parts)

    def get_feature_names(self):
        names = list(self.num_names_)
        if self.cat_:
            names += list(self.oh_.get_feature_names_out(self.cat_))
        if self.clusters:
            names += [f"dist_C{i}" for i in range(4)] + [f"is_C{i}" for i in range(4)]
        return names


# ----------------------------------------------------------------- tunable prior correction
class PriorPower(BaseEstimator, ClassifierMixin):
    """Predict argmax_c P(c | x) / P(c)^tau. tau=0 is the plain rule, tau=1 the full prior correction of Module 3."""

    def __init__(self, base=None, tau=0.0):
        self.base, self.tau = base, tau

    def fit(self, X, y):
        self.classes_, code = np.unique(np.asarray(y), return_inverse=True)
        self.prior_ = np.bincount(code) / len(code)
        self.base_ = clone(self.base).fit(X, code)
        return self

    def predict_proba(self, X):
        S = self.base_.predict_proba(X) / self.prior_ ** self.tau
        return S / S.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class TauTree(BaseEstimator, ClassifierMixin):
    """Decision tree whose class weights are P(c)^-tau (tau=0: unweighted, tau=1: equivalent to class_weight='balanced')."""

    def __init__(self, criterion="gini", max_depth=5, min_samples_leaf=50, tau=1.0, random_state=42):
        self.criterion, self.max_depth, self.min_samples_leaf = criterion, max_depth, min_samples_leaf
        self.tau, self.random_state = tau, random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_, code = np.unique(y, return_inverse=True)
        prior = np.bincount(code) / len(code)
        w = prior ** -self.tau
        self.tree_model_ = DecisionTreeClassifier(
            criterion=self.criterion, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
            class_weight=dict(enumerate(w / (w * prior).sum())), random_state=self.random_state).fit(X, code)
        return self

    def predict_proba(self, X):
        return self.tree_model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.tree_model_.predict(X)]


class OrdinalTree(BaseEstimator, ClassifierMixin):
    """Frank & Hall ordinal decomposition: two weighted binary trees for P(y > low) and P(y > medium)."""
    ORDER = ("low", "medium", "high")

    def __init__(self, max_depth=5, min_samples_leaf=50, tau=1.0, random_state=42):
        self.max_depth, self.min_samples_leaf, self.tau, self.random_state = max_depth, min_samples_leaf, tau, random_state

    def fit(self, X, y):
        r = np.array([self.ORDER.index(v) for v in y])
        self.classes_ = np.array(self.ORDER)
        self.prior_ = np.bincount(r, minlength=3) / len(r)
        self.trees_ = [DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                              random_state=self.random_state).fit(X, (r > t).astype(int)) for t in (0, 1)]
        return self

    def predict_proba(self, X):
        g0, g1 = (t.predict_proba(X)[:, 1] for t in self.trees_)
        P = np.clip(np.stack([1 - g0, g0 - g1, g1], 1), 1e-9, None) / self.prior_ ** self.tau
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class BinnedNB(BaseEstimator, ClassifierMixin):
    """CategoricalNB on quantile-binned numeric features (bins fit on the training data) + 0/1 one-hot columns, with tau."""

    def __init__(self, n_num=None, n_bins=3, alpha=1.0, tau=0.0):
        self.n_num, self.n_bins, self.alpha, self.tau = n_num, n_bins, alpha, tau

    def _bin(self, X):
        n = self.n_num_
        return np.hstack([self.kb_.transform(X[:, :n]), X[:, n:]]).astype(int)

    def fit(self, X, y):
        import warnings
        self.n_num_ = X.shape[1] if self.n_num is None else self.n_num
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.kb_ = KBinsDiscretizer(self.n_bins, encode="ordinal", strategy="quantile").fit(X[:, :self.n_num_])
        self.pp_ = PriorPower(CategoricalNB(alpha=self.alpha, min_categories=max(self.n_bins, 2)), self.tau).fit(self._bin(X), y)
        self.classes_ = self.pp_.classes_
        return self

    def predict_proba(self, X):
        return self.pp_.predict_proba(self._bin(X))

    def predict(self, X):
        return self.pp_.predict(self._bin(X))


# ----------------------------------------------------------------- X-Means (Pelleg & Moore, 2000)
def _bic(X, labels, centers):
    """BIC of a hard spherical-Gaussian mixture with one shared variance (Pelleg & Moore)."""
    n, d = X.shape
    k = len(centers)
    sse = ((X - centers[labels]) ** 2).sum()
    var = max(sse / (d * (n - k)), 1e-12)
    nj = np.bincount(labels, minlength=k)
    ll = (nj * np.log(np.maximum(nj, 1) / n)).sum() - n * d / 2 * np.log(2 * np.pi * var) - d * (n - k) / 2
    return ll - (k * (d + 1)) / 2 * np.log(n)


def xmeans(X, k_min=2, k_max=20, seed=42):
    """Alternate global K-Means with BIC-tested 2-way splits. Returns (labels, history of (k, global BIC, n proposed splits))."""
    k, hist, init = k_min, [], None
    while True:
        km = KMeans(k, n_init=10 if init is None else 1, init="k-means++" if init is None else init, random_state=seed).fit(X)
        new, splits = [], 0
        for c in range(k):
            S = X[km.labels_ == c]
            if len(S) > 20:
                two = KMeans(2, n_init=5, random_state=seed).fit(S)
                if _bic(S, two.labels_, two.cluster_centers_) > _bic(S, np.zeros(len(S), int), S.mean(0, keepdims=True)):
                    new += list(two.cluster_centers_)
                    splits += 1
                    continue
            new.append(km.cluster_centers_[c])
        hist.append((k, _bic(X, km.labels_, km.cluster_centers_), splits))
        if splits == 0 or k >= k_max:
            return km.labels_, hist
        init = np.array(new[:k_max])
        k = len(init)


# ----------------------------------------------------------------- parametrised Module 4 encoding + rule classifier
TERTILE = {"n_listings": "size", "price_median": "price", "mileage_median": "mileage", "daysonmarket_median": "dom",
           "horsepower_median": "hp", "listing_span_days": "span", "pct_isCab": "cab"}
CATEG = {"dominant_body_type": "body", "dominant_listing_color": "color", "dominant_transmission": "trans",
         "franchise_status": "dealer", "new_used_profile": "inventory"}


def fit_encoder(ref, bins=3, strategy="quantile", max_support=0.9):
    ref = ref.copy()   # parallel workers may receive a read-only frame
    if strategy == "quantile":
        edges = {c: np.unique(ref[c].quantile(np.arange(1, bins) / bins).values) for c in TERTILE}
    else:  # equal width
        edges = {c: np.linspace(ref[c].min(), ref[c].max(), bins + 1)[1:-1] for c in TERTILE}
    enc = {"edges": edges, "sav_med": ref.savings_amount_median[ref.savings_amount_median > 0].median()}
    sup = encode_items(ref, enc).mean()
    enc["vocab"] = sup.index[sup <= max_support].tolist()
    return enc


def encode_items(d, enc):
    """Boolean seller x item matrix without rating items; bin j of a variable is the item 'name=j'."""
    cols = {}
    for c, n in TERTILE.items():
        v = d[c].to_numpy(float)
        b = np.searchsorted(enc["edges"][c], v, side="left")
        cols[n] = np.where(np.isnan(v), None, np.array([f"{n}={j}" for j in b], dtype=object))
    s = d.savings_amount_median
    cols["savings"] = np.select([s == 0, s <= enc["sav_med"], s > enc["sav_med"]],
                                ["savings=none", "savings=low", "savings=high"], default=None)
    cols["salvage"] = np.where(d.pct_salvage > 0, "salvage=yes", None)
    cols["theft"] = np.where(d.pct_theft_title > 0, "theft=yes", None)
    for c, n in CATEG.items():
        cols[n] = np.where(d[c].notna(), n + "=" + d[c].astype(str), None)
    B = pd.get_dummies(pd.DataFrame(cols, index=d.index), prefix="", prefix_sep="").astype(bool)
    return B.reindex(columns=sorted(enc["vocab"]), fill_value=False) if "vocab" in enc else B


class RuleClassifier:
    """Module 4 classifier with the 'best matching antecedent' aggregation; tau generalises its tie to the class prior."""

    def __init__(self, min_sup=0.005, max_ante=4, bins=3, strategy="quantile"):
        self.min_sup, self.max_ante, self.bins, self.strategy = min_sup, max_ante, bins, strategy

    def fit(self, d, target, classes):
        from mlxtend.frequent_patterns import apriori
        self.classes_ = np.array(classes)
        self.enc_ = fit_encoder(d, self.bins, self.strategy)
        B = encode_items(d, self.enc_)
        fi = apriori(B, min_support=self.min_sup, use_colnames=True, max_len=self.max_ante, low_memory=True)
        self.items_, self.itemsets_ = list(B.columns), list(fi.itemsets)
        col = {c: i for i, c in enumerate(self.items_)}
        A = np.zeros((len(fi), len(self.items_)), dtype=np.int16)
        for j, s in enumerate(fi.itemsets):
            A[j, [col[i] for i in s]] = 1
        self.A_, self.alen_ = A, A.sum(1)
        Y = np.stack([(d[target] == c).to_numpy() for c in classes], 1).astype(float)
        M = self.match(d)
        self.cnt_ = np.stack([M[Y[:, k] > 0].sum(0) for k in range(len(classes))], 1).astype(float)
        del M
        self.conf_ = self.cnt_ / self.cnt_.sum(1, keepdims=True)
        self.prior_ = Y.mean(0)
        self.lift_ = self.conf_ / self.prior_
        return self

    def match(self, d):
        X = encode_items(d, self.enc_).to_numpy(np.int16)
        return np.vstack([(X[i:i + 1000] @ self.A_.T) == self.alen_ for i in range(0, len(X), 1000)])   # chunks keep memory low

    def predict(self, d, tau=0.0, M=None):
        M = self.match(d) if M is None else M
        score = self.conf_[np.where(M, self.lift_.max(1), -1).argmax(1)] / self.prior_ ** tau
        return self.classes_[score.argmax(1)]
