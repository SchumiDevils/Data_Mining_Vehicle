"""Shared estimators for Module 3 (kept in a module so parallel CV workers can import them)."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier


class PriorCorrectedKNN(BaseEstimator, ClassifierMixin):
    """KNN classifier with an optional class-prior-corrected decision rule.

    KNN has no class weights. With an imbalanced target, the neighbourhood vote
    is dominated by the majority class as k grows. With prior_correction=True,
    the predicted class is argmax_c P(c | x) / P(c), where P(c) is the class
    frequency in the training data. That is the same rule as reweighting every
    class to equal total weight (the analogue of class_weight="balanced" for
    trees and uniform priors for Naive Bayes). predict_proba is unchanged, so
    per-class ROC curves and AUC are unaffected.
    """

    def __init__(self, n_neighbors=5, p=2, weights="uniform", prior_correction=False):
        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights
        self.prior_correction = prior_correction

    def fit(self, X, y):
        y = np.asarray(y)
        # Labels are encoded as integers before fitting: scikit-learn 1.6 fails in predict_proba
        # (brute-force search, Manhattan distance, uniform weights) when the labels are strings.
        self.classes_, y_codes = np.unique(y, return_inverse=True)
        self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors, p=self.p, weights=self.weights).fit(X, y_codes)
        self.prior_ = np.bincount(y_codes, minlength=len(self.classes_)) / len(y_codes)
        return self

    def predict_proba(self, X):
        return self.knn_.predict_proba(X)

    def predict(self, X):
        P = self.predict_proba(X)
        if self.prior_correction:
            P = P / self.prior_
        return self.classes_[P.argmax(axis=1)]
