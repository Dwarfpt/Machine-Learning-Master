"""Model combinations (ensembles) built from the best single models.

* MajorityVoteClassifier — "hard voting": every model says 1 (flop) or 0 using its OWN tuned threshold;
  the game is flagged when the majority agrees (e.g. at least 3 of 5). predict_proba returns the share of
  votes, so the ensemble can still be ranked (PR-AUC) and calibrated.
* soft voting (sklearn VotingClassifier, average of probabilities) and stacking (StackingClassifier: a logistic
  regression learns how to weigh the models' out-of-fold probabilities) are used as they are in sklearn.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class MajorityVoteClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimators: list[tuple[str, object]], thresholds: list[float] | None = None):
        self.estimators = estimators
        self.thresholds = thresholds

    def fit(self, X, y):
        self.estimators_ = [(n, clone(e).fit(X, y)) for n, e in self.estimators]
        self.thresholds_ = list(self.thresholds or [0.5] * len(self.estimators))
        self.classes_ = np.array([0, 1])
        return self

    def votes(self, X) -> np.ndarray:
        """(n_samples, n_models) matrix of 0/1 votes."""
        return np.column_stack([(e.predict_proba(X)[:, 1] >= t).astype(int)
                                for (_, e), t in zip(self.estimators_, self.thresholds_)])

    def predict_proba(self, X) -> np.ndarray:
        share = self.votes(X).mean(axis=1)
        return np.column_stack([1 - share, share])

    def predict(self, X) -> np.ndarray:
        return (self.votes(X).sum(axis=1) > len(self.estimators_) / 2).astype(int)
