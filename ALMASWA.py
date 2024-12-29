from __future__ import annotations
import collections
import math
from river import base, utils

__all__ = ["ALMASWAClassifier"]

class ALMASWAClassifier(base.Classifier):
    """Approximate Large Margin Algorithm (ALMA) with Stochastic Weight Averaging (SWA).

    Parameters
    ----------
    p : int
        The order of the norm used in the update rule (L1, L2, etc.).
    alpha : float
        Margin relaxation parameter, controls the margin.
    B : float
        Scaling factor for the margin.
    C : float
        Scaling factor for the learning rate.
    swa_start : int
        The number of iterations after which SWA should start.
    swa_freq : int
        The frequency (in iterations) at which weights are averaged.
    
    Attributes
    ----------
    w : collections.defaultdict
        The current weights.
    w_avg : collections.defaultdict
        The averaged weights used for prediction.
    swa_n : int
        Counter for how many times weights have been averaged (used in SWA).
    k : int
        The number of instances seen during training.

    Examples
    --------
    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing

    >>> dataset = datasets.Phishing()

    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     linear_model.ALMAClassifier(swa_start=100, swa_freq=10)
    ... )

    >>> metric = metrics.Accuracy()

    >>> evaluate.progressive_val_score(dataset, model, metric)
    Accuracy: 82.56%

    References
    ----------
    [^1]: [Gentile, Claudio. "A new approximate maximal margin classification algorithm." Journal of Machine Learning Research 2.Dec (2001): 213-242](http://www.jmlr.org/papers/volume2/gentile01a/gentile01a.pdf)

    """

    def __init__(self, p=2, alpha=0.9, B=1 / 0.9, C=2**0.5, swa_start=100, swa_freq=10):
        self.p = p
        self.alpha = alpha
        self.B = B
        self.C = C
        self.swa_start = swa_start  # When to start SWA
        self.swa_freq = swa_freq    # Frequency of weight averaging
        self.w = collections.defaultdict(float)
        self.w_avg = collections.defaultdict(float)  # Averaged weights
        self.swa_n = 0              # Counter for the number of weight averages
        self.k = 1                  # Instance counter

    def _raw_dot(self, x, w):
        """Compute the dot product between the input and given weight vector."""
        return utils.math.dot(x, w)

    def predict_proba_one(self, x):
        """Predict the probability using the averaged weights."""
        # Use averaged weights if SWA has started
        if self.swa_n > 0:
            yp = utils.math.sigmoid(self._raw_dot(x, self.w_avg))
        else:
            yp = utils.math.sigmoid(self._raw_dot(x, self.w))
        return {False: 1 - yp, True: yp}

    def _average_weights(self):
        """Averages the current weights with the stored SWA weights."""
        for i, w_i in self.w.items():
            self.w_avg[i] = (self.w_avg[i] * self.swa_n + w_i) / (self.swa_n + 1)
        self.swa_n += 1

    def learn_one(self, x, y):
        """Update the model with a single instance."""
        # Convert 0 to -1 for label
        y = int(y or -1)

        # Calculate margin
        gamma = self.B * math.sqrt(self.p - 1) / math.sqrt(self.k)

        # If the margin condition is violated
        if y * self._raw_dot(x, self.w) < (1 - self.alpha) * gamma:
            # Learning rate
            eta = self.C / (math.sqrt(self.p - 1) * math.sqrt(self.k))

            # Update weights
            for i, xi in x.items():
                self.w[i] += eta * y * xi

            # Normalize weights
            norm = utils.math.norm(self.w, order=self.p)
            for i in x:
                self.w[i] /= max(1, norm)

        # SWA logic: average the weights after certain iterations
        if self.k >= self.swa_start and (self.k - self.swa_start) % self.swa_freq == 0:
            self._average_weights()

        # Increment instance counter
        self.k += 1