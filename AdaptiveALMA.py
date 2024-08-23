import collections
import math
from collections import deque
from river import base, utils

class AdaptiveALMA(base.Classifier):
    def __init__(self, p=2, alpha=0.9, B=1 / 0.9, C=2**0.5, window_size=400, threshold=0.05):
        self.p = p
        self.alpha = alpha
        self.B = B
        self.C = C
        self.w = collections.defaultdict(float)
        self.k = 1
        self.window_size = window_size
        self.threshold = threshold
        self.error_window = deque(maxlen=window_size)
        self.gamma_log = []
        self.eta_log = []
        self.error_rate_log = []
        self.adjustments_log = []

    def _raw_dot(self, x):
        return utils.math.dot(x, self.w)

    def predict_proba_one(self, x):
        yp = utils.math.sigmoid(self._raw_dot(x))
        return {False: 1 - yp, True: yp}

    def learn_one(self, x, y):
        proba = self.predict_proba_one(x)
        predicted_y = max(proba, key=proba.get)
        actual_y = bool(y)
        error = int(predicted_y != actual_y)
        self.error_window.append(error)

        # Only store the last 1000 entries
        self.gamma_log = self.gamma_log[-10:]
        self.eta_log = self.eta_log[-10:]
        self.error_rate_log = self.error_rate_log[-10:]
        self.adjustments_log = self.adjustments_log[-10:]
        
        if len(self.error_window) == self.window_size:
            average_error = sum(self.error_window) / self.window_size
            self.error_rate_log.append(average_error)
            if average_error > self.threshold:
                self.handle_drift()

        y = int(y or -1)
        gamma = self.B * math.sqrt(self.p - 1) / math.sqrt(self.k)
        self.gamma_log.append(gamma)
        
        if y * self._raw_dot(x) < (1 - self.alpha) * gamma:
            eta = self.C / (math.sqrt(self.p - 1) * math.sqrt(self.k))
            self.eta_log.append(eta)
            for i, xi in x.items():
                self.w[i] += eta * y * xi

            norm = utils.math.norm(self.w, order=self.p)
            for i in x:
                self.w[i] /= max(1, norm)

            self.k += 1

    def handle_drift(self):
    # Only adjust if the number of calls to this function is a multiple of 5
        self.C *= 0.98 #decreases the learning rate 0.98 
        self.alpha *= 0.12 #0.12 
        self.adjustments_log.append((self.k, self.C, self.alpha))