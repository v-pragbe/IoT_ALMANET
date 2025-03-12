from __future__ import annotations
import collections
import math

import tensorflow as tf
from river import base, utils

__all__ = ["ALMASWAClassifier", "HybridALMASWANeuralNetworkClassifier"]

class ALMASWAClassifier(base.Classifier):
    def __init__(
        self,
        p=2,
        alpha=0.9,
        B=1 / 0.9,
        C=2**0.5,
        swa_start=100,
        swa_freq=10
    ):
        self.p = p
        self.alpha = alpha
        self.B = B
        self.C = C
        self.swa_start = swa_start
        self.swa_freq = swa_freq

        self.w = collections.defaultdict(float)
        self.w_avg = collections.defaultdict(float)
        self.swa_n = 0
        self.k = 1

    def _raw_dot(self, x, w):
        return utils.math.dot(x, w)

    def _average_weights(self):
        for i, w_i in self.w.items():
            self.w_avg[i] = (self.w_avg[i] * self.swa_n + w_i) / (self.swa_n + 1)
        self.swa_n += 1

    def learn_one(self, x, y):
        y = int(y or -1)  # convert True->1, False->-1
        gamma = self.B * math.sqrt(self.p - 1) / math.sqrt(self.k)

        # Check margin violation
        if y * self._raw_dot(x, self.w) < (1 - self.alpha) * gamma:
            eta = self.C / (math.sqrt(self.p - 1) * math.sqrt(self.k))
            for i, xi in x.items():
                self.w[i] += eta * y * xi
            norm = utils.math.norm(self.w, order=self.p)
            if norm > 1:
                for i in self.w:
                    self.w[i] /= norm

        # Update SWA
        if self.k >= self.swa_start and (self.k - self.swa_start) % self.swa_freq == 0:
            self._average_weights()

        self.k += 1
        return self

    def predict_proba_one(self, x):
        if self.swa_n > 0:
            score = self._raw_dot(x, self.w_avg)
        else:
            score = self._raw_dot(x, self.w)
        prob = utils.math.sigmoid(score)
        return {False: 1 - prob, True: prob}


class HybridALMASWANeuralNetworkClassifier(base.Classifier):
    def __init__(
        self,
        alma_p=2,
        alma_alpha=0.9,
        alma_B=1 / 0.9,
        alma_C=2**0.5,
        alma_swa_start=100,
        alma_swa_freq=10,
        hidden_units=8,
        learning_rate=0.01,
        ensemble_weight=0.5
    ):
        # ALMA model
        self.alma = ALMASWAClassifier(
            p=alma_p,
            alpha=alma_alpha,
            B=alma_B,
            C=alma_C,
            swa_start=alma_swa_start,
            swa_freq=alma_swa_freq,
        )

        # Simple Keras model without specifying input shape
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_units, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        self.ensemble_weight = ensemble_weight

        # Keep track of feature order
        self.feature_order = []
        self.model_built = False

    def _update_feature_order(self, x: dict):
        for feat in x.keys():
            if feat not in self.feature_order:
                self.feature_order.append(feat)

    def _dict_to_tensor(self, x: dict) -> tf.Tensor:
        return tf.convert_to_tensor(
            [x.get(f, 0.0) for f in self.feature_order],
            dtype=tf.float32
        )[None, :]  # shape => (1, n_features)

    def learn_one(self, x, y):
        # Ensure we record any new features
        self._update_feature_order(x)

        x_tensor = self._dict_to_tensor(x)

        # Build the model if it's not yet built (dimension known on first sample)
        if not self.model_built:
            # Let Keras infer shapes by making one forward pass
            _ = self.model(x_tensor)  
            self.model_built = True

        # Convert y to {0, 1} for TF cross-entropy
        y_tf = tf.constant([[1.0 if y else 0.0]], dtype=tf.float32)

        # Train the neural net
        with tf.GradientTape() as tape:
            pred = self.model(x_tensor, training=True)  # shape => (1,1)
            loss = tf.keras.losses.binary_crossentropy(y_tf, pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Train the ALMA model (expects True->+1, False->-1)
        self.alma.learn_one(x, y)

        return self

    def predict_proba_one(self, x: dict):
        # Get ALMA probability
        p_alma = self.alma.predict_proba_one(x)[True]

        # Handle feature expansion carefully:
        # if new features appear after the model is built, shape mismatch might occur
        # for this example, we'll just ignore them
        if self.model_built:
            # Create the input vector using the existing feature_order
            x_tensor = tf.convert_to_tensor(
                [[x.get(f, 0.0) for f in self.feature_order]],
                dtype=tf.float32
            )
            p_nn = float(self.model(x_tensor, training=False).numpy()[0, 0])
        else:
            # If the model isn't built yet, fallback on ALMA only
            p_nn = 0.5

        p_combined = self.ensemble_weight * p_alma + (1 - self.ensemble_weight) * p_nn
        return {False: 1 - p_combined, True: p_combined}