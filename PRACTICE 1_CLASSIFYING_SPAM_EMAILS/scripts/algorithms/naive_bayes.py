"""
Thuật toán: Multinomial Naive Bayes (From Scratch)
Đặc tả chi tiết: docs/algorithms/naive_bayes_classifier.md
"""
import numpy as np
from scipy import sparse


class MultinomialNaiveBayesScratch:
    """Class Multinomial Naive Bayes tự cài đặt thuần túy bằng NumPy."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)  # Hệ số làm mịn Laplace

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            # 1. Xác suất tiên nghiệm: log P(c) = log(N_c / N)
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / n_samples)
            # 2. Đếm tần suất từ vựng trong lớp c + làm mịn Laplace
            counts = np.asarray(X_c.sum(axis=0)).ravel() + self.alpha
            # 3. Xác suất có điều kiện: P(w|c) = (count + alpha) / (total + alpha * V)
            self.feature_log_prob_[idx] = np.log(counts / counts.sum())
        return self

    def _joint_log_likelihood(self, X):
        # Điểm log = log P(c) + sum(x_j * log P(w_j|c))
        scores = X.dot(self.feature_log_prob_.T) if sparse.issparse(X) else np.dot(X, self.feature_log_prob_.T)
        return scores + self.class_log_prior_

    def predict(self, X):
        return self.classes[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        exp_jll = np.exp(jll - np.max(jll, axis=1, keepdims=True))  # Softmax ổn định số học
        return exp_jll / np.sum(exp_jll, axis=1, keepdims=True)
