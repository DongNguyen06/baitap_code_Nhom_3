import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin


class MultinomialNaiveBayesScratch(ClassifierMixin, BaseEstimator):
    """Class Multinomial Naive Bayes tự cài đặt thuần túy bằng NumPy."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)  # Hệ số làm mịn Laplace

    def get_params(self, deep: bool = True) -> dict:
        """Hàm tự viết trả về tham số mô hình (Python thuần túy)."""
        return {'alpha': self.alpha}

    def set_params(self, **params):
        """Hàm tự viết cập nhật tham số mô hình (Python thuần túy)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """
        Huấn luyện mô hình Naive Bayes từ dữ liệu X, y:
        1. Xác suất tiên nghiệm P(c): log P(c) = log(N_c / N)
        2. Tần suất từ vựng và xác suất có điều kiện P(w|c) kèm làm mịn Laplace.
        """
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.class_log_prior_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_log_prob_ = np.zeros((n_classes, n_features), dtype=np.float64)

        for idx, c in enumerate(self.classes_):
            mask = (y_arr == c)
            X_c = X[mask]
            # 1. Xác suất tiên nghiệm
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / n_samples)
            # 2. Đếm tần suất từ vựng trong lớp c + làm mịn Laplace alpha
            counts = np.asarray(X_c.sum(axis=0)).ravel() + self.alpha
            # 3. Xác suất có điều kiện: P(w|c) = (count + alpha) / (total + alpha * V)
            self.feature_log_prob_[idx] = np.log(counts / counts.sum())

        # Hệ số trọng số phục vụ giải thích mô hình
        if n_classes == 2:
            self.coef_ = (self.feature_log_prob_[1] - self.feature_log_prob_[0]).reshape(1, -1)
            self.intercept_ = np.array([self.class_log_prior_[1] - self.class_log_prior_[0]])
        else:
            self.coef_ = self.feature_log_prob_
            self.intercept_ = self.class_log_prior_
        return self

    def _joint_log_likelihood(self, X):
        """Tính điểm log-likelihood: log P(c) + sum(x_j * log P(w_j|c))"""
        scores = X.dot(self.feature_log_prob_.T) if sparse.issparse(X) else np.dot(X, self.feature_log_prob_.T)
        return scores + self.class_log_prior_

    def predict_proba(self, X) -> np.ndarray:
        """Dự đoán xác suất các lớp qua hàm Softmax ổn định số học."""
        jll = self._joint_log_likelihood(X)
        exp_jll = np.exp(jll - np.max(jll, axis=1, keepdims=True))
        return exp_jll / np.sum(exp_jll, axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """Dự đoán nhãn có log-likelihood cực đại."""
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def score(self, X, y) -> float:
        """Đo độ chính xác Accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
