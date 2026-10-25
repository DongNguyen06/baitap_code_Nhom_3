import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ElasticNetRegressionScratch:
    """
    Mo hinh ElasticNet Regression (ket hop dieu phat L1 va L2) tu code tu dau (scratch).
    Toi uu hoa bang Coordinate Descent ket hop Soft-Thresholding:
        Ham muc tieu theo chuan Scikit-Learn:
            (1 / (2 * n_samples)) * ||y - X w - b||_2^2
            + alpha * l1_ratio * ||w||_1
            + 0.5 * alpha * (1 - l1_ratio) * ||w||_2^2
    """
    def __init__(self, alpha=0.53, l1_ratio=0.30, max_iter=2000, tol=1e-5):
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.bias = 0.0
        self.intercept_ = 0.0
        self.weights = None
        self.coef_ = None
        self.n_iter_ = 0

    @staticmethod
    def _soft_thresholding(rho, threshold):
        """Toan tu thu nho duong mem (Soft-thresholding operator)."""
        if rho > threshold:
            return rho - threshold
        elif rho < -threshold:
            return rho + threshold
        else:
            return 0.0

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X_arr.shape

        # Triet tieu he so chan bang cach trung tam hoa bien muc tieu y
        self.bias = float(np.mean(y_arr))
        self.intercept_ = self.bias
        y_centered = y_arr - self.bias

        # Khoi tao trong so va phan du r
        self.weights = np.zeros(n_features, dtype=np.float64)
        r = y_centered.copy()

        # Tinh truoc chuan binh phuong cua tung dac trung: norm_x[j] = sum(X[:, j]**2)
        norm_x = np.sum(X_arr ** 2, axis=0)

        # Cac he so phat L1 va L2 theo ty le quy chuan cua Scikit-Learn
        l1_penalty = n_samples * self.alpha * self.l1_ratio
        l2_penalty = n_samples * self.alpha * (1.0 - self.l1_ratio)

        for iteration in range(self.max_iter):
            max_weight_change = 0.0
            for j in range(n_features):
                denom = norm_x[j] + l2_penalty
                if denom == 0:
                    continue

                w_old = self.weights[j]
                # rho_j = dot(X[:, j], r) + norm_x[j] * w_old
                rho_j = np.dot(X_arr[:, j], r) + norm_x[j] * w_old
                w_new = self._soft_thresholding(rho_j, l1_penalty) / denom

                # Cap nhat phan du hieu qua: r_moi = r_cu + X[:, j] * (w_old - w_new)
                r += X_arr[:, j] * (w_old - w_new)
                self.weights[j] = w_new

                max_weight_change = max(max_weight_change, abs(w_old - w_new))

            if max_weight_change < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.coef_ = self.weights.copy()
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr @ self.coef_ + self.intercept_

    def score(self, X, y):
        y_arr = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        return float(1.0 - (ss_res / ss_tot))


if __name__ == '__main__':
    # Tu dong cau hinh ma hoa UTF-8 cho Windows Terminal
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    # Step 1: Doc du lieu thuc te va chuan bi X, y (Tu dong nhan dien duong dan chuan)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'housing_market_data.csv')
    df = pd.read_csv(data_path)
    X = df.drop(columns=['house_price'])
    y = df['house_price']

    # Step 2: Phan chia tap Train (65%) - Test (35%) & Chuan hoa dac trung
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=12)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 3: Khoi tao va huan luyen mo hinh ElasticNet Regression Scratch
    model = ElasticNetRegressionScratch(alpha=0.53, l1_ratio=0.30)
    model.fit(X_train_scaled, y_train)

    # Step 4: Du doan va in ket qua danh gia mo hinh
    y_pred = model.predict(X_test_scaled)
    rmse_test = np.sqrt(np.mean((np.asarray(y_test) - y_pred) ** 2))
    num_zeros = int(np.sum(model.coef_ == 0.0))

    print(f"Alpha (Tong hop)   : {model.alpha:.2f}")
    print(f"L1 Ratio (Ty le)   : {model.l1_ratio:.2f}")
    print(f"R2 Train           : {model.score(X_train_scaled, y_train):.2f}")
    print(f"R2 Test            : {model.score(X_test_scaled, y_test):.2f}")
    print(f"RMSE Test          : {rmse_test:.2f}")
    print(f"Bien bi triet tieu : {num_zeros} / {len(model.coef_)}")
