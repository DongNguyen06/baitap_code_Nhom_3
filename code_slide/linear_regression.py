import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LinearRegressionScratch:
    """
    Mo hinh Linear Regression (Ordinary Least Squares - OLS) tu code tu dau (scratch).
    Su dung phuong trinh chuan (Normal Equation) voi ma tran gia nghich dao (Pseudo-inverse):
        theta = (X_b^T @ X_b)^(-1) @ X_b^T @ y = pinv(X_b) @ y
    """
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples = X_arr.shape[0]

        # Them cot 1 cho bias (he so chan)
        X_b = np.c_[np.ones((n_samples, 1)), X_arr]

        # Nghiem dong qua ma tran gia nghich dao pinv (on dinh khi co da cong tuyen)
        theta = np.linalg.pinv(X_b) @ y_arr

        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]
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

    # Step 3: Khoi tao va huan luyen mo hinh Linear Regression Scratch
    model = LinearRegressionScratch()
    model.fit(X_train_scaled, y_train)

    # Step 4: Du doan va in ket qua danh gia mo hinh
    y_pred = model.predict(X_test_scaled)
    rmse_test = np.sqrt(np.mean((np.asarray(y_test) - y_pred) ** 2))

    print(f"R2 Train : {model.score(X_train_scaled, y_train):.2f}")
    print(f"R2 Test  : {model.score(X_test_scaled, y_test):.2f}")
    print(f"RMSE Test: {rmse_test:.2f}")
