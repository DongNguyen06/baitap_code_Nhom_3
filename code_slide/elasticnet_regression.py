import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score

# Tự động cấu hình mã hóa UTF-8 cho Windows Terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Step 1: Đọc dữ liệu thực tế và chuẩn bị X, y (Tự động nhận diện đường dẫn chuẩn)
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'housing_market_data.csv')
df = pd.read_csv(data_path)
X = df.drop(columns=['house_price'])
y = df['house_price']

# Step 2: Phân chia tập Train (65%) - Test (35%) & Chuẩn hóa
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=12)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Tự động tìm alpha và l1_ratio tối ưu bằng ElasticNetCV (5-Fold Cross-Validation)
model = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5, random_state=12)
model.fit(X_train_scaled, y_train)

# Step 4: Dự đoán và in kết quả đánh giá mô hình
y_pred = model.predict(X_test_scaled)
num_zeros = sum(model.coef_ == 0)

print(f"Alpha tối ưu (CV)   : {model.alpha_:.2f}")
print(f"L1 Ratio tối ưu (CV): {model.l1_ratio_:.2f}")
print(f"R2 Train : {model.score(X_train_scaled, y_train):.2f}")
print(f"R2 Test  : {r2_score(y_test, y_pred):.2f}")
print(f"RMSE Test: {mean_squared_error(y_test, y_pred)**0.5:.2f}")
print(f"Biến bị loại bỏ (=0): {num_zeros} / {len(model.coef_)}")
