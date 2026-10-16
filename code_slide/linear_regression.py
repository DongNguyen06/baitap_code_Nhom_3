import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Đọc dữ liệu thực tế và chuẩn bị X, y (Tự động nhận diện đường dẫn chuẩn)
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'housing_market_data.csv')
df = pd.read_csv(data_path)
X = df.drop(columns=['house_price'])
y = df['house_price']

# Step 2: Phân chia tập Train (65%) - Test (35%) & Chuẩn hóa đặc trưng
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=12)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Khởi tạo và huấn luyện mô hình Linear Regression (OLS)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 4: Dự đoán và in kết quả đánh giá mô hình
y_pred = model.predict(X_test_scaled)
print(f"R2 Train : {model.score(X_train_scaled, y_train):.2f}")
print(f"R2 Test  : {r2_score(y_test, y_pred):.2f}")
print(f"RMSE Test: {mean_squared_error(y_test, y_pred)**0.5:.2f}")
