# Linear Regression (Hồi quy Tuyến tính Cổ điển - OLS)

## 1. Name
- **Tên:** Linear Regression (Ordinary Least Squares - OLS).

## 2. Problem
- Dự đoán giá trị liên tục (giá nhà `house_price`) dựa trên các đặc trưng thị trường và hạ tầng. Cần mô hình đơn giản, tính toán nhanh và trực quan để làm mốc so sánh (baseline).

## 3. Target
- Tối thiểu hóa tổng bình phương sai số (Residual Sum of Squares - RSS).
- Tìm nghiệm đóng tối ưu toàn cục mà không cần bước học lặp lại (learning rate).

## 4. Input / Output
| Thành phần | Dữ liệu | Mô tả |
|---|---|---|
| **Input** | `X` (17 đặc trưng) | Các chỉ số kinh tế, xã hội, môi trường đã chuẩn hóa |
| **Output** | `house_price` | Giá trị dự đoán liên tục của căn nhà |

## 5. Core features
- **Nghiệm đóng (Closed-form):** Giải trực tiếp bằng ma trận giả nghịch đảo pinv, hội tụ tức thì.
- **Dễ giải thích:** Trọng số $w_i$ thể hiện trực tiếp mức tăng/giảm giá nhà theo từng đặc trưng.
- **Mô hình nền tảng:** Làm thước đo cơ sở để đối chiếu hiệu quả của các mô hình phạt điều chuẩn.
- **Nhạy cảm với đa cộng tuyến:** Dễ bị quá khớp (overfitting) khi các đặc trưng phụ thuộc lẫn nhau.

## 6. Technology
- **Python 3**, **NumPy** (tự cài đặt phương trình chuẩn từ đầu), **Pandas**, **Scikit-learn** (`StandardScaler`, `train_test_split`).

## 7. Evaluation
- **R2-Score:** Tỷ lệ phần trăm biến thiên của giá nhà được mô hình giải thích (càng gần 1 càng tốt).
- **RMSE (Root Mean Squared Error):** Độ lệch trung bình giữa giá dự đoán và giá thực tế.
- **Khoảng cách Train - Test:** Đánh giá mức độ quá khớp (overfitting) của mô hình.

## 8. How it work
1. **Bổ sung hệ số chặn (Bias):** Ghép thêm cột 1 vào ma trận đặc trưng: $\mathbf{X}_b = [\mathbf{1}, \mathbf{X}]$.
2. **Hàm mục tiêu OLS:** $J(\boldsymbol{\theta}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}_b \boldsymbol{\theta}\|_2^2$.
3. **Phương trình chuẩn:** Triệt tiêu đạo hàm về 0: $\mathbf{X}_b^T (\mathbf{X}_b \boldsymbol{\theta} - \mathbf{y}) = \mathbf{0}$.
4. **Nghiệm đóng pinv:** $\boldsymbol{\theta} = (\mathbf{X}_b^T \mathbf{X}_b)^{-1} \mathbf{X}_b^T \mathbf{y} = \mathbf{X}_b^\dagger \mathbf{y}$.
5. **Dự đoán:** $\hat{y} = \mathbf{X} \mathbf{w} + b$.

## 9. Data used
- **Dataset:** `housing_market_data.csv` (dữ liệu bất động sản gồm 17 đặc trưng và cột mục tiêu `house_price`).
- **Tiền xử lý:** Chuẩn hóa `StandardScaler` trên tập Train để đồng nhất thang đo.
- **Chia tập:** 65% Train / 35% Test.

## 10. Demo
```bash
python linear_regression.py
```
Khung kết quả mẫu:
```text
R2 Train : 0.xx
R2 Test  : 0.xx
RMSE Test: x.xx
```
