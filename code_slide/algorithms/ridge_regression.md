# Ridge Regression (Hồi quy Sườn núi - Phạt L2)

## 1. Name
- **Tên:** Ridge Regression ($L_2$ Regularization / Tikhonov).

## 2. Problem
- Dữ liệu xuất hiện hiện tượng đa cộng tuyến (các biến phụ thuộc lẫn nhau) khiến trọng số của mô hình tuyến tính bị thổi phồng quá mức, dẫn đến dự đoán kém trên dữ liệu mới.

## 3. Target
- Thêm hình phạt $L_2$ để co cụm trọng số, giảm phương sai (variance) và nâng cao khả năng khái quát hóa.

## 4. Input / Output
| Thành phần | Dữ liệu | Mô tả |
|---|---|---|
| **Input** | `X` (17 đặc trưng) | Dữ liệu đặc trưng nhà ở đã được co giãn thang đo |
| **Output** | `house_price` | Giá trị dự đoán ổn định của căn nhà |

## 5. Core features
- **Điều chuẩn L2:** Thêm đại lượng $\alpha \|\mathbf{w}\|_2^2$ nhằm siết chặt biên độ các trọng số lớn.
- **Co cụm trọng số (Weight Shrinkage):** Ép trọng số về sát 0 nhưng không triệt tiêu hoàn toàn biến nào.
- **Trị đa cộng tuyến:** Bảo đảm ma trận $(\mathbf{X}_b^T \mathbf{X}_b + \alpha \mathbf{I})$ luôn khả nghịch, giải quyết triệt để lỗi suy biến.
- **Nghiệm đóng ổn định:** Tính toán trực tiếp bằng `np.linalg.solve`, không cần lặp Gradient Descent.

## 6. Technology
- **Python 3**, **NumPy** (tự cài đặt nghiệm đóng Ridge), **Pandas**, **Scikit-learn** (`StandardScaler`, `train_test_split`).

## 7. Evaluation
- **R2-Score:** Đo lường độ khớp trên tập Train và tập Test.
- **RMSE:** Sai số trung bình thực tế khi áp dụng trên tập kiểm tra.
- **Độ ổn định trọng số:** Kiểm soát sự thu nhỏ của vector trọng số khi tăng $\alpha$.

## 8. How it work
1. **Chuẩn hóa:** Đưa 17 đặc trưng về trung bình 0, phương sai 1 để hình phạt áp dụng công bằng.
2. **Hàm mục tiêu Ridge:** $J(\boldsymbol{\theta}) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}_b \boldsymbol{\theta}\|_2^2 + \frac{\alpha}{2} \|\mathbf{w}\|_2^2$.
3. **Ma trận đơn vị không phạt bias:** Tạo ma trận $\mathbf{I}$ kích thước $(d+1) \times (d+1)$ với $\mathbf{I}[0, 0] = 0$.
4. **Nghiệm đóng:** $\boldsymbol{\theta} = (\mathbf{X}_b^T \mathbf{X}_b + \alpha \mathbf{I})^{-1} \mathbf{X}_b^T \mathbf{y}$.
5. **Điều chỉnh siêu tham số:** Tăng $\alpha$ để tăng lực phạt; khi $\alpha = 0$ trở về mô hình OLS thuần túy.

## 9. Data used
- **Dataset:** `housing_market_data.csv` (17 đặc trưng nhà ở, biến mục tiêu `house_price`).
- **Tiền xử lý:** Chuẩn hóa `StandardScaler` (bắt buộc đối với hồi quy có phạt điều chuẩn).
- **Chia tập:** 65% Train / 35% Test.

## 10. Demo
```bash
python ridge_regression.py
```
Khung kết quả mẫu:
```text
Alpha (L2) : xx.xx
R2 Train   : 0.xx
R2 Test    : 0.xx
RMSE Test  : x.xx
```
