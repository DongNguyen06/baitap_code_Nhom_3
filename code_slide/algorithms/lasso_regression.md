# Lasso Regression (Hồi quy Lasso - Phạt L1)

## 1. Name
- **Tên:** Lasso Regression ($L_1$ Regularization / Coordinate Descent).

## 2. Problem
- Dữ liệu có nhiều đặc trưng gây nhiễu hoặc dư thừa thông tin. Cần một cơ chế vừa dự đoán chính xác, vừa tự động gạt bỏ các yếu tố không quan trọng để mô hình gọn gàng.

## 3. Target
- Ép triệt để trọng số của các biến không quan trọng về đúng bằng 0, tạo ra mô hình thưa (sparse model).

## 4. Input / Output
| Thành phần | Dữ liệu | Mô tả |
|---|---|---|
| **Input** | `X` (17 đặc trưng) | Đặc trưng bất động sản đã chuẩn hóa |
| **Output** | `house_price`, biến lọc | Giá nhà dự đoán và danh sách các biến được giữ lại |

## 5. Core features
- **Lọc đặc trưng tự động:** Tự động loại bỏ biến rác bằng cách đưa trọng số về đúng số 0.
- **Hình phạt L1:** Thêm đại lượng $\alpha \|\mathbf{w}\|_1$ vào hàm mất mát.
- **Hạ tọa độ (Coordinate Descent):** Tối ưu từng trọng số luân phiên, không phụ thuộc đạo hàm tại điểm 0.
- **Toán tử Soft-Thresholding:** Thu nhỏ ngưỡng mềm giúp mô hình chọn lọc biến dứt khoát.

## 6. Technology
- **Python 3**, **NumPy** (cài đặt thuật toán Coordinate Descent từ đầu), **Pandas**, **Scikit-learn**.

## 7. Evaluation
- **R2-Score:** Đo lường độ chính xác tổng thể trên Train và Test.
- **RMSE:** Sai số dự báo trung bình theo đơn vị giá nhà.
- **Số lượng biến bị triệt tiêu:** Số đặc trưng bị gán trọng số 0 trên tổng số 17 biến đầu vào.

## 8. How it work
1. **Hàm mục tiêu Lasso:** $J(\mathbf{w}, b) = \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\mathbf{w} - b\|_2^2 + \alpha \|\mathbf{w}\|_1$.
2. **Hạ tọa độ:** Cố định các biến còn lại, tối ưu trọng số thứ $j$: $\rho_j = \mathbf{x}_j^T (\mathbf{y} - \mathbf{X}_{-j}\mathbf{w}_{-j} - b)$.
3. **Toán tử ngưỡng mềm:** $w_j = \frac{\text{sign}(\rho_j) \cdot \max(0, |\rho_j| - n\alpha)}{\|\mathbf{x}_j\|_2^2}$.
4. **Cập nhật hệ số chặn:** $b = \frac{1}{n} \sum (y_i - \mathbf{x}_i^T \mathbf{w})$.
5. **Hội tụ:** Lặp lại các chu kỳ duyệt qua toàn bộ biến cho đến khi độ thay đổi nhỏ hơn ngưỡng `tol`.

## 9. Data used
- **Dataset:** `housing_market_data.csv` (17 đặc trưng nhà ở, cột dự đoán `house_price`).
- **Tiền xử lý:** Chuẩn hóa `StandardScaler` trên tập Train.
- **Chia tập:** 65% Train / 35% Test.

## 10. Demo
```bash
python lasso_regression.py
```
Khung kết quả mẫu:
```text
Alpha (L1)         : x.xx
R2 Train           : 0.xx
R2 Test            : 0.xx
RMSE Test          : x.xx
Bien bi triet tieu : xx / 17
```
