# ElasticNet Regression (Hồi quy Dung hòa L1 và L2)

## 1. Name
- **Tên:** ElasticNet Regression (Dung hòa $L_1$ và $L_2$ Regularization).

## 2. Problem
- Dữ liệu vừa có nhiều biến dư thừa cần loại bỏ, vừa có các nhóm biến tương quan cao mà Lasso dễ chọn ngẫu nhiên một biến và bỏ rơi các biến còn lại trong nhóm.

## 3. Target
- Dung hòa ưu điểm của cả Ridge ($L_2$) và Lasso ($L_1$), giữ lại cả nhóm đặc trưng tương quan quan trọng đồng thời triệt tiêu các đặc trưng gây nhiễu.

## 4. Input / Output
| Thành phần | Dữ liệu | Mô tả |
|---|---|---|
| **Input** | `X` (17 đặc trưng) | Dữ liệu đặc trưng nhà ở đã qua chuẩn hóa |
| **Output** | `house_price`, biến lọc | Giá nhà dự đoán cùng số lượng biến thực tế được giữ lại |

## 5. Core features
- **Hiệu ứng nhóm (Grouping Effect):** Giữ hoặc giảm đều cả cụm biến có tương quan mạnh thay vì loại bỏ cảm tính.
- **Dung hòa 2 hình phạt:** Kết hợp linh hoạt giữa L1 (lọc biến) và L2 (ổn định trọng số).
- **Tỷ lệ l1_ratio ($\rho$):** Tự do trượt từ thuần Ridge ($\rho = 0$) sang thuần Lasso ($\rho = 1$).
- **Tối ưu Coordinate Descent:** Sử dụng thuật toán hạ tọa độ cải tiến có co cụm mẫu số theo $L_2$.

## 6. Technology
- **Python 3**, **NumPy** (cài đặt thuật toán Coordinate Descent dung hòa từ đầu), **Pandas**, **Scikit-learn**.

## 7. Evaluation
- **R2-Score:** Đo lường độ tương thích trên tập Train và tập Test.
- **RMSE:** Sai lệch dự đoán giá nhà so với giá thực tế.
- **Số lượng biến bị triệt tiêu:** Tỷ lệ số biến được rút gọn phù hợp với bài toán.

## 8. How it work
1. **Hàm mục tiêu ElasticNet:** $J(\mathbf{w}, b) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\mathbf{w} - b\|_2^2 + \alpha \rho \|\mathbf{w}\|_1 + \frac{\alpha (1-\rho)}{2} \|\mathbf{w}\|_2^2$.
2. **Tính toán tương quan phần dư:** $\rho_j = \mathbf{x}_j^T (\mathbf{y} - \mathbf{X}_{-j}\mathbf{w}_{-j} - b)$.
3. **Ngưỡng mềm cải tiến:** $w_j = \frac{\text{sign}(\rho_j) \cdot \max(0, |\rho_j| - n \alpha \rho)}{\|\mathbf{x}_j\|_2^2 + n \alpha (1 - \rho)}$.
4. **Cập nhật hệ số chặn:** $b = \frac{1}{n} \sum (y_i - \mathbf{x}_i^T \mathbf{w})$.
5. **Hội tụ:** Lặp lại việc quét qua toàn bộ trọng số cho tới khi độ thay đổi cực đại nhỏ hơn ngưỡng `tol`.

## 9. Data used
- **Dataset:** `housing_market_data.csv` (17 đặc trưng bất động sản, cột mục tiêu `house_price`).
- **Tiền xử lý:** Chuẩn hóa `StandardScaler` tập Train.
- **Chia tập:** 65% Train / 35% Test.

## 10. Demo
```bash
python elasticnet_regression.py
```
Khung kết quả mẫu:
```text
Alpha (Tong hop)   : x.xx
L1 Ratio (Ty le)   : x.xx
R2 Train           : 0.xx
R2 Test            : 0.xx
RMSE Test          : x.xx
Bien bi triet tieu : xx / 17
```
