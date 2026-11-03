# Linear Regression (Hồi quy Tuyến tính Cổ điển - OLS)

### 1. Mục tiêu
* Giải quyết bài toán **hồi quy (Regression)**: Dự đoán một giá trị số thực liên tục $y$ dựa trên mối quan hệ tuyến tính với các đặc trưng đầu vào.
* Tìm một đường thẳng (hoặc siêu phẳng trong không gian nhiều chiều) vừa vặn nhất với dữ liệu, đóng vai trò làm mô hình đối chứng cơ sở (baseline) trực quan và dễ giải thích nhất.

### 2. Input / Output
* **Đầu vào (Input):** Bảng dữ liệu chứa các vector đặc trưng số $x = (x_1, x_2, \dots, x_n)$ và giá trị mục tiêu thực tế liên tục $y \in \mathbb{R}$.
* **Đầu ra (Output):** Con số dự đoán cụ thể $\hat{y} = w \cdot x + b$.
* **Tham số mô hình học:** Bộ trọng số $w = (w_1, w_2, \dots, w_n)$ và hệ số chặn (bias) $b$.

### 3. Evaluation Metrics / Criteria (Tiêu chí & Độ đo)
* **Tiêu chí tối ưu (Loss Function):** Tối thiểu hóa tổng bình phương sai số (Mean Squared Error - MSE / RSS) giữa giá trị thực tế $y$ và giá trị dự đoán $\hat{y}$: $J = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$.
* **Độ đo đánh giá hiệu năng:**
  * **R2-Score ($R^2$):** Tỷ lệ biến thiên của biến mục tiêu mà mô hình giải thích được (càng gần 1 càng tốt).
  * **RMSE (Root Mean Squared Error):** Độ lệch sai số trung bình thực tế (cùng đơn vị với biến mục tiêu, phạt nặng các lỗi lệch lớn).
  * **MAE (Mean Absolute Error):** Sai số tuyệt đối trung bình, phản ánh độ lệch thực tế trực quan và ít bị ảnh hưởng bởi ngoại lai.

### 4. How it works (Cách hoạt động)
*(Tóm lược: Tìm nghiệm tối ưu trực tiếp trong một bước tính toán đại số ma trận duy nhất bằng công thức Normal Equation hoặc tối ưu từng bước qua Gradient Descent kèm phạt L2).*

1. **Thêm hệ số chặn (b):** Ghép thêm cột toàn số 1 vào bảng dữ liệu ($X_b = [1, X]$) để học mức giá trị nền tảng cơ bản khi các đặc trưng bằng 0.
2. **Đo sai số tổng thể:** Tính khoảng cách chênh lệch bình phương giữa giá trị thực tế và đường thẳng dự đoán.
3. **Giải phương trình nghiệm đóng (Normal Equation):** Tính thẳng bộ trọng số tối ưu qua công thức ma trận có điều chuẩn Ridge L2: $\theta = (X_b^T X_b + \lambda I)^{-1} X_b^T y$.
4. **Dự đoán:** Mẫu mới có giá trị bằng tích vô hướng các đặc trưng nhân với trọng số tương ứng cộng hệ số chặn: $\hat{y} = w \cdot x + b$.

### 5. Dữ liệu sử dụng
* **Dạng dữ liệu phù hợp:** Dữ liệu dạng bảng số (tabular), trong đó biến mục tiêu là đại lượng đo lường liên tục.
* **Yêu cầu tiền xử lý:** Cần chuẩn hóa thang đo (StandardScaler), xử lý ngoại lai, và loại bỏ các đặc trưng đa cộng tuyến cao ($|r| > 0.6$).
