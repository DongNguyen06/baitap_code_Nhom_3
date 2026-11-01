# Ridge Regression (Hồi quy Sườn núi - Phạt L2)

### 1. Mục tiêu
* Giải quyết bài toán **hồi quy tuyến tính khi dữ liệu bị đa cộng tuyến** (các biến đầu vào phụ thuộc lẫn nhau) hoặc khi mô hình bị học vẹt (Overfitting).
* Sử dụng hình phạt L2 để co nhỏ các trọng số về sát 0 nhưng không triệt tiêu hoàn toàn, giúp mô hình ổn định và khái quát hóa tốt hơn trên dữ liệu mới.

### 2. Input / Output
* **Đầu vào (Input):** Bảng dữ liệu chứa các vector đặc trưng số $x = (x_1, x_2, \dots, x_n)$ đã được chuẩn hóa thang đo và giá trị mục tiêu liên tục $y \in \mathbb{R}$.
* **Đầu ra (Output):** Con số dự đoán liên tục $\hat{y}$ với độ ổn định cao, không bị phóng đại sai số.
* **Tham số mô hình học:** Bộ trọng số co cụm $w$ và hệ số chặn (bias) $b$.

### 3. Evaluation Metrics / Criteria (Tiêu chí & Độ đo)
* **Tiêu chí tối ưu (Loss Function):** Sai số MSE cộng thêm hình phạt L2 (bình phương độ lớn các trọng số): $J = \text{MSE} + \alpha \sum_{i=1}^n w_i^2$. Trọng số nào càng lớn sẽ bị phạt càng nặng.
* **Độ đo đánh giá hiệu năng:**
  * **R2-Score & RMSE:** Đánh giá độ chính xác và mức độ sai số thực tế trên cả tập Train và tập Test.
  * **Độ co cụm trọng số:** Theo dõi mức độ thu nhỏ của các trọng số khi tăng dần hệ số phạt $\alpha$.

### 4. How it works (Cách hoạt động)
*(Tóm lược: Giữ nguyên cơ chế dự đoán tuyến tính nhưng cộng thêm hình phạt bình phương L2 vào hàm mất mát để ép các trọng số co nhỏ lại về gần 0, chống thổi phồng giá trị).*

1. **Chuẩn hóa dữ liệu:** Đưa toàn bộ các đặc trưng về cùng thang đo (như Z-score/StandardScaler) để hình phạt L2 được áp dụng công bằng cho mọi biến.
2. **Kích hoạt hình phạt L2:** Bắt buộc các trọng số lớn phải co cụm lại về sát 0, làm mượt đường dự đoán và giảm phương sai (variance) của mô hình.
3. **Giải nghiệm đóng an toàn:** Cộng thêm ma trận đơn vị $\alpha I$ vào ma trận tính toán: $\theta = (X_b^T X_b + \alpha I)^{-1} X_b^T y$, đảm bảo ma trận luôn khả nghịch và tính ra nghiệm ổn định ngay cả khi có đa cộng tuyến.
4. **Điều chỉnh lực phạt ($\alpha$):** Tăng $\alpha$ để tăng khả năng chống quá khớp; nếu đặt $\alpha = 0$, mô hình sẽ quay trở lại Linear Regression cổ điển.

### 5. Data used (Dữ liệu sử dụng)
* **Dạng dữ liệu phù hợp:** Dữ liệu dạng bảng số có nhiều thuộc tính tương quan cao với nhau (đa cộng tuyến) hoặc số thuộc tính xấp xỉ số mẫu ($d \approx N$).
* **Yêu cầu tiền xử lý:** **Bắt buộc phải chuẩn hóa thang đo (StandardScaler)** trước khi huấn luyện vì nếu các biến lệch thang đo, hình phạt L2 sẽ tác động bất bình đẳng.
* **Đặc điểm tương thích:** Giữ lại đầy đủ tất cả các đặc trưng nhưng kiểm soát chặt chẽ độ lớn của chúng, đặc biệt bền vững trước nhiễu.
* **Ví dụ ứng dụng thực tế:** Định giá tài sản trong thị trường có nhiều yếu tố tương quan, dự báo chuỗi chỉ số tài chính, ước lượng chỉ số kinh tế vĩ mô.
