# Lasso Regression (Hồi quy Lasso - Phạt L1 Tự động Lọc biến)

### 1. Mục tiêu
* Giải quyết bài toán **hồi quy tuyến tính kết hợp tự động chọn lọc đặc trưng (Feature Selection)**.
* Sử dụng hình phạt L1 (trị tuyệt đối) để ép thẳng trọng số của các biến không quan trọng hoặc gây nhiễu về đúng bằng 0, tạo ra một mô hình tinh gọn (sparse model) và dễ giải thích nhất.

### 2. Input / Output
* **Đầu vào (Input):** Bảng dữ liệu chứa các vector đặc trưng số $x = (x_1, x_2, \dots, x_n)$ đã chuẩn hóa và giá trị mục tiêu liên tục $y \in \mathbb{R}$.
* **Đầu ra (Output):** Con số dự đoán liên tục $\hat{y}$ cùng danh sách các đặc trưng quan trọng được giữ lại ($w \ne 0$).
* **Tham số mô hình học:** Bộ trọng số thưa $w$ (nhiều phần tử bằng 0) và hệ số chặn (bias) $b$.

### 3. Evaluation Metrics / Criteria (Tiêu chí & Độ đo)
* **Tiêu chí tối ưu (Loss Function):** Sai số MSE kết hợp hình phạt L1 (tổng trị tuyệt đối của các trọng số): $J = \text{MSE} + \alpha \sum_{i=1}^n |w_i|$.
* **Độ đo đánh giá hiệu năng:**
  * **R2-Score & RMSE:** Đánh giá độ chính xác và mức độ sai số dự đoán.
  * **Tỷ lệ triệt tiêu biến (Sparsity):** Số lượng đặc trưng có trọng số bị đưa về đúng bằng 0 trên tổng số biến ban đầu.

### 4. How it works (Cách hoạt động)
*(Tóm lược: Tối ưu từng biến một thông qua thuật toán hạ tọa độ; những biến có đóng góp yếu hơn ngưỡng phạt sẽ bị ép thẳng trọng số về 0 để tự động loại bỏ biến).*

1. **Khởi tạo:** Gán toàn bộ các trọng số ban đầu bằng 0 ($w = 0, b = 0$).
2. **Soi xét từng biến (Hạ tọa độ - Coordinate Descent):** Cố định toàn bộ các biến khác, chỉ tối ưu một biến duy nhất đang xét dựa trên phần sai số còn lại của mô hình.
3. **Áp dụng ngưỡng mềm (Soft-Thresholding):**
   * Nếu mức độ đóng góp của biến nhỏ hơn lực phạt $\alpha$, trọng số của biến đó lập tức bị gán bằng 0 (loại hẳn biến khỏi mô hình).
   * Nếu biến thực sự quan trọng, giữ lại và thu nhỏ bớt một lượng bằng $\alpha$.
4. **Lặp hội tụ:** Quét tuần tự lần lượt qua tất cả các biến cho đến khi bộ trọng số đạt trạng thái ổn định.
5. **Dự đoán:** Chỉ những đặc trưng còn giữ trọng số khác 0 mới tham gia tính toán kết quả: $\hat{y} = \sum_{j \in \{w \ne 0\}} w_j x_j + b$.

### 5. Data used (Dữ liệu sử dụng)
* **Dạng dữ liệu phù hợp:** Dữ liệu dạng bảng số có số lượng đặc trưng rất lớn nhưng nhiều biến dư thừa, không liên quan hoặc gây nhiễu ($d \gg N$).
* **Yêu cầu tiền xử lý:** **Bắt buộc phải chuẩn hóa thang đo (StandardScaler)** để đảm bảo hình phạt L1 tác động công bằng lên mọi biến.
* **Đặc điểm tương thích:** Cực kỳ hữu dụng khi cần một mô hình vừa dự đoán chính xác vừa chỉ ra rõ ràng đâu là những biến thực sự quan trọng nhất.
* **Ví dụ ứng dụng thực tế:** Phân tích dữ liệu y sinh / biểu hiện gene (hàng chục nghìn gene nhưng chỉ vài gene gây bệnh), định giá sản phẩm dựa trên tập thuộc tính khổng lồ, lọc biến trong kinh tế lượng.
