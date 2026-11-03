# Decision Tree Regressor (Cây Quyết định Hồi quy)

### 1. Mục tiêu
* Giải quyết bài toán **hồi quy phi tuyến tính**: Dự đoán giá trị số thực liên tục $y$ bằng cách chia nhỏ không gian dữ liệu phức tạp thành các vùng độc lập đơn giản.
* Tạo ra mô hình có tính minh bạch và trực quan cao nhất trong Machine Learning: toàn bộ quá trình suy luận giống hệt như một sơ đồ cây các câu hỏi điều kiện "Nếu ... thì ..." (If-Else), giúp con người dễ dàng hiểu rõ lý do vì sao mô hình đưa ra con số dự đoán đó.

### 2. Input / Output
* **Đầu vào (Input):** Bảng dữ liệu chứa các vector đặc trưng số $x = (x_1, x_2, \dots, x_n)$ và giá trị mục tiêu liên tục $y \in \mathbb{R}$.
* **Đầu ra (Output):** Con số dự đoán cụ thể $\hat{y}$ (chính là giá trị trung bình cộng của các mẫu rơi vào cùng một nút lá).
* **Tham số mô hình học:** Cấu trúc rẽ nhánh của cây (các cặp thuộc tính và ngưỡng phân tách: $x_j \le s$) cùng giá trị dự đoán tại từng nút lá $\bar{y}_{\text{leaf}}$.

### 3. Evaluation Metrics / Criteria (Tiêu chí & Độ đo)
* **Tiêu chí tối ưu (Splitting Criterion):** 
  * Sử dụng tiêu chí **Giảm phương sai (Variance Reduction / Mean Squared Error - MSE)**: Tại mỗi bước rẽ nhánh, thuật toán tìm thuộc tính $j$ và ngưỡng cắt $s$ sao cho tổng sai số bình phương của 2 nhánh con (Trái $L$ và Phải $R$) là nhỏ nhất:
    $$\text{MSE}_{\text{split}} = \frac{N_L}{N} \text{MSE}_L + \frac{N_R}{N} \text{MSE}_R$$
* **Độ đo đánh giá hiệu năng:**
  * **R2-Score ($R^2$):** Đánh giá tỷ lệ biến thiên của biến mục tiêu mà cây giải thích được.
  * **RMSE & MAE:** Đo lường độ lệch sai số dự đoán thực tế so với nhãn thật.

### 4. How it works (Cách hoạt động)
*(Tóm lược: Cắt không gian dữ liệu thành các ô nhỏ riêng biệt bằng chuỗi câu hỏi Đúng/Sai, giá trị dự đoán của mỗi ô bằng trung bình cộng giá trị của các mẫu rơi vào ô đó).*

1. **Bắt đầu từ gốc (Root node):** Coi toàn bộ tập dữ liệu ban đầu là một tập hợp chung duy nhất.
2. **Tìm nhát cắt tối ưu qua Phân vị (Quantile Splits):** Thay vì thử hàng chục nghìn điểm chia, thuật toán lấy mẫu các điểm phân vị phân bố để tìm nhanh ngưỡng rẽ nhánh giúp giảm phương sai (MSE) nhiều nhất.
3. **Phân nhánh đệ quy:** Chia tập dữ liệu thành 2 nhánh con: nhánh thỏa mãn điều kiện sang bên trái, nhánh không thỏa mãn sang bên phải.
4. **Dừng mọc cây (Stopping Conditions):** Dừng lại khi chạm ngưỡng `max_depth`, `min_samples_split`, hoặc `min_samples_leaf`.
5. **Dự đoán:** Mẫu mới đi từ gốc theo các câu hỏi điều kiện đến khi chạm lá; giá trị dự đoán bằng trung bình cộng các mẫu trong lá đó.

### 5. Dữ liệu sử dụng
* **Dạng dữ liệu phù hợp:** Dữ liệu dạng bảng số, có mối quan hệ phi tuyến tính phức tạp giữa các đặc trưng và biến mục tiêu.
* **Yêu cầu tiền xử lý:** Không nhạy cảm với thang đo; cần khống chế độ sâu để chống quá khớp (Overfitting).
