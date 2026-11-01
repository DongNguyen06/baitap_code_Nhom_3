# Logistic Regression (Hồi quy Logistic)

### 1. Mục tiêu
* Giải quyết bài toán **phân loại nhị phân** (chia dữ liệu thành 2 lớp: 0 hoặc 1, có thể mở rộng cho bài toán nhiều lớp).
* Tìm một ranh giới phân tách (siêu phẳng) giữa 2 lớp bằng cách gán cho mỗi đặc trưng một trọng số: đặc trưng nghiêng về lớp 1 sẽ mang trọng số dương kéo điểm lên, đặc trưng nghiêng về lớp 0 mang trọng số âm kéo điểm xuống.
* Mô hình có tính minh bạch cao: nhìn vào trọng số có thể biết ngay đặc trưng nào tác động mạnh nhất đến kết quả dự đoán.

### 2. Input / Output
* **Đầu vào (Input):** Bảng dữ liệu chứa các vector đặc trưng số $x = (x_1, x_2, \dots, x_n)$ và nhãn thực tế $y \in \{0, 1\}$.
* **Đầu ra (Output):** Nhãn dự đoán $\hat{y} \in \{0, 1\}$ dựa trên xác suất $p \in [0, 1]$ sau khi so sánh với ngưỡng quyết định (mặc định 0.5).
* **Tham số mô hình học:** Bộ trọng số $w$ và hệ số chặn (bias) $b$.

### 3. Evaluation Metrics / Criteria (Tiêu chí & Độ đo)
* **Tiêu chí tối ưu (Loss Function):**
  * Sử dụng **Log-Loss (Binary Cross-Entropy)**: Hàm này phạt cực kỳ nặng những dự đoán tự tin thái quá mà bị sai (ví dụ: mô hình chắc chắn 99% là lớp 0 nhưng nhãn thật lại là lớp 1).
  * Kết hợp **Regularization (L1 / L2)**: Chiếc phanh hãm giúp triệt tiêu các đặc trưng gây nhiễu (L1) hoặc co nhỏ trọng số về gần 0 (L2), ngăn mô hình học vẹt (chống Overfitting).
* **Độ đo đánh giá hiệu năng:**
  * **Accuracy:** Tỷ lệ đoán trúng tổng thể trên toàn bộ tập kiểm tra.
  * **Precision:** Khi mô hình đã phán quyết là lớp 1 thì độ chuẩn xác đạt bao nhiêu % (tránh bắt nhầm).
  * **Recall:** Bắt trúng được bao nhiêu % các mẫu thực tế thuộc lớp 1 (tránh bỏ sót).
  * **F1-Score:** Điểm số cân bằng điều hòa giữa Precision và Recall.
  * **Confusion Matrix:** Bảng ma trận tổng kết số lượng các ca đoán đúng và đoán sai (TP, FP, TN, FN).

### 4. How it works (Cách hoạt động)
*(Tóm lược: Tính tổng điểm tuyến tính của các đặc trưng, nén qua hàm Sigmoid để quy ra xác suất từ 0 đến 1, đo sai số và cập nhật trọng số từng bước bằng Gradient Descent cho đến khi sai số đạt mức nhỏ nhất).*

1. **Khởi tạo:** Gán toàn bộ trọng số ban đầu bằng 0 ($w = 0, b = 0$).
2. **Tính điểm và nén xác suất:** Với mỗi mẫu dữ liệu, tính tổng điểm $z = w \cdot x + b$, sau đó đưa qua hàm **Sigmoid** để nén về giá trị xác suất mượt mà: $p = \frac{1}{1 + e^{-z}}$.
3. **Đo sai số:** So sánh xác suất vừa đoán với nhãn thật thông qua hàm mất mát Log-Loss để biết mô hình đang đoán lệch bao nhiêu.
4. **Tính hướng điều chỉnh (Gradient):** Tính đạo hàm của hàm mất mát để xác định xem cần tăng hay giảm trọng số của từng đặc trưng.
5. **Cập nhật trọng số:** Điều chỉnh trọng số từng bước một ngược chiều đạo hàm theo tốc độ học (Learning Rate): $w \leftarrow w - \alpha \cdot \text{grad}$. Quá trình này lặp lại qua nhiều vòng (epochs) cho đến khi sai số tổng thể đạt mức nhỏ nhất.
6. **Ra quyết định:** Khi có mẫu dữ liệu mới, nếu xác suất $p \ge 0.5$ thì gán nhãn 1, ngược lại nếu $p < 0.5$ thì gán nhãn 0.

### 5. Data used (Dữ liệu sử dụng)
* **Dạng dữ liệu phù hợp:** Dữ liệu dạng bảng (tabular), các vector đặc trưng số hoặc biến phân loại đã được mã hóa số.
* **Yêu cầu tiền xử lý:** Cần chuẩn hóa thang đo (Scaling), điền giá trị khuyết thiếu và hạn chế hiện tượng các biến phụ thuộc lẫn nhau (đa cộng tuyến).
* **Đặc điểm tương thích:** Hoạt động rất tốt khi dữ liệu có ranh giới phân tách gần như tuyến tính; tốc độ huấn luyện và suy luận rất nhanh, dễ triển khai thực tế.
* **Ví dụ ứng dụng thực tế:** Dự đoán khách hàng rời bỏ dịch vụ (Customer Churn), chấm điểm rủi ro tín dụng ngân hàng, chẩn đoán nguy cơ bệnh lý, phân loại thư rác (Spam/Ham).
