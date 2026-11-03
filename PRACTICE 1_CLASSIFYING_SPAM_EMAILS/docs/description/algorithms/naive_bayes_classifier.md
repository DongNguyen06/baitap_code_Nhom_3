# Naive Bayes Classifier (Phân loại Naive Bayes)

### 1. Mục tiêu
* Giải quyết bài toán **phân loại** (hỗ trợ tự nhiên cả bài toán 2 lớp và nhiều lớp) dựa trên định lý xác suất Bayes.
* Dựa vào giả định "ngây thơ": xem tất cả các đặc trưng xuất hiện hoàn toàn độc lập với nhau khi đã biết nhãn lớp, từ đó tính toán xem một mẫu dữ liệu nghiêng về lớp nào có xác suất cao hơn.
* Cung cấp một mô hình baseline đơn giản, tốc độ huấn luyện siêu tốc và tốn rất ít tài nguyên máy tính tính toán.

### 2. Input / Output
* **Đầu vào (Input):** Vector đặc trưng số $x = (x_1, x_2, \dots, x_n)$ (dạng tần suất xuất hiện, nhị phân có/không, hoặc biến liên tục) và nhãn thực tế $y$.
* **Đầu ra (Output):** Nhãn lớp dự đoán $\hat{y}$ có tổng xác suất cao nhất kèm xác suất cụ thể.
* **Tham số mô hình học:** Xác suất xuất hiện nền tảng của từng lớp $P(c)$ và bảng xác suất có điều kiện của từng đặc trưng theo từng lớp $P(x_i \mid c)$.

### 3. Evaluation Metrics / Criteria (Tiêu chí & Độ đo)
* **Tiêu chí tối ưu:** Sử dụng phương pháp thống kê đếm trực tiếp kết hợp kỹ thuật **làm mịn Laplace ($\alpha = 1.0$)** để phòng trường hợp gặp đặc trưng mới lạ không bị nhân toàn bộ xác suất về 0; không cần dùng vòng lặp tối ưu hàm mất mát.
* **Độ đo đánh giá hiệu năng:**
  * **Accuracy:** Tỷ lệ đoán trúng tổng thể trên toàn bộ tập kiểm tra.
  * **Precision:** Khi mô hình đã dự đoán một lớp thì độ chuẩn xác đạt bao nhiêu % (tránh đoán nhầm).
  * **Recall:** Bao phủ và bắt trúng được bao nhiêu % các mẫu thực tế của lớp đó (tránh bỏ sót).
  * **F1-Score:** Điểm số cân bằng điều hòa giữa Precision và Recall.
  * **Confusion Matrix:** Bảng ma trận tổng kết số lượng các ca đoán đúng và đoán sai giữa các lớp.

### 4. How it works (Cách hoạt động)
*(Tóm lược: Đếm tần suất xuất hiện của các đặc trưng trong từng lớp, cộng làm mịn Laplace để tránh nhân với 0, sau đó chuyển sang cộng log-xác suất để chọn ra lớp có tổng điểm cao nhất).*

1. **Thống kê xác suất lớp:** Đếm tỷ lệ số lượng mẫu của từng lớp trên tổng số dữ liệu để có xác suất nền tảng: $P(c) = \frac{N_c}{N}$.
2. **Đếm tần suất đặc trưng:** Đếm xem từng đặc trưng $x_i$ xuất hiện bao nhiêu lần trong từng lớp $c$.
3. **Làm mịn Laplace (cộng 1):** Cộng thêm 1 vào tử số và cộng thêm tổng số đặc trưng vào mẫu số, đảm bảo không có đặc trưng nào bị gán xác suất bằng 0: $P(x_i \mid c) = \frac{\text{count}(x_i, c) + 1}{\sum \text{count}(x, c) + d}$.
4. **Đổi sang cộng Log:** Chuyển phép nhân các xác suất nhỏ thành phép cộng Log (Log-Sum) để máy tính tính toán chính xác, không bị lỗi tràn số: $\text{Score}(c) = \ln P(c) + \sum \ln P(x_i \mid c)$.
5. **Dự đoán:** Khi có mẫu dữ liệu mới, cộng tổng điểm Log của các đặc trưng có mặt trong mẫu. Lớp nào đạt tổng điểm lớn nhất thì gán nhãn lớp đó: $\hat{y} = \arg\max_c \text{Score}(c)$.

### 5. Data used (Dữ liệu sử dụng)
* **Dạng dữ liệu phù hợp:** Dữ liệu dạng bảng rời rạc, ma trận đếm số lần xuất hiện, ma trận thưa nhiều chiều (như Bag-of-Words, TF-IDF trong xử lý văn bản).
* **Yêu cầu tiền xử lý:** Cần lọc bớt các từ dừng vô nghĩa, làm sạch ký tự lạ, và áp dụng làm mịn Laplace khi dữ liệu có nhiều giá trị hiếm.
* **Đặc điểm tương thích:** Cực kỳ mạnh khi dữ liệu có số chiều rất lớn (hàng nghìn thuộc tính), chỉ cần quét qua dữ liệu một lần là học xong, không tốn tài nguyên tính toán.
* **Ví dụ ứng dụng thực tế:** Phân loại email rác (Spam/Ham), phân loại chủ đề tin tức, phân tích cảm xúc bình luận (tích cực/tiêu cực), phát hiện nội dung độc hại.
