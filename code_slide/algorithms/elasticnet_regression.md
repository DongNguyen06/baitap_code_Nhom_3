# ElasticNet Regression (Hồi quy Dung hòa L1 và L2)

### 1. Mục tiêu
* Kết hợp thế mạnh của cả **Ridge ($L_2$)** và **Lasso ($L_1$)**, khắc phục nhược điểm chọn biến ngẫu nhiên của Lasso khi gặp các nhóm biến có tương quan mạnh với nhau.
* Vừa loại bỏ các đặc trưng gây nhiễu về đúng bằng 0, vừa giữ lại được cả nhóm các đặc trưng tương quan mật thiết cùng phát triển.

### 2. Input / Output
* **Đầu vào (Input):** Bảng dữ liệu chứa các vector đặc trưng số $x = (x_1, x_2, \dots, x_n)$ đã chuẩn hóa và giá trị mục tiêu liên tục $y \in \mathbb{R}$.
* **Đầu ra (Output):** Con số dự đoán liên tục $\hat{y}$ cùng số lượng đặc trưng được chắt lọc cân bằng.
* **Tham số mô hình học:** Bộ trọng số dung hòa $w$ và hệ số chặn (bias) $b$.

### 3. Evaluation Metrics / Criteria (Tiêu chí & Độ đo)
* **Tiêu chí tối ưu (Loss Function):** Kết hợp đồng thời sai số MSE cùng cả hai lực phạt L1 và L2:
  $$J = \text{MSE} + \alpha \cdot \rho \sum_{i=1}^n |w_i| + \frac{\alpha (1-\rho)}{2} \sum_{i=1}^n w_i^2$$
  *(Trong đó $\rho = \text{l1\_ratio} \in [0, 1]$ là tỷ lệ chia sẻ giữa L1 và L2).*
* **Độ đo đánh giá hiệu năng:**
  * **R2-Score & RMSE:** Đánh giá độ chính xác tổng thể trên tập kiểm tra.
  * **Tỷ lệ biến giữ lại:** Tỷ lệ đặc trưng được rút gọn hợp lý, không bị triệt tiêu quá đà như Lasso thuần túy.

### 4. How it works (Cách hoạt động)
*(Tóm lược: Dùng lực phạt L1 để loại bỏ các biến rác và dùng lực phạt L2 để giữ lại các nhóm biến có quan hệ tương quan chặt chẽ, tạo ra sự cân bằng tối ưu).*

1. **Thiết lập tỷ lệ phối hợp:** Dùng tham số $\rho$ (`l1_ratio`) để điều chỉnh: nghiêng về 1 là thiên về lọc biến (Lasso), nghiêng về 0 là thiên về ổn định trọng số (Ridge).
2. **Hạ tọa độ luân phiên:** Duyệt qua từng biến một cách tuần tự để tính toán mức độ cải thiện sai số của mô hình.
3. **Lọc biến theo nhóm:** 
   * Các biến thực sự vô dụng hoặc gây nhiễu sẽ bị thành phần L1 đưa thẳng về 0.
   * Các nhóm biến cùng có mối tương quan mạnh sẽ được thành phần L2 bảo vệ để cùng giữ lại, tránh hiện tượng bị loại bỏ ngẫu nhiên.
4. **Hội tụ:** Lặp lại các chu kỳ duyệt qua từng biến cho đến khi toàn bộ trọng số đạt trạng thái cân bằng ổn định.

### 5. Data used (Dữ liệu sử dụng)
* **Dạng dữ liệu phù hợp:** Dữ liệu có số chiều lớn, nhiều thuộc tính có tương quan phức tạp theo từng nhóm (Collinear groups).
* **Yêu cầu tiền xử lý:** **Bắt buộc chuẩn hóa thang đo (StandardScaler)** trước khi đưa vào mô hình.
* **Đặc điểm tương thích:** Là giải pháp an toàn và toàn diện nhất trong họ các mô hình tuyến tính có điều chuẩn khi chưa biết chắc nên ưu tiên lọc biến (L1) hay làm mượt (L2).
* **Ví dụ ứng dụng thực tế:** Phân tích dữ liệu tài chính - thị trường chứng khoán (các mã cổ phiếu cùng ngành biến động cùng nhau), nghiên cứu biến dị di truyền sinh học, dự báo thị trường bán lẻ đa kênh.
