# Support Vector Machine Classifier (Máy vector hỗ trợ - Linear SVM)

### 1. Mục tiêu
* Giải quyết bài toán **phân loại** (chủ yếu nhị phân và có thể mở rộng cho đa lớp).
* Tìm một đường ranh giới (siêu phẳng) phân tách rõ ràng giữa 2 lớp dữ liệu sao cho **khoảng cách an toàn (lề - margin) đối với cả 2 lớp là lớn nhất** (nguyên lý lề cực đại).
* Giúp mô hình có khả năng tổng quát hóa rất cao trên dữ liệu mới, không bị ảnh hưởng bởi các điểm dữ liệu nằm xa ranh giới và kiểm soát lỗi bắt nhầm cực tốt.

### 2. Input / Output
* **Đầu vào (Input):** Bảng dữ liệu chứa các vector đặc trưng số $x = (x_1, x_2, \dots, x_n)$ và nhãn thực tế quy đổi $y \in \{-1, +1\}$.
* **Đầu ra (Output):** Nhãn dự đoán $\hat{y} \in \{0, 1\}$ (hoặc $\{-1, +1\}$) dựa vào việc mẫu dữ liệu nằm ở phía bên nào của siêu phẳng phân tách.
* **Tham số mô hình học:** Vector trọng số pháp tuyến $w$, hệ số chặn $b$ và tập hợp các **Support Vectors** (những điểm dữ liệu then chốt nằm sát lề quyết định vị trí ranh giới).

### 3. Evaluation Metrics / Criteria (Tiêu chí & Độ đo)
* **Tiêu chí tối ưu (Loss Function):**
  * Sử dụng **Hinge Loss**: Nếu mẫu dữ liệu nằm đúng bên và cách xa ranh giới an toàn (khoảng cách $\text{margin} \ge 1$) thì không bị phạt gì cả (Loss = 0). Nếu mẫu dữ liệu lấn vào khoảng lề an toàn hoặc nằm sai bên thì sẽ bị phạt một lượng đúng bằng khoảng cách vi phạm.
  * Kết hợp **L2 Regularization**: Ép độ rộng của lề luôn ở mức lớn nhất ($2/\|w\|$) để mô hình không bị quá phụ thuộc vào nhiễu.
* **Độ đo đánh giá hiệu năng:**
  * **Accuracy:** Tỷ lệ đoán trúng tổng thể trên toàn bộ tập kiểm tra.
  * **Precision:** Khi đã phán đoán là lớp mục tiêu thì độ chuẩn xác đạt bao nhiêu % (rất mạnh ở việc hạn chế báo nhầm).
  * **Recall:** Bắt trúng được bao nhiêu % các mẫu thực tế (tránh bỏ sót).
  * **F1-Score:** Điểm số cân bằng điều hòa giữa Precision và Recall.
  * **Confusion Matrix:** Bảng ma trận tổng kết số lượng các ca đoán đúng và đoán sai (TP, FP, TN, FN).

### 4. How it works (Cách hoạt động)
*(Tóm lược: Thiết lập siêu phẳng có khoảng lề an toàn lớn nhất, chỉ phạt những điểm lấn lề hoặc nằm sai bên bằng Hinge Loss, và chỉ giữ lại các điểm sát ranh giới làm điểm tựa Support Vectors).*

1. **Khởi tạo:** Gán trọng số ban đầu bằng 0 ($w = 0, b = 0$) và quy ước nhãn hai lớp thành $\{-1, +1\}$.
2. **Tính khoảng cách lề:** Với mỗi mẫu dữ liệu, tính giá trị lề hình học: $\text{margin} = y \cdot (w \cdot x + b)$.
3. **Phát hiện điểm vi phạm (Support Vectors):**
   * Nếu $\text{margin} \ge 1$: Mẫu dữ liệu đã nằm an toàn đúng phía $\rightarrow$ Không phạt (Hinge Loss = 0).
   * Nếu $\text{margin} < 1$: Mẫu dữ liệu bị lấn vào vùng an toàn hoặc đoán sai $\rightarrow$ Bị phạt theo mức độ vi phạm: $1 - \text{margin}$.
4. **Cập nhật trọng số:** Điều chỉnh hướng nghiêng của siêu phẳng để sửa sai cho các điểm vi phạm lề, đồng thời dùng phanh L2 kéo độ mở của lề rộng ra.
5. **Dự đoán:** Khi có mẫu mới, tính giá trị $z = w \cdot x + b$. Nếu $z \ge 0$ thì gán nhãn 1 (hoặc $+1$), nếu $z < 0$ thì gán nhãn 0 (hoặc $-1$).

### 5. Data used (Dữ liệu sử dụng)
* **Dạng dữ liệu phù hợp:** Dữ liệu dạng bảng số, ma trận đặc trưng có số chiều lớn (như ma trận TF-IDF văn bản, vector đặc trưng ảnh, dữ liệu sinh học).
* **Yêu cầu tiền xử lý:** **Bắt buộc phải chuẩn hóa thang đo (Scaling)** vì SVM tính toán khoảng cách hình học, nếu các biến có thang đo lệch nhau thì khoảng lề sẽ bị méo.
* **Đặc điểm tương thích:** Hoạt động cực kỳ ổn định trong không gian nhiều chiều, ít bị quá khớp (overfitting) do vị trí của siêu phẳng chỉ phụ thuộc vào một lượng nhỏ các điểm tựa (Support Vectors).
* **Ví dụ ứng dụng thực tế:** Nhận dạng chữ viết tay/hình ảnh, phân loại tài liệu văn bản, phân loại thư rác (Spam/Ham), chẩn đoán y sinh học.
