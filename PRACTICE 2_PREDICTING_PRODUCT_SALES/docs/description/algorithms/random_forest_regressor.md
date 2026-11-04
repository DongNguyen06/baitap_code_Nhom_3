# Random Forest Regressor (Rừng Ngẫu nhiên Hồi quy)

### 1. Mục tiêu
* Giải quyết bài toán **hồi quy dữ liệu bảng phức tạp**, khắc phục triệt để nhược điểm bất ổn định và dễ học vẹt của một cây quyết định đơn lẻ.
* Áp dụng **học kết hợp (Ensemble Learning - Bagging)**: Bằng cách tổng hợp ý kiến từ hàng chục cây quyết định mọc độc lập trên các góc nhìn dữ liệu ngẫu nhiên khác nhau, mô hình triệt tiêu phương sai (variance), mang lại kết quả dự đoán cực kỳ ổn định và chính xác.

### 2. Input / Output
* **Đầu vào (Input):** Bảng dữ liệu chứa các vector đặc trưng số $x = (x_1, x_2, \dots, x_n)$ và giá trị mục tiêu liên tục $y \in \mathbb{R}$.
* **Đầu ra (Output):** Con số dự đoán liên tục $\hat{y}$ (là giá trị trung bình cộng dự đoán của toàn bộ các cây trong rừng).
* **Tham số mô hình học:** Tập hợp $B$ cây quyết định độc lập $\{T_1, T_2, \dots, T_B\}$ cùng cấu trúc rẽ nhánh riêng biệt của từng cây.

### 3. Evaluation Metrics / Criteria (Tiêu chí & Độ đo)
* **Tiêu chí tối ưu:** 
  * Tối ưu hóa sai số MSE trên từng cây kết hợp cơ chế giảm phương sai tổng thể của kỹ thuật Bagging:
    $$\text{Var}_{\text{forest}} = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
* **Độ đo đánh giá hiệu năng:**
  * **R2-Score ($R^2$):** Đánh giá mức độ giải thích dữ liệu của cả khu rừng trên tập kiểm tra.
  * **RMSE & MAE:** Đo lường độ lệch sai số dự đoán thực tế so với nhãn thật.
  * **Feature Importances:** Trọng số đóng góp của từng đặc trưng vào việc giảm độ bất định trong các cây.

### 4. How it works (Cách hoạt động)
*(Tóm lược: Tạo ra nhiều cây quyết định từ các tập dữ liệu con ngẫu nhiên và tập thuộc tính ngẫu nhiên, sau đó lấy trung bình cộng kết quả của tất cả các cây để ra quyết định cuối cùng bền vững nhất).*

1. **Lấy mẫu có hoàn lại (Bootstrap Sampling):** Từ tập dữ liệu $N$ mẫu ban đầu, thực hiện bốc thăm ngẫu nhiên có hoàn lại để tạo ra $B$ tập dữ liệu con.
2. **Chọn ngẫu nhiên tập thuộc tính (Random Subspace):** Tại mỗi nút phân nhánh của từng cây, thuật toán chỉ chọn ngẫu nhiên một nhóm nhỏ các thuộc tính (thường là $\sqrt{d}$ hoặc $d/3$) để tìm nhát cắt.
3. **Huấn luyện độc lập:** Xây dựng đồng thời các cây quyết định hoàn toàn độc lập với nhau.
4. **Tổng hợp kết quả (Aggregation):** Khi có một mẫu mới cần dự đoán, cho mẫu chạy qua toàn bộ $B$ cây trong rừng để thu được $B$ kết quả dự đoán riêng lẻ. Con số dự đoán cuối cùng là trung bình cộng:
   $$\hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(x)$$

### 5. Dữ liệu sử dụng
* **Dạng dữ liệu phù hợp:** Dữ liệu dạng bảng số có nhiều thuộc tính phi tuyến, dữ liệu có hiện tượng tương tác phức tạp hoặc có lẫn nhiều nhiễu.
* **Yêu cầu tiền xử lý:** Không cần chuẩn hóa thang đo; tự động xếp hạng độ quan trọng của các đặc trưng (`feature_importances_`).
