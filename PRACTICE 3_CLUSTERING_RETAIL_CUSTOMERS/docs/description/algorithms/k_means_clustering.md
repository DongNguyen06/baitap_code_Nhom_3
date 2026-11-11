# K-Means Clustering (Phân cụm Dựa trên Tâm cụm — Centroid-based)

### 1. Mục tiêu
* Giải quyết bài toán **Học không giám sát (Unsupervised Learning) — Phân cụm (Clustering)**: Tự động gom $N$ khách hàng vào $K$ nhóm đặc trưng (customer segments) riêng biệt dựa trên độ tương đồng về độ tuổi, thu nhập và thói quen chi tiêu mà không cần nhãn định sẵn.
* Tối ưu hóa tính thuần nhất bên trong mỗi cụm (intra-cluster cohesion) bằng cách cực tiểu hóa tổng bình phương khoảng cách từ mỗi điểm dữ liệu đến trọng tâm cụm tương ứng (WCSS / Inertia).

### 2. Input / Output
* **Đầu vào (Input):**
  * Ma trận đặc trưng số $X \in \mathbb{R}^{N \times D}$ (các đặc trưng đã được chuẩn hóa qua StandardScaler: Age, Annual Income, Spending Score).
  * Siêu tham số số cụm $K \ge 2$, số bước lặp tối đa `max_iter`, ngưỡng dung sai hội tụ `tol`, và phương thức khởi tạo (`k-means++` hoặc `random`).
* **Đầu ra (Output):**
  * Mảng nhãn phân cụm của từng mẫu: $\text{labels} \in \{0, 1, \dots, K-1\}^N$.
  * Tọa độ $K$ trọng tâm cụm trong không gian đặc trưng: $\text{cluster\_centers\_} \in \mathbb{R}^{K \times D}$.
  * Tổng bình phương sai số nội cụm (Inertia / WCSS): $J = \sum_{i=1}^N \min_{k} ||x_i - \mu_k||^2$.

### 3. Evaluation Metrics / Criteria (Tiêu chí & Độ đo)
* **Hàm mất mát tối ưu (Objective Function / Inertia):**
  $$\mathcal{J}(C, \mu) = \sum_{k=1}^K \sum_{x_i \in C_k} ||x_i - \mu_k||_2^2$$
  Trong đó $\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$ là trọng tâm (centroid) của cụm $C_k$.
* **Độ đo đánh giá chất lượng phân cụm:**
  * **Elbow Method (Đường cong khuỷu tay):** Quan sát độ suy giảm của Inertia khi tăng $K$ từ 2 đến 10 để tìm điểm uốn (khuỷu tay) tối ưu.
  * **Silhouette Score:** Đánh giá độ tách biệt giữa các cụm và độ kết dính nội cụm (giá trị thuộc $[-1, 1]$, càng gần 1 càng phân cụm rõ nét):
    $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  * **Davies-Bouldin Index:** Tỷ số độ phân tán nội cụm trên khoảng cách giữa các tâm cụm (càng nhỏ càng tốt).
  * **Calinski-Harabasz Index:** Tỷ số phương sai giữa các cụm trên phương sai nội cụm (càng lớn càng tốt).

### 4. How it works (Cách hoạt động)
*(Tóm lược: Thuật toán lặp luân phiên 2 bước Expectation-Maximization gồm Gán nhãn gần nhất và Tái tính toán trọng tâm đến khi hội tụ).*

1. **Khởi tạo trọng tâm (K-Means++ Initialization):**
   * Chọn ngẫu nhiên tâm cụm đầu tiên $\mu_1$ từ tập dữ liệu.
   * Với mỗi tâm tiếp theo $\mu_k$ ($k=2,\dots,K$), chọn một điểm dữ liệu $x$ với xác suất tỷ lệ thuận với bình phương khoảng cách đến tâm gần nhất đã chọn $D(x)^2$:
     $$P(x) = \frac{D(x)^2}{\sum_{x' \in X} D(x')^2}$$
   * Cơ chế K-Means++ giúp các tâm ban đầu trải rộng đều trong không gian dữ liệu, hạn chế rơi vào nghiệm cực tiểu địa phương xấu.
2. **Bước gán nhãn (Assignment Step — Expectation):**
   * Với mỗi điểm $x_i$, đo khoảng cách Euclidean tới tất cả $K$ trọng tâm:
     $$c_i = \arg\min_{k \in \{1,\dots,K\}} ||x_i - \mu_k||_2^2$$
3. **Bước cập nhật trọng tâm (Update Step — Maximization):**
   * Tính lại tọa độ tâm cụm bằng trung bình cộng tọa độ của các điểm thuộc cụm đó:
     $$\mu_k^{(t+1)} = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$
   * Xử lý ngoại lệ cụm rỗng: Nếu một cụm không có điểm nào rơi vào, gán lại tâm đó tại điểm dữ liệu có khoảng cách xa nhất so với tâm hiện tại của nó.
4. **Kiểm tra điều kiện dừng (Convergence Check):**
   * Dừng lặp khi độ dịch chuyển của các trọng tâm nhỏ hơn ngưỡng dung sai:
     $$\sum_{k=1}^K ||\mu_k^{(t+1)} - \mu_k^{(t)}||_2 < \text{tol}$$
   * Hoặc khi số bước lặp đạt `max_iter` (mặc định 300).
5. **Dự đoán mẫu mới:** Điểm dữ liệu mới được gán vào cụm có khoảng cách Euclidean tới tâm cụm là nhỏ nhất.

### 5. Dữ liệu sử dụng
* **Dạng dữ liệu phù hợp:** Dữ liệu số liên tục dạng bảng (numerical tabular data), phân phối cụm dạng cầu lồi (spherical/convex clusters).
* **Yêu cầu tiền xử lý:** 
  * Bắt buộc phải chuẩn hóa thang đo (StandardScaler / MinMaxScaler) vì khoảng cách Euclidean rất nhạy cảm với các biến có biên độ lớn (ví dụ `Annual Income` lên tới hàng chục nghìn trong khi `Age` chỉ từ 18-70).
  * Cần loại bỏ các đặc trưng ID vô nghĩa (`CustomerID`) và mã hóa đặc trưng phân loại (`Gender`).
