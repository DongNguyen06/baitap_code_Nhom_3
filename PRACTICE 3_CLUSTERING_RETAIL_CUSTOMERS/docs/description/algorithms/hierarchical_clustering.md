# Hierarchical Agglomerative Clustering (Phân cụm Thứ bậc Tích tụ)

### 1. Mục tiêu
* Giải quyết bài toán **Học không giám sát (Unsupervised Learning) — Phân cụm Phân cấp (Hierarchical Clustering)**: Xây dựng cấu trúc cây gom cụm lồng ghép nhiều cấp độ (nested hierarchy) theo chiến lược từ dưới lên (Bottom-up / Agglomerative) từ $N$ cụm đơn lẻ ban đầu gộp dần thành 1 cụm duy nhất.
* Cung cấp **ma trận liên kết (Linkage Matrix)** chuẩn xác để trực quan hóa biểu đồ cây **Dendrogram**, cho phép người phân tích quan sát toàn diện khoảng cách tương đồng giữa các nhóm khách hàng và linh hoạt cắt cây ở số lượng cụm $K$ tối ưu mà không cần huấn luyện lại từ đầu.

### 2. Input / Output
* **Đầu vào (Input):**
  * Ma trận đặc trưng số $X \in \mathbb{R}^{N \times D}$ (các đặc trưng đã được chuẩn hóa qua StandardScaler).
  * Số lượng cụm mục tiêu $K$ cần cắt lát ($1 \le K \le N$).
  * Tiêu chí liên kết `linkage`: `'ward'` (cực tiểu hóa phương sai nội cụm), `'complete'` (khoảng cách lớn nhất giữa hai điểm), hoặc `'average'` (khoảng cách trung bình giữa mọi cặp điểm).
* **Đầu ra (Output):**
  * Mảng nhãn phân cụm sau khi cắt cây: $\text{labels} \in \{0, 1, \dots, K-1\}^N$.
  * Ma trận liên kết $\text{linkage\_matrix\_} \in \mathbb{R}^{(N-1) \times 4}$ tương thích chuẩn `scipy.cluster.hierarchy.dendrogram`, trong đó mỗi hàng $i$ đại diện cho một bước gộp:
    $$Z[i] = [c_1, c_2, \text{dist}, \text{cluster\_size}]$$
  * Danh sách các bước gộp con: $\text{children\_} \in \mathbb{N}^{(N-1) \times 2}$ và khoảng cách gộp $\text{distances\_} \in \mathbb{R}^{N-1}$.

### 3. Evaluation Metrics / Criteria (Tiêu chí & Độ đo)
* **Khoảng cách liên kết Ward (Ward's Linkage Distance):**
  * Tiêu chí Ward tìm cặp cụm $u$ và $v$ sao cho khi gộp lại làm tăng tổng bình phương sai số nội cụm (Error Sum of Squares — $\Delta \text{ESS}$) là nhỏ nhất:
    $$\Delta \text{ESS}(u, v) = \frac{|u| \cdot |v|}{|u| + |v|} ||\mu_u - \mu_v||_2^2$$
  * Khoảng cách liên kết Ward tương thích chuẩn Scipy/Scikit-Learn:
    $$d(u, v) = \sqrt{\frac{2 |u| \cdot |v|}{|u| + |v|}} ||\mu_u - \mu_v||_2$$
    Trong đó $\mu_u, \mu_v$ lần lượt là trọng tâm và $|u|, |v|$ là số phần tử của hai cụm.
* **Độ đo đánh giá phân cụm thứ bậc:**
  * **Biểu đồ cây phân cấp (Dendrogram):** Trục hoành biểu diễn các mẫu khách hàng; trục tung biểu diễn khoảng cách liên kết $d(u, v)$ tại thời điểm gộp. Vị trí cắt ngang cây lý tưởng là nơi có khoảng cách dọc lớn nhất mà không bị cắt bởi bất kỳ đường liên kết ngang nào.
  * **Hệ số tương quan Cophenetic (Cophenetic Correlation Coefficient):** Đo lường mức độ bảo toàn khoảng cách gốc giữa các cặp điểm sau khi chuyển thành khoảng cách cophenetic trên cây Dendrogram (càng gần 1 càng tốt).
  * **Silhouette Score:** Đánh giá độ phân tách và độ kết dính cụm tại lát cắt $K$.

### 4. How it works (Cách hoạt động)
*(Tóm lược: Khởi tạo mỗi điểm dữ liệu là một cụm riêng rẽ, sau đó thực hiện tuần tự $N-1$ bước gộp cặp cụm có khoảng cách liên kết nhỏ nhất và ghi nhận vào ma trận liên kết Linkage Matrix).*

1. **Khởi tạo trạng thái ban đầu:**
   * Ban đầu có $N$ cụm riêng lẻ, mỗi cụm chứa đúng 1 điểm dữ liệu với ID từ $0$ đến $N-1$.
   * Trọng tâm cụm $\mu_i = x_i$ và kích thước cụm $|C_i| = 1$.
2. **Vòng lặp tích tụ từng bước ($N-1$ bước gộp):**
   * Tại mỗi bước $t \in \{0, 1, \dots, N-2\}$:
     * Đo khoảng cách liên kết Ward $d(u, v)$ giữa tất cả các cặp cụm đang hoạt động (active clusters).
     * Tìm cặp cụm $(u^*, v^*)$ có khoảng cách liên kết nhỏ nhất:
       $$(u^*, v^*) = \arg\min_{u < v} d(u, v)$$
     * Tạo một cụm mới gộp từ $u^*$ và $v^*$ mang chỉ số mới là $N + t$.
     * Kích thước cụm mới: $|C_{\text{new}}| = |C_{u^*}| + |C_{v^*}|$.
     * Trọng tâm cụm mới: $\mu_{\text{new}} = \frac{|C_{u^*}| \mu_{u^*} + |C_{v^*}| \mu_{v^*}}{|C_{u^*}| + |C_{v^*}|}$.
     * Ghi nhận thông tin bước gộp vào dòng thứ $t$ của ma trận liên kết $Z$:
       $$Z[t] = [u^*, v^*, d(u^*, v^*), |C_{\text{new}}|]$$
     * Hủy kích hoạt hai cụm cũ $u^*, v^*$ và kích hoạt cụm mới $N + t$.
3. **Cắt cây phân cấp (Tree Cut để lấy $K$ cụm):**
   * Quá trình gộp dừng lại tại bước $t = N - K$ khi số lượng cụm đang hoạt động đúng bằng $K$.
   * Các điểm dữ liệu thuộc cùng một cụm hoạt động tại thời điểm này sẽ được gán cùng một nhãn phân cụm từ $0$ đến $K-1$.

### 5. Dữ liệu sử dụng
* **Dạng dữ liệu phù hợp:** Dữ liệu dạng bảng số có quy mô vừa và nhỏ ($N \le 5.000$ mẫu) do độ phức tạp thời gian tính toán là $O(N^2 \log N)$ hoặc $O(N^3)$. Với tập `Mall_Customers.csv` gồm $N=200$ dòng, thuật toán chạy tức thời trong vài chục mili-giây và vẽ được biểu đồ Dendrogram cực kỳ sắc nét.
* **Yêu cầu tiền xử lý:**
  * Bắt buộc phải chuẩn hóa thang đo bằng `StandardScaler` vì tiêu chí Ward tính toán trên khoảng cách Euclidean giữa các trọng tâm.
