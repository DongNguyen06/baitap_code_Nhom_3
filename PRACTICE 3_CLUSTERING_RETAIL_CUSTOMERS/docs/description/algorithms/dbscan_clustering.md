# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

### 1. Mục tiêu
* Giải quyết bài toán **Học không giám sát (Unsupervised Learning) — Phân cụm dựa trên mật độ (Density-based Clustering)**: Tự động phát hiện các cụm dữ liệu có mật độ dày đặc, bao quanh bởi các vùng có mật độ thưa thớt mà không cần biết trước số lượng cụm $K$.
* Tự động nhận diện và cô lập các điểm dữ liệu dị biệt, ngoại lai hoặc hành vi khách hàng bất thường dưới dạng **điểm nhiễu (Noise points / Outliers)** gắn nhãn $-1$.
* Khắc phục hoàn toàn nhược điểm của K-Means: Có thể phát hiện các cụm có hình dạng phi lồi (non-convex, hình trăng khuyết, hình xoắn ốc) và không bị ép buộc mọi điểm dữ liệu vào một cụm cụ thể.

### 2. Input / Output
* **Đầu vào (Input):**
  * Ma trận đặc trưng số $X \in \mathbb{R}^{N \times D}$ (đã được chuẩn hóa qua StandardScaler).
  * Bán kính lân cận $\varepsilon$ (`eps` $> 0$): Bán kính hình cầu xung quanh một điểm để xác định mật độ cục bộ.
  * Số lượng điểm lân cận tối thiểu `min_samples` ($\ge 1$): Ngưỡng số điểm tối thiểu nằm trong bán kính $\varepsilon$ để một điểm được coi là điểm lõi (Core point).
* **Đầu ra (Output):**
  * Mảng nhãn phân cụm: $\text{labels} \in \{-1, 0, 1, \dots, C-1\}^N$, trong đó $-1$ đại diện cho điểm nhiễu (noise) và $C$ là số lượng cụm tự động phát hiện được.
  * Chỉ số các điểm lõi: $\text{core\_sample\_indices\_} \in \mathbb{N}^M$.
  * Tọa độ các điểm lõi trong không gian đặc trưng: $\text{components\_} \in \mathbb{R}^{M \times D}$.

### 3. Evaluation Metrics / Criteria (Tiêu chí & Độ đo)
* **Khái niệm mật độ cốt lõi:**
  * **$\varepsilon$-Lân cận ($N_\varepsilon(p)$):** Tập hợp tất cả các điểm có khoảng cách Euclidean tới $p$ không vượt quá $\varepsilon$:
    $$N_\varepsilon(p) = \{q \in X \mid ||p - q||_2 \le \varepsilon\}$$
  * **Điểm lõi (Core Point):** Điểm $p$ có số điểm lân cận thỏa mãn $|N_\varepsilon(p)| \ge \text{min\_samples}$.
  * **Điểm biên (Border Point):** Điểm $q$ không phải điểm lõi ($|N_\varepsilon(q)| < \text{min\_samples}$) nhưng nằm trong $\varepsilon$-lân cận của một điểm lõi $p$.
  * **Điểm nhiễu (Noise / Outlier):** Điểm không phải điểm lõi và cũng không thuộc lân cận của bất kỳ điểm lõi nào (gán nhãn $-1$).
* **Độ đo đánh giá chất lượng phân cụm mật độ:**
  * **Silhouette Score:** Tính toán trên tập các điểm đã được phân cụm hợp lệ (bỏ qua điểm nhiễu $-1$ hoặc đánh giá tổng thể):
    $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  * **Tỷ lệ điểm nhiễu (Noise Ratio):** Tỷ lệ phần trăm số mẫu bị gán nhãn $-1$: $R_{\text{noise}} = \frac{N_{\text{noise}}}{N} \times 100\%$.
  * **Biểu đồ k-Distance Graph (k-NN Elbow):** Kỹ thuật tìm tham số $\varepsilon$ tối ưu bằng cách sắp xếp khoảng cách từ mỗi điểm tới láng giềng thứ $k$ (`k = min_samples - 1`) tăng dần và tìm điểm uốn (khuỷu tay).

### 4. How it works (Cách hoạt động)
*(Tóm lược: Xác định toàn bộ các điểm lõi dựa trên bán kính $\varepsilon$ và số láng giềng tối thiểu, sau đó sử dụng thuật toán duyệt theo chiều rộng BFS để lan truyền và mở rộng từng cụm mật độ liên thông).*

1. **Tính toán ma trận khoảng cách:**
   * Tính ma trận khoảng cách Euclidean giữa mọi cặp điểm dữ liệu: $D_{ij} = ||x_i - x_j||_2$.
2. **Xác định $\varepsilon$-lân cận & Điểm lõi:**
   * Với mỗi điểm $i$, tìm danh sách láng giềng: $N_\varepsilon(i) = \{j \mid D_{ij} \le \varepsilon\}$.
   * Nếu $|N_\varepsilon(i)| \ge \text{min\_samples}$, đánh dấu $i$ là Điểm lõi (Core point).
3. **Mở rộng cụm (Cluster Expansion qua hàng đợi BFS):**
   * Khởi tạo mảng nhãn toàn bộ bằng $-1$ (chưa phân cụm / nhiễu).
   * Đặt biến đếm cụm hiện tại `cluster_id = 0`.
   * Lặp qua từng điểm lõi $i$:
     * Nếu điểm $i$ chưa được gán cụm (nhãn là $-1$):
       * Gán nhãn cho $i$: $\text{labels}[i] = \text{cluster\_id}$.
       * Đưa toàn bộ lân cận $N_\varepsilon(i)$ vào hàng đợi mở rộng (Queue / FIFO).
       * Trong khi hàng đợi chưa rỗng:
         * Lấy điểm $q$ ra khỏi hàng đợi.
         * Nếu $q$ trước đó bị coi là nhiễu, gán lại thành phần của cụm hiện tại: $\text{labels}[q] = \text{cluster\_id}$.
         * Nếu $q$ chưa từng được xét (chưa thuộc cụm nào):
           * Gán nhãn: $\text{labels}[q] = \text{cluster\_id}$.
           * Nếu $q$ cũng là một điểm lõi ($|N_\varepsilon(q)| \ge \text{min\_samples}$), nạp toàn bộ các láng giềng của $q$ vào hàng đợi mở rộng.
       * Tăng `cluster_id += 1` để sẵn sàng cho cụm mật độ tiếp theo.
4. **Xử lý điểm còn lại:**
   * Mọi điểm không thể vươn tới (not density-reachable) từ bất kỳ điểm lõi nào sẽ giữ nguyên nhãn $-1$ (Noise).

### 5. Dữ liệu sử dụng
* **Dạng dữ liệu phù hợp:** Dữ liệu số liên tục dạng bảng có cụm mật độ đồng đều, có chứa nhiễu hoặc có hình dạng cụm phức tạp không thể tách biệt bằng siêu cầu của K-Means.
* **Yêu cầu tiền xử lý:**
  * **Cực kỳ nhạy cảm với thang đo:** Bắt buộc chuẩn hóa dữ liệu bằng `StandardScaler` để các trục tọa độ có cùng phương sai đơn vị $\sigma = 1$, giúp bán kính cầu $\varepsilon$ có ý nghĩa đồng nhất trên mọi chiều đặc trưng.
  * Cần tinh chỉnh bộ đôi tham số $(\varepsilon, \text{min\_samples})$ phù hợp: $\varepsilon$ quá nhỏ dẫn tới hầu hết dữ liệu bị coi là nhiễu; $\varepsilon$ quá lớn sẽ gom toàn bộ dữ liệu vào một cụm duy nhất.
