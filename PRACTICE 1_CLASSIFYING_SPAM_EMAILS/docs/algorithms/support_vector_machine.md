## 1. Name
**Linear Support Vector Machine (Linear SVM)** – Phân loại email Spam/Ham bằng siêu phẳng lề cực đại.
## 2. Problem
Phân loại nhị phân email Spam (1) và Ham (0) trên không gian văn bản số chiều rất lớn. Cần khả năng tổng quát hóa cao và hạn chế tối đa việc bắt nhầm thư công việc (False Positive).
## 3. Target
- Tìm siêu phẳng tối ưu $\mathbf{w}^T \mathbf{x} + b = 0$ phân tách hai lớp với độ rộng lề (margin) lớn nhất $2 / \|\mathbf{w}\|$.
- Đạt hiệu năng phân loại cao nhất và kiểm soát nghiêm ngặt lỗi bắt nhầm thư thường.
## 4. Input / Output
| Thành phần | Dữ liệu | Mô tả |
|---|---|---|
| **Input** | `text_clean` | Văn bản email sạch $\rightarrow$ Vector TF-IDF số chiều cao |
| **Output** | `label`, `score` | Điểm lề $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$ và nhãn `1` (Spam) / `0` (Ham) |
## 5. Core features
- **Tối đa hóa lề (Max-Margin):** Tối ưu khoảng cách an toàn giữa ranh giới và các mẫu dữ liệu khó nhất.
- **Hàm mất mát Hinge Loss:** Chỉ phạt các điểm nằm trong vùng lề hoặc phân loại sai (Support Vectors).
- **Điều chuẩn L2:** Kiểm soát độ phẳng của siêu phẳng, đặc biệt phù hợp dữ liệu văn bản thưa.
- **Tối ưu Subgradient Descent:** Huấn luyện nhanh trên dữ liệu lớn, hội tụ ổn định.
## 6. Technology
- **Python 3.10+**, **NumPy** (cài đặt Subgradient Descent từ đầu), **Pandas** (xử lý dữ liệu), **Scikit-learn** (`LinearSVC`, `train_test_split`).
## 7. Evaluation
Đánh giá trên tập kiểm thử độc lập (Test Set) qua các thước đo:
- **Accuracy:** Tỷ lệ dự đoán chính xác toàn diện trên tập kiểm thử.
- **Precision:** Tránh bắt nhầm email quan trọng (chỉ tiêu sống còn của SVM).
- **Recall:** Khả năng phát hiện triệt để các hình thức thư rác tinh vi.
- **F1-Score:** Thước đo tổng hợp thể hiện đẳng cấp phân loại của siêu phẳng.
- **Confusion Matrix:** Bảng chi tiết [TN, FP, FN, TP].
## 8. How it work
1. **Vector hóa & Đổi nhãn:** Chuyển văn bản thành vector TF-IDF $\mathbf{x}$; đổi nhãn $\{0, 1\} \to \{-1, +1\}$.
2. **Siêu phẳng quyết định:** $\mathbf{w}^T \mathbf{x} + b = 0$ với vùng đệm an toàn $\mathbf{w}^T \mathbf{x} + b = \pm 1$.
3. **Mục tiêu Soft-Margin:** Tối thiểu hóa $J(\mathbf{w}, b) = \frac{\lambda}{2} \|\mathbf{w}\|^2 + \frac{1}{n} \sum \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$.
4. **Nhận diện điểm vi phạm:** Mẫu $i$ vi phạm lề khi $y_i(\mathbf{w}^T \mathbf{x}_i + b) < 1$.
5. **Cập nhật Subgradient:** Cập nhật trọng số $\mathbf{w}$ theo hướng giảm thiểu phạt Hinge Loss và điều chuẩn L2.
6. **Quyết định:** Nếu $\mathbf{w}^T \mathbf{x} + b \ge 0 \rightarrow$ **Spam (1)**, ngược lại $\rightarrow$ **Ham (0)**.
## 9. Data used
- **Dataset:** [Phishing & Spam Email Dataset (CEAS_08.csv trên Kaggle)](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=CEAS_08.csv).
- **Dữ liệu thực tế:** `clean_emails.csv` (đã lọc rác HTML, khử trùng lặp nội dung).
- **Chia tập:** 80% Train / 20% Test (phân tầng Stratified).
## 10. Demo
Lệnh thực thi:
```bash
python scripts/algorithms/support_vector_machine.py --data data/processed/clean_emails.csv
```
Khung kết quả mẫu:
```text
Epoch  30/30 - loss: 0.xxxx
=== KẾT QUẢ LINEAR SVM ===
  accuracy: 0.xxxx | precision: 0.xxxx | recall: 0.xxxx | f1: 0.xxxx
Confusion matrix [[TN, FP], [FN, TP]]: [[...], [...]]
Email mẫu: "URGENT: Verify account now" -> Điểm lề: +x.xxxx -> SPAM
```
