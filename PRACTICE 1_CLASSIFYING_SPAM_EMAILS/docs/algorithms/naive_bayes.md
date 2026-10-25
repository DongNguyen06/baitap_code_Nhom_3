## 1. Name
**Multinomial Naive Bayes** – Phân loại email Spam/Ham bằng xác suất Bayes đa thức.
## 2. Problem
Phân loại nhị phân email Spam (1) và Ham (0) dựa trên tần suất xuất hiện của từ ngữ. Cần bộ lọc cực nhanh, tính toán nhẹ và xử lý tốt dữ liệu văn bản.
## 3. Target
- Dự đoán nhãn có xác suất hậu nghiệm lớn nhất $\text{argmax}_c P(c \mid \mathbf{d})$.
- Làm mô hình đối chuẩn (baseline) kinh điển để so sánh với Logistic Regression và SVM.
## 4. Input / Output
| Thành phần | Dữ liệu | Mô tả |
|---|---|---|
| **Input** | `text_clean` | Văn bản email sạch $\rightarrow$ Ma trận đếm từ / TF-IDF |
| **Output** | `label`, `prob` | Nhãn `1` (Spam) / `0` (Ham) và phân phối xác suất 2 lớp |
## 5. Core features
- **Huấn luyện một lượt:** Chỉ đếm tần suất xuất hiện của từ vựng, không cần lặp Gradient Descent.
- **Làm mịn Laplace ($\alpha$):** Tránh lỗi xác suất bằng 0 khi gặp từ vựng mới chưa từng thấy.
- **Log-Likelihood:** Tính toán trong không gian Log để chống tràn số dưới (numerical underflow).
- **Giả định độc lập:** Giả định các từ xuất hiện độc lập khi đã biết lớp của email.
## 6. Technology
- **Python 3.10+**, **NumPy** (cài đặt thuật toán từ đầu), **Pandas** (xử lý dữ liệu), **Scikit-learn** (`TfidfVectorizer`, `train_test_split`).
## 7. Evaluation
Đánh giá trên tập kiểm thử độc lập (Test Set) qua các thước đo:
- **Accuracy:** Tỷ lệ phân loại chính xác toàn diện trên toàn bộ email.
- **Precision:** Tránh báo động nhầm (giữ hộp thư công việc an toàn).
- **Recall:** Tránh bỏ sót thư rác (thường bị ảnh hưởng do các từ khóa tương quan).
- **F1-Score:** Điểm cân bằng điều hòa giữa Precision và Recall.
- **Confusion Matrix:** Bảng đối soát chi tiết [TN, FP, FN, TP].
## 8. How it work
1. **Vector hóa:** Đếm tần suất xuất hiện của các từ trong email thành vector $\mathbf{x}$.
2. **Định lý Bayes:** $P(c \mid \mathbf{d}) \propto P(c) \cdot \prod_i P(w_i \mid c)^{x_i}$.
3. **Xác suất tiên nghiệm:** $P(c) = \frac{\text{Số email lớp } c}{\text{Tổng số email}}$.
4. **Xác suất có điều kiện (Laplace):** $P(w_i \mid c) = \frac{\text{count}(w_i, c) + \alpha}{\sum \text{count}(w, c) + \alpha \cdot V}$.
5. **Dự đoán qua Log:** $\hat{c} = \text{argmax}_c \left[ \ln P(c) + \sum x_i \ln P(w_i \mid c) \right]$.
6. **Chuẩn hóa xác suất:** Áp dụng Softmax trên Log-Likelihood để xuất xác suất $P \in [0, 1]$.
## 9. Data used
- **Dataset:** [Phishing & Spam Email Dataset (CEAS_08.csv trên Kaggle)](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=CEAS_08.csv).
- **Dữ liệu thực tế:** `clean_emails.csv` (đã lọc rác HTML, khử trùng lặp nội dung).
- **Chia tập:** 80% Train / 20% Test (phân tầng Stratified).
## 10. Demo
Lệnh thực thi:
```bash
python scripts/algorithms/naive_bayes.py --data data/processed/clean_emails.csv
```
Khung kết quả mẫu:
```text
=== KẾT QUẢ MULTINOMIAL NAIVE BAYES ===
  accuracy: 0.xxxx | precision: 0.xxxx | recall: 0.xxxx | f1: 0.xxxx
Confusion matrix [[TN, FP], [FN, TP]]: [[...], [...]]
Email mẫu: "URGENT: Verify account now" -> P(Spam): 0.xxxx -> SPAM
```
