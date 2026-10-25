# Logistic Regression (Hồi quy Logistic)

## 1. Name
- **Tên:** Logistic Regression (Phân loại email Spam/Ham theo xác suất).

## 2. Problem
- Phân loại email là **Spam (1)** hay **Ham (0)**. Cần độ chính xác cao và tránh tối đa việc chặn nhầm email công việc (False Positive).

## 3. Target
- Dự đoán xác suất rủi ro $P(\text{Spam}) \in [0, 1]$ để người dùng tự chọn ngưỡng cảnh báo.
- Làm mô hình cơ sở (baseline) so sánh đối đầu với SVM và Naive Bayes.

## 4. Input / Output
| Thành phần | Dữ liệu | Mô tả |
|---|---|---|
| **Input** | `text_clean` | Văn bản email sạch $\rightarrow$ Vector TF-IDF |
| **Output** | `label`, `prob` | Nhãn `1` (Spam) / `0` (Ham) và xác suất $P \in [0, 1]$ |

## 5. Core features
- **Ra xác suất:** Cho ra xác suất (probability) thay vì chỉ nhãn cứng.
- **Điều chuẩn L2:** Chống học vẹt (overfitting), kiểm soát độ lớn trọng số từ vựng.
- **Dễ giải thích:** Trọng số $w > 0$ là dấu hiệu thư rác, $w < 0$ là thư công việc.
- **Tối ưu Mini-batch:** Huấn luyện nhanh, xử lý mượt mà ma trận thưa từ TF-IDF.

## 6. Technology
- **Python 3**, **NumPy** (tự cài đặt thuật toán), **Pandas** (xử lý dữ liệu), **Scikit-learn** (`TfidfVectorizer`, `train_test_split`).

## 7. Evaluation
- **Accuracy:** Tỷ lệ đoán đúng toàn bộ email.
- **Precision:** Tránh báo động nhầm (tránh chặn nhầm thư công việc).
- **Recall:** Tránh bỏ sót (tránh để lọt thư rác vào hòm thư đến).
- **F1-Score:** Điểm cân bằng tối ưu giữa Precision và Recall.
- **Confusion Matrix:** Bảng đối soát chi tiết [TN, FP, FN, TP].

## 8. How it work
1. **Vector hóa:** Chuyển văn bản thành vector đặc trưng TF-IDF $\mathbf{x}$.
2. **Điểm tuyến tính:** $z = \mathbf{w}^T \mathbf{x} + b$.
3. **Hàm Sigmoid:** $p = \frac{1}{1 + e^{-z}}$ (ép điểm $z$ về khoảng xác suất từ 0 đến 1).
4. **Hàm mất mát:** Log-Loss + L2 để đo sai số giữa nhãn thực và dự đoán.
5. **Gradient Descent:** Cập nhật trọng số theo hướng giảm sai số: $\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \text{grad}$.
6. **Phân loại:** Nếu $p \ge 0.5 \rightarrow$ **Spam**, ngược lại $\rightarrow$ **Ham**.

## 9. Data used
- **Dataset:** [Phishing & Spam Email Dataset (CEAS_08.csv trên Kaggle)](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=CEAS_08.csv).
- **Dữ liệu thực tế:** `clean_emails.csv` (đã lọc rác HTML, khử trùng lặp nội dung).
- **Chia tập:** 80% Train / 20% Test (phân tầng Stratified).

## 10. Demo
```bash
python scripts/algorithms/logistic_regression.py --data data/processed/clean_emails.csv
```
Khung kết quả mẫu:
```text
Epoch  30/30 - loss: 0.xxxx
=== KẾT QUẢ LOGISTIC REGRESSION ===
  accuracy: 0.xxxx | precision: 0.xxxx | recall: 0.xxxx | f1: 0.xxxx
Confusion matrix [[TN, FP], [FN, TP]]: [[...], [...]]
Email mẫu: "URGENT: Verify account now" -> Xác suất: 0.xxxx -> SPAM
```