# BẢN ĐẶC TẢ DỰ ÁN (PROJECT DEFINITION) — SPAMGUARD-ML
## PRACTICE 1: CLASSIFYING SPAM EMAILS — NHÓM 3 (UTH)
> **Giảng viên hướng dẫn:** Tiến sĩ Nguyễn Thị Khánh Tiên  
> **Tập dữ liệu nguồn:** `data/raw/CEAS_08.csv` (39.154 email thực tế từ hội thảo CEAS)

---

## 1. Tên dự án (Project Name)
**SpamGuard-ML**: Hệ thống phân loại thư rác (Email Spam Classification Engine) ứng dụng học máy.

## 2. Bài toán (Problem Statement)
* **Mục tiêu:** Xây dựng quy trình học máy phân loại nhị phân tự động xác định một email là **Spam** (Thư rác - nhãn `1`) hay **Ham** (Hợp lệ - nhãn `0`).
* **Đầu vào dữ liệu:** Chuỗi văn bản tổng hợp từ tiêu đề và nội dung email: `text = subject + " " + body`.
* **Đặc thù bài toán:** Dữ liệu thư tín thực tế cần chú trọng độ chuẩn xác (Precision) để tránh chặn nhầm email quan trọng của người dùng.

## 3. Đối tượng sử dụng (Target Users)
* **Người dùng cuối:** Nhập email vào Ô chat dòng lệnh (CLI) hoặc Web tương tác để kiểm tra nhanh thư rác thời gian thực.
* **Kỹ sư / Thành viên nhóm:** Huấn luyện mô hình, tinh chỉnh siêu tham số và đánh giá trực tiếp trên file Jupyter Notebook chung (`spam_classification.ipynb`).
* **Hội đồng đánh giá môn học (UTH):** Đánh giá tính đúng đắn toán học, quy trình xử lý dữ liệu và kết quả so sánh mô hình.

## 4. Đặc tả Input / Output (I/O Specification)
| Luồng | Tên trường | Kiểu dữ liệu | Định dạng / Ví dụ | Ý nghĩa kỹ thuật |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | `email_text` | `str` | Chuỗi ký tự UTF-8 | Tiêu đề và nội dung email cần phân loại |
| **Input (batch)** | `df` | `pd.DataFrame` | Bảng dữ liệu CEAS_08 | 7 trường: `sender, receiver, date, subject, body, label, urls` |
| **Output** | `label` | `str` / `int` | `"SPAM"` (`1`) / `"HAM"` (`0`) | Kết quả phân loại nhị phân |
| **Output** | `confidence` | `float` | $50.0\% - 100.0\%$ | Độ tin cậy (xác suất cao nhất từ `predict_proba`) |
| **Output** | `latency_ms` | `float` | mili-giây (ms) | Thời gian xử lý từ lúc nhập text đến khi có kết quả |

## 5. Tính năng cốt lõi (Core Features)
1. **Làm sạch dữ liệu văn bản:** Bóc tách mã HTML, chuyển chữ thường, loại bỏ từ dừng (stop words).
2. **Trích xuất đặc trưng kết hợp:** Biến đổi văn bản qua TF-IDF ngram (1,2) kết hợp các đặc trưng số đếm (tần suất `!`, `$`, URL, độ dài).
3. **Huấn luyện 3 thuật toán học máy (theo chuẩn yêu cầu đề bài):**
   - Naive Bayes (`MultinomialNB`)
   - Logistic Regression
   - Support Vector Machine (`LinearSVC` có hiệu chuẩn xác suất)
4. **Giao diện tương tác:** Ô Chat dòng lệnh (Console Chat) và Web Demo Streamlit.

## 6. Ràng buộc công nghệ (Technology Constraints)
* **Ngôn ngữ:** Python 3.10+ (tương thích Windows, macOS, Linux).
* **Thư viện cốt lõi:** `scikit-learn`, `pandas`, `numpy`, `joblib`, `streamlit`, `matplotlib`, `seaborn`.
* **Định dạng thực thi:** Toàn bộ luồng huấn luyện được tích hợp trong **1 file Jupyter Notebook duy nhất** (`notebooks/spam_classification.ipynb`); Python Script (`.py`) dùng cho ô chat và web demo.
* **Đường dẫn:** 100% sử dụng đường dẫn tương đối từ thư mục gốc của bài thực hành.

## 7. Tiêu chí đánh giá (Evaluation Criteria)
* **Phương pháp kiểm thử:** Đánh giá độc lập trên tập Test (chia tập Train/Test theo tỷ lệ 80/20 có phân tầng `stratify`).
* **Các chỉ số đo lường:**
  - **Accuracy:** Đo lường tỷ lệ dự đoán chính xác tổng thể.
  - **Precision (Spam):** Đo lường độ chuẩn xác trên nhãn Spam (chỉ số ưu tiên nhằm tránh việc phân loại nhầm thư quan trọng vào hòm thư rác).
  - **Recall (Spam):** Đo lường khả năng bao quát, không bỏ sót thư rác.
  - **F1-score:** Đo lường sự cân bằng hài hòa giữa Precision và Recall.
  - **Confusion Matrix:** Vẽ biểu đồ nhiệt (Heatmap) thể hiện chi tiết ma trận nhầm lẫn của từng mô hình.
* **So sánh mô hình:** Đối chiếu kết quả giữa 3 mô hình (Naive Bayes, Logistic Regression, SVM) để chọn ra mô hình có hiệu năng toàn diện nhất lưu vào `best_model.joblib`.
* **Thời gian phản hồi (Latency):** Tối ưu hóa thời gian dự đoán phục vụ phản hồi thời gian thực.

---

### Tóm tắt tập dữ liệu (`CEAS_08.csv`):
* **Nguồn gốc:** Tập dữ liệu email thực tế từ hội thảo quốc tế CEAS 2008.
* **Mục tiêu tiền xử lý:** Kiểm tra và loại bỏ các dòng trùng lặp nội dung, xử lý dữ liệu khuyết thiếu và chuẩn hóa văn bản trước khi đưa vào huấn luyện mô hình.
