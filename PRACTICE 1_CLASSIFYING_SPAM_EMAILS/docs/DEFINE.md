# BẢN ĐẶC TẢ DỰ ÁN (PROJECT DEFINITION SPECIFICATION)
## PRACTICE 1: CLASSIFYING SPAM EMAILS

> **Mã tài liệu:** `DOC-DEF-001`  
> **Dự án:** `SpamGuard-ML (Email Spam Classification Engine)`  
> **Tập dữ liệu nguồn:** `data/raw/spam.csv` (Kaggle Dataset: 5.572 dòng, 2 cột `Category` & `Message`)  
> **Quy chuẩn thực thi:** Bám sát toàn diện 4 bước Workflow và các mục mở rộng (Feature Engineering, Hyperparameter Tuning, Ensemble Methods) theo đúng đề bài.

---

## 1. Mục tiêu bài toán (Problem Statement)
Cho tập dữ liệu email/tin nhắn `spam.csv`, xây dựng mô hình học máy phân loại chính xác từng thông điệp là **"spam"** (thư rác) hay **"not spam" (ham)** (thư hợp lệ).

---

## 2. Đặc tả tập dữ liệu thực tế (Dataset Specification)
* **Đường dẫn dữ liệu:** `data/raw/spam.csv` (đường dẫn tương đối trong dự án)
* **Quy mô dữ liệu:** `5.572` dòng dữ liệu thực tế.
  * Lớp `ham` (hợp lệ): 4.825 mẫu (86.6%)
  * Lớp `spam` (thư rác): 747 mẫu (13.4%)
  * *Nhận xét:* Dữ liệu bị mất cân bằng nhãn (imbalanced data), do đó không chỉ nhìn vào `Accuracy` mà bắt buộc phải tối ưu hóa `Precision` và `F1-score`.
* **Cấu trúc 2 cột gốc:**
  1. `Category`: Nhãn phân loại (`ham` hoặc `spam`).
  2. `Message`: Chuỗi văn bản email / tin nhắn cần phân loại.

---

## 3. Khung quy trình chuẩn 4 bước & Các hạng mục mở rộng (Workflow & Considerations)

Hệ thống bám sát 100% đúng 4 bước theo đề bài kết hợp các yêu cầu bổ sung (Additional Considerations):

### Bước 1: Data Preprocessing & Feature Engineering
* **Trích xuất đặc trưng bổ trợ (Feature Engineering - theo gợi ý đề bài):**
  Trước khi làm sạch văn bản, trích xuất các tín hiệu spam quan trọng:
  * `char_freq_exclamation`: Tần suất/Số lượng dấu chấm cảm (`!`) - dấu hiệu spam điển hình.
  * `char_freq_currency`: Tần suất xuất hiện ký hiệu tiền tệ (`$`, `£`, `€`).
  * `uppercase_ratio`: Tỷ lệ chữ viết HOA trong văn bản (ví dụ: *FREE, WINNER, URGENT*).
  * `message_length`: Tổng độ dài ký tự của thông điệp.
* **Làm sạch văn bản (Cleaning):**
  * Bóc tách các thẻ HTML (`<[^>]+>`).
  * Chuyển toàn bộ về chữ thường (lowercase).
  * Loại bỏ từ dừng (stop words tiếng Anh chuẩn của NLTK / Scikit-learn).
  * Xử lý dấu câu phù hợp để không làm mất thông tin ngữ nghĩa.
* **Vector hóa đặc trưng (Numerical Conversion):**
  * Sử dụng kỹ thuật **TF-IDF (Term Frequency - Inverse Document Frequency)** với `ngram_range=(1, 2)`, `max_features=5000`.
  * Kết hợp ma trận TF-IDF với các đặc trưng số thông qua `ColumnTransformer` / `FeatureUnion`.
* **Chia tách dữ liệu (Data Splitting):**
  * Chia tách độc lập: **80% Training set** và **20% Testing set** với cơ chế `stratify=y` và `random_state=42` để bảo toàn nguyên vẹn tỷ lệ phân bố nhãn.

### Bước 2: Model Training, Hyperparameter Tuning & Ensemble Methods
* **Huấn luyện 3 thuật toán phân loại cốt lõi theo đề bài:**
  1. **Logistic Regression:** Mô hình hóa xác suất phân loại nhị phân.
  2. **Support Vector Machines (SVM):** Sử dụng `LinearSVC` được hiệu chuẩn xác suất qua `CalibratedClassifierCV` để tìm siêu phẳng phân tách tối ưu.
  3. **Naive Bayes:** Sử dụng `MultinomialNB` dựa trên định lý Bayes với giả định độc lập có điều kiện.
* **Tối ưu hóa siêu tham số (Hyperparameter Tuning - theo đề bài):**
  * Áp dụng `GridSearchCV` (5-Fold Cross Validation) trên tập Training:
    * Logistic Regression: Tinh chỉnh hệ số điều chuẩn $C \in [0.1, 1.0, 10.0]$, solver.
    * Linear SVM: Tinh chỉnh $C \in [0.1, 1.0, 10.0]$.
    * MultinomialNB: Tinh chỉnh tham số làm mịn Laplace $\alpha \in [0.1, 0.5, 1.0]$.
* **Mô hình kết hợp (Ensemble Methods - theo đề bài):**
  * Thử nghiệm thêm mô hình tổ hợp `VotingClassifier` (kết hợp mềm/cứng giữa 3 mô hình trên) hoặc `RandomForestClassifier` nhằm cải thiện khả năng tổng quát hóa và chống overfitting.

### Bước 3: Model Evaluation (Đánh giá mô hình toàn diện)
* Đánh giá hiệu năng của tất cả các mô hình trên tập Testing set độc lập (20%) thông qua đúng 4 chỉ số:
  1. **Accuracy** (Độ chính xác tổng thể).
  2. **Precision** (Độ chuẩn xác trên lớp spam - giảm thiểu tối đa False Positive, tránh đưa nhầm email quan trọng vào hòm thư rác).
  3. **Recall** (Độ nhạy - tỷ lệ bắt trúng toàn bộ thư rác).
  4. **F1-score** (Trung bình điều hòa giữa Precision và Recall).
* **Phân tích chi tiết (Confusion Matrix & Error Analysis):**
  * Xuất ma trận nhầm lẫn (Confusion Matrix) để theo dõi chi tiết số ca dự đoán đúng/sai trên từng lớp.
* **Đóng gói mô hình tối ưu (Model Artifact):**
  * Tự động chọn mô hình có điểm F1-score và Precision cân bằng cao nhất làm Best Model.
  * Xuất mô hình hoàn chỉnh (bao gồm toàn bộ Pipeline tiền xử lý + bộ phân loại) ra `models/best_model.joblib`.
  * Xuất báo cáo điểm số chi tiết ra `models/metadata.json`.

### Bước 4: Model Deployment (Triển khai & Ô Chat tương tác)
* Nạp mô hình đã đóng gói `best_model.joblib` để phân loại email mới chưa từng thấy.
* **Giao diện Console Chat tương tác (`scripts/run_predict.py`):**
  * Chạy trực tiếp qua terminal `python scripts/run_predict.py`.
  * Vòng lặp tương tác: Chờ người dùng nhập văn bản trực tiếp `💬 Nhập email:`.
  * Phản hồi tức thì nhãn: `[ 🚨 SPAM ]` hoặc `[ ✅ HAM ]` kèm độ tin cậy `%` (Confidence) và thời gian phản hồi (Latency $< 50\text{ms}$).
  * Hỗ trợ gõ `exit` hoặc `quit` để thoát êm thuận.

---

## 4. Tiêu chí nghiệm thu (Acceptance Criteria)
* Bám sát đầy đủ cả 4 bước Workflow lẫn 3 mục mở rộng (Feature Engineering, Tuning, Ensemble) của đề bài.
* Chạy trơn tru trên tập dữ liệu chuẩn 5.572 dòng của `data/raw/spam.csv`.
* Các chỉ số đánh giá `Accuracy`, `Precision`, `F1-score` trên tập Test đạt $\ge 95\%$.
* Độ trễ phản hồi trong ô chat $< 50\text{ms / câu}$.
* Code tuân thủ module hóa, có unit test và file Jupyter Notebook trực quan hóa để phục vụ báo cáo.
