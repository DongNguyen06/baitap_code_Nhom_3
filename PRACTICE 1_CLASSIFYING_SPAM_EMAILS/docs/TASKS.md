# KẾ HOẠCH PHÂN RÃ CÔNG VIỆC THEO 4 BƯỚC WORKFLOW (TASKS BREAKDOWN)
## PRACTICE 1: CLASSIFYING SPAM EMAILS

> **Mã tài liệu:** `DOC-TSK-001`  
> **Nguyên tắc:** Bám sát toàn diện 4 bước Workflow và các mục mở rộng (Feature Engineering, Hyperparameter Tuning, Ensemble) của đề bài.  
> **Trạng thái:** `READY TO IMPLEMENT (SẴN SÀNG TRIỂN KHAI)`  

---

## 📌 BẢNG TỔNG QUAN 4 MILESTONES

| Milestone | Tương ứng trong Đề bài | Nhiệm vụ kỹ thuật cốt lõi | Trạng thái |
| :--- | :--- | :--- | :---: |
| **M1** | **1. Data Preprocessing & Feature Engineering** | Trích xuất đặc trưng bổ trợ (ký tự !, $, viết hoa), làm sạch text, TF-IDF, chia Train/Test 80/20 | ⏳ Ready |
| **M2** | **2. Model Training & Hyperparameter Tuning** | Huấn luyện 3 mô hình (NB, LR, SVM) + Tinh chỉnh siêu tham số (GridSearchCV) + Ensemble (Voting/RF) | ⏳ Ready |
| **M3** | **3. Model Evaluation** | Đánh giá 4 chỉ số (Acc, Prec, Rec, F1), vẽ Confusion Matrix, lưu `best_model.joblib` | ⏳ Ready |
| **M4** | **4. Model Deployment & Interactive Chat** | Xây dựng Ô Chat Console tương tác `scripts/run_predict.py`, Unit Test và Notebook làm báo cáo | ⏳ Ready |

---

## CHI TIẾT KẾ HOẠCH TỪNG MILESTONE

### 🟢 MILESTONE 1: DATA PREPROCESSING & FEATURE ENGINEERING
*Mục tiêu đề bài:* 
* Clean the data by removing stop words, punctuation, and HTML tags.
* Feature Engineering: Character frequency (!, $), word frequency, message length, uppercase ratio.
* Convert text data into numerical features (using TF-IDF).
* Split the dataset into training and testing sets.

* **Task 1.1: Quản lý cấu hình & Nạp dữ liệu nguồn (`src/config.py`)**
  * Nạp `data/raw/spam.csv` (5.572 dòng) bằng Pandas.
  * Chuẩn hóa tên cột: `Category` (`ham` $\rightarrow$ 0, `spam` $\rightarrow$ 1) và `Message`.
* **Task 1.2: Xây dựng hàm trích xuất đặc trưng bổ trợ (`src/preprocessing.py`)**
  * Trích xuất số lượng dấu cảm thán `!`.
  * Trích xuất số lượng ký hiệu tiền tệ (`$`, `£`, `€`).
  * Trích xuất tỷ lệ chữ in hoa (`uppercase_ratio`).
  * Trích xuất độ dài ký tự của văn bản (`message_length`).
* **Task 1.3: Làm sạch văn bản & Vector hóa TF-IDF (`src/preprocessing.py`)**
  * Bóc tách các thẻ HTML bằng regex `<[^>]+>`.
  * Chuyển chữ thường, loại bỏ stop words tiếng Anh.
  * Cấu hình `TfidfVectorizer(max_features=5000, ngram_range=(1,2))`.
  * Ghép nối nhánh TF-IDF và nhánh đặc trưng số thành 1 Transformer thống nhất.
* **Task 1.4: Chia tách tập dữ liệu**
  * Chia 80% Train và 20% Test bằng `train_test_split(stratify=y, test_size=0.2, random_state=42)`.
* **Tiêu chí kiểm thử M1:** Pipeline tiền xử lý biến đổi chuỗi thô thành ma trận đặc trưng đúng kích thước, không sinh giá trị NaN/Null.

---

### 🟢 MILESTONE 2: MODEL TRAINING, HYPERPARAMETER TUNING & ENSEMBLE
*Mục tiêu đề bài:* 
* Train classification models: Logistic Regression, Support Vector Machines (SVM), Naive Bayes.
* Hyperparameter Tuning: Optimize model parameters (regularization, alpha).
* Ensemble Methods: Combine multiple models (Voting / Random Forest).

* **Task 2.1: Xây dựng Pipeline cho 3 thuật toán cốt lõi (`src/trainer.py`)**
  * Mô hình 1: `MultinomialNB()` (Naive Bayes).
  * Mô hình 2: `LogisticRegression(max_iter=1000)` (Logistic Regression).
  * Mô hình 3: `CalibratedClassifierCV(LinearSVC())` (SVM có hiệu chuẩn xác suất).
* **Task 2.2: Tối ưu hóa siêu tham số bằng GridSearchCV (`src/trainer.py`)**
  * Naive Bayes: Tinh chỉnh $\alpha \in [0.1, 0.5, 1.0]$.
  * Logistic Regression: Tinh chỉnh $C \in [0.1, 1.0, 10.0]$.
  * SVM: Tinh chỉnh $C \in [0.1, 1.0, 10.0]$.
  * Thực hiện 5-Fold Cross Validation trên tập 80% Training.
* **Task 2.3: Xây dựng mô hình kết hợp Ensemble (`src/trainer.py`)**
  * Xây dựng `VotingClassifier(voting='soft'/'hard')` kết hợp cả 3 mô hình tốt nhất, hoặc mô hình `RandomForestClassifier`.
* **Tiêu chí kiểm thử M2:** Tất cả các mô hình hoàn thành quá trình fit và tìm được bộ siêu tham số tốt nhất mà không lỗi tràn bộ nhớ.

---

### 🟢 MILESTONE 3: MODEL EVALUATION & ERROR ANALYSIS
*Mục tiêu đề bài:* 
* Evaluate performance on testing set using metrics: accuracy, precision, recall, and F1-score.
* Bổ sung Confusion Matrix để đánh giá bài toán lệch lớp.

* **Task 3.1: Đánh giá chi tiết trên tập Testing (20%)**
  * Đo đạc và in bảng so sánh hiệu năng cho cả 4 mô hình (NB, LR, SVM, Ensemble) với đúng 4 chỉ số:
    1. $\text{Accuracy}$
    2. $\text{Precision}$ (ưu tiên cao nhất để hạn chế đánh nhầm thư quan trọng thành spam)
    3. $\text{Recall}$
    4. $\text{F1-score}$
  * Tính toán ma trận nhầm lẫn (Confusion Matrix): số ca True Positive, False Positive, True Negative, False Negative.
* **Task 3.2: Lựa chọn Best Model & Xuất Artifact**
  * Tự động chọn mô hình có điểm cân bằng Precision và F1-score cao nhất.
  * Lưu toàn bộ End-to-End Pipeline vào `models/best_model.joblib`.
  * Lưu báo cáo chỉ số và cấu hình vào `models/metadata.json`.
* **Task 3.3: Script huấn luyện tự động toàn trình (`scripts/run_train.py`)**
  * Cho phép người dùng chạy 1 lệnh duy nhất để tự động thực thi Bước 1 $\rightarrow$ Bước 2 $\rightarrow$ Bước 3.
* **Tiêu chí kiểm thử M3:** File `models/best_model.joblib` được tạo ra hợp lệ, nạp lại được và nhận trực tiếp chuỗi văn bản thô để dự đoán.

---

### 🟢 MILESTONE 4: MODEL DEPLOYMENT, INTERACTIVE CHAT & BÁO CÁO
*Mục tiêu đề bài:* 
* Deploy the trained model to classify new, unseen emails.
* Cung cấp ô Chat tương tác nhập văn bản trực tiếp trong Terminal.
* Bổ sung Notebook trực quan hóa để phục vụ báo cáo bài tập lớn.

* **Task 4.1: Xây dựng Module suy luận (`src/inference.py`)**
  * Nạp `best_model.joblib` vào RAM duy nhất 1 lần khi khởi động.
  * Cung cấp hàm `predict_email(text)` trả về: Nhãn (`SPAM` hoặc `HAM`), Độ tin cậy (`confidence_score %`), và Thời gian xử lý (`latency_ms`).
* **Task 4.2: Xây dựng Giao diện Ô Chat Console tương tác (`scripts/run_predict.py`)**
  * Khởi tạo giao diện terminal thân thiện, có khung hướng dẫn rõ ràng.
  * Vòng lặp `while True`:
    * Nhập `💬 Bạn: `
    * Nếu nhập `exit` hoặc `quit`: Thoát chương trình êm thuận.
    * Nếu nhập chuỗi trống: Nhắc người dùng nhập lại.
    * Nếu nhập nội dung email: Dự đoán ngay lập tức và in:
      `🤖 AI: [ 🚨 SPAM ] | Độ tin cậy: XX.XX% | Độ trễ: X.X ms`
* **Task 4.3: Viết bộ kiểm thử tự động toàn diện (`tests/test_workflow.py`)**
  * Kiểm thử trơn tru toàn bộ luồng từ tiền xử lý, huấn luyện đến dự đoán mẫu email thực tế.
* **Task 4.4: Tạo Jupyter Notebook báo cáo trực quan (`notebooks/eda_and_experiments.ipynb`)**
  * Vẽ biểu đồ phân bố nhãn (Ham vs Spam).
  * Biểu đồ phân tích độ dài và tần suất ký tự đặc biệt giữa Spam và Ham.
  * Biểu đồ nhiệt ma trận nhầm lẫn (Confusion Matrix Heatmap).
  * Bảng so sánh trực quan hiệu năng các mô hình phục vụ báo cáo/thuyết trình.
* **Tiêu chí kiểm thử M4:** Người dùng chạy `python scripts/run_predict.py` gõ thử câu bất kỳ đều nhận kết quả trong nháy mắt ($< 5\text{ms}$) mà không phát sinh lỗi.
