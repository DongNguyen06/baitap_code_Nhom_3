# THIẾT KẾ DỰ ÁN — PRACTICE 1: CLASSIFYING SPAM EMAILS
## BÀI THỰC HÀNH HỌC MÁY (MACHINE LEARNING LAB) — NHÓM 3 (UTH)

---

### 1. WHAT — LÀM CÁI GÌ?
* **Bài toán:** Phân loại nhị phân trên dữ liệu văn bản email: xác định một email là **Thư thường (Ham - 0)** hay **Thư rác (Spam - 1)**.
* **Đầu vào / Đầu ra:**
  * *Đầu vào:* Chuỗi ký tự thô gồm Tiêu đề (`Subject`) và Nội dung (`Body`) email.
  * *Đầu ra:* Nhãn dự đoán (`Ham` hoặc `Spam`) kèm xác suất / độ tin cậy của phán đoán.
* **Dữ liệu thực nghiệm (`CEAS_08.csv`):**
  * Gốc: 39.154 dòng, 2 cột văn bản; Ham chiếm 44,2% (17.312 email), Spam chiếm 55,8% (21.842 email).
  * Sau làm sạch & khử trùng lặp: Còn lại 32.660 email sạch (Ham: 52,3%, Spam: 47,7%).
* **Phạm vi bài Lab:**
  * *Có làm:* Làm sạch dữ liệu, trích xuất đặc trưng kết hợp (TF-IDF + 7 đặc trưng miền), tự code 3 thuật toán Scratch (Naive Bayes, Logistic Regression, SVM), tinh chỉnh siêu tham số, so sánh đối chứng và dựng demo dự đoán tương tác.
  * *Không làm:* Không dựng hệ thống production thương mại, không lọc email server thời gian thực, không xử lý đa ngôn ngữ ngoài tiếng Anh.
* **Kết quả mong đợi:** Một pipeline chạy thông suốt từ đầu đến cuối, bảng so sánh 3 mô hình, chỉ số kiểm định trên tập Test đạt $\text{Accuracy, Precision, F1} \ge 95\%$.

---

### 2. WHY — TẠI SAO LÀM & TẠI SAO CHỌN NHƯ VẬY?
* **Vì sao chọn bài toán lọc Spam:** Đây là bài toán kinh điển của phân loại văn bản trong Học máy, kích thước dữ liệu vừa vặn, dễ huấn luyện trên máy cá nhân và rất sát với thực tế đời sống.
* **Vì sao dùng Học máy thay vì luật cứng (Rule-based):** Các luật cứng (như hễ thấy từ `FREE`, `VIAGRA` là chặn) rất dễ bị người gửi spam lách qua (dùng teencode, chèn dấu cách, thẻ HTML ẩn) và dễ chặn nhầm thư quan trọng. Học máy tự học mẫu phân bố thống kê từ hàng chục nghìn email nên thích ứng linh hoạt hơn nhiều.
* **Vì sao chọn các giải pháp kỹ thuật cụ thể:**
  * *TF-IDF (1,2-gram) + Chi-Square:* Biểu diễn tần suất từ kèm độ quan trọng trong toàn tập văn bản, giữ cụm 2 từ (như `click here`, `risk free`), lọc lấy 5.000 từ quan trọng nhất để giảm chiều tính toán.
  * *7 đặc trưng miền thủ công:* Bắt các "dấu vân tay" đặc trưng của spam mà từ vựng đơn lẻ bỏ sót (tỷ lệ chữ HOA, số dấu `!`, ký hiệu tiền tệ `$`, số link URL, tiền tố `Re:`, nén log độ dài).
  * *Tự cài đặt 3 thuật toán Scratch (NB, LR, SVM):* Đại diện 3 trường phái học máy khác nhau (Xác suất thống kê — Tối ưu hóa hàm mất mát — Siêu phẳng phân tách lề cực đại).
  * *Chia tập 80/20 có Stratify:* Giữ nguyên tỷ lệ phân bố giữa Ham và Spam ở cả 2 tập học và thi, chống học lệch.
  * *Trọng tâm Precision & F1-Score (thay vì chỉ nhìn Accuracy):* Trong bài toán email, lỗi bắt nhầm một thư công việc quan trọng vào hòm thư rác (False Positive) gây hậu quả tai hại hơn nhiều so với việc lỡ sót một lá thư rác (False Negative).
* **Mục tiêu học tập:** Nắm trọn vẹn chu trình một dự án Khoa học Dữ liệu thực tế: từ dữ liệu thô lộn xộn đến mô hình dự đoán hoàn chỉnh.

---

### 3. WHICH / WHEN / WHERE — DÙNG CÁI NÀO, KHI NÀO, Ở ĐÂU?

#### WHICH — Công cụ, thuật toán và tiêu chí đánh giá
| Thành phần | Lựa chọn kỹ thuật trong dự án | Lý do lựa chọn |
| :--- | :--- | :--- |
| **Thư viện cốt lõi** | `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn` | Bộ công cụ chuẩn mực cho tiền xử lý, tính toán ma trận và trực quan hóa. |
| **3 Thuật toán Scratch** | 1. Multinomial Naive Bayes (Laplace $\alpha$)<br>2. Logistic Regression (L2 Regularization $C$)<br>3. Linear SVM (Hinge Loss, Pegasos $C$) | Tự code bằng NumPy để hiểu bản chất thuật toán. |
| **Thước đo đánh giá** | Accuracy, Precision, Recall, F1-Score & Ma trận nhầm lẫn (Confusion Matrix) | Đánh giá toàn diện năng lực nhận diện spam và đo đếm chính xác số ca bắt nhầm (FP). |
| **Tiêu chí chọn Best Model** | Mô hình có **F1-Score cao nhất** kết hợp **Precision cao** (hạn chế tối đa FP) | Cân bằng hoàn hảo giữa khả năng bắt thư rác và độ an toàn cho thư thường. |

#### WHEN — Tiến trình thực hiện & Điều kiện dừng
* **Giai đoạn 1:** Khám phá & Làm sạch dữ liệu (`clean_emails.csv`).
* **Giai đoạn 2:** Trích xuất đặc trưng kết hợp & Tách tập Train/Test 80/20 (`train.csv`, `test.csv`).
* **Giai đoạn 3:** Huấn luyện 3 mô hình Scratch & Tinh chỉnh siêu tham số qua 5-Fold Stratified CV.
* **Giai đoạn 4:** Đánh giá độc lập trên tập Test, vẽ Confusion Matrix và xuất Top 15 từ khóa spam.
* **Giai đoạn 5:** Đóng gói Pipeline vào `models/saved_models/` và dựng giao diện thử nghiệm tương tác.
* **Điều kiện dừng & Nghiệm thu:** Khi cả 3 mô hình vượt ngưỡng $\text{F1} \ge 95\%$, trong đó mô hình Quán quân kiểm soát số ca bắt nhầm $\text{FP} \le 50$ trên hơn 6.500 mẫu kiểm tra.

#### WHERE — Cây thư mục tổ chức thực tế của dự án
```text
PRACTICE 1_CLASSIFYING_SPAM_EMAILS/
├── data/
│   ├── raw/                      # Dữ liệu gốc
│   │   └── CEAS_08.csv           # Tập dữ liệu email thô ban đầu (39.154 email)
│   ├── processed/                # Dữ liệu đã xử lý
│   │   ├── clean_emails.csv      # Dữ liệu sạch sau khử trùng lặp NLP (32.660 email)
│   │   ├── train.csv             # Tập huấn luyện trực quan mở được Excel (80% — 26.128 email) 
│   │   ├── test.csv              # Tập kiểm tra trực quan mở được Excel (20% — 6.532 email) 
│   │   └── train_test_data.joblib# Tệp nhị phân nạp nhanh cho code Python
│   └── scripts/                  # Kịch bản tiền xử lý dữ liệu
│       ├── data_preprocessing.py # Module xử lý khuyết thiếu, clean_text và khử trùng lặp
│       └── eda.py                # Module khảo sát cấu trúc, phân tích đơn biến, đa biến và IQR
├── docs/
│   ├── description/              # Đặc tả dự án, thiết kế bài lab & phân công công việc
│   │   ├── algorithms/           # Lý thuyết 3 thuật toán Scratch (NB, LR, SVM)
│   │   │   ├── logistic_regression.md
│   │   │   ├── naive_bayes_classifier.md
│   │   │   └── support_vector_machine_classifier.md
│   │   ├── project_description.md# Mô tả tổng quan bài lab
│   │   ├── project_design.md     # Bản thiết kế chi tiết 5W1H này
│   │   └── TASKS.md              # Bảng phân công công việc nhóm
│   ├── images/                   # Biểu đồ trực quan hóa EDA và kết quả mô hình
│   │   ├── eda_01_univariate_analysis.png
│   │   ├── eda_02_length_distribution_boxplot.png
│   │   ├── model_01_confusion_matrices.png
│   │   ├── model_02_top_15_spam_keywords.png
│   │   └── model_03_performance_comparison.png
│   └── report/                   # Các trang báo cáo HTML độc lập nộp giảng viên
│       ├── index.html            # Bảng điều khiển trung tâm (Dashboard)
│       ├── 01_data_report.html   # Báo cáo khám phá & làm sạch dữ liệu
│       ├── 02_feature_pipeline_report.html # Báo cáo kỹ thuật đặc trưng & Pipeline
│       ├── 03_model_evaluation_report.html # Báo cáo huấn luyện & đánh giá mô hình
├── models/
│   ├── algorithms/               # Mã nguồn 3 thuật toán tự cài đặt From Scratch (NumPy)
│   │   ├── logistic_regression.py
│   │   ├── naive_bayes_classifier.py
│   │   └── support_vector_machine.py
│   └── saved_models/             # Lưu trữ mô hình tối ưu và siêu dữ liệu
│       ├── best_model.joblib     # Mô hình quán quân Linear SVM
│       ├── linear_svm.joblib
│       ├── logistic_regression.joblib
│       ├── naive_bayes.joblib
│       ├── preprocessor.joblib   # Pipeline biến đổi ColumnTransformer đã fit
│       └── metadata.json         # Bảng chỉ số nghiệm thu và Top từ khóa spam
├── notebooks/                    # 4 Notebooks thực hành tuần tự từng chặng
│   ├── 01_data_cleaning_and_eda.ipynb
│   ├── 02_feature_engineering_and_pipeline.ipynb
│   ├── 03_model_training_evaluation_and_explain.ipynb
│   └── demo.ipynb                # Notebook thử nghiệm dự đoán email tương tác đa mô hình
├── scripts/                      # Các module tái sử dụng và công cụ thực thi
│   ├── feature_engineering.py    # Chuẩn bị dữ liệu & trích xuất đặc trưng
│   ├── model_training.py         # Huấn luyện mô hình & tinh chỉnh GridSearchCV
│   ├── model_evaluation.py       # Đánh giá trên tập test & trích xuất Top từ khóa
│   ├── run_pipeline.py           # Kịch bản chạy tự động toàn bộ quy trình
│   └── test_all_components.py    # Bộ kiểm thử tự động toàn diện hệ thống
├── README.md
└── requirements.txt
```

---

### 4. HOW — QUY TRÌNH XÂY DỰNG 5 BƯỚC (ĐẦU VÀO -> XỬ LÝ -> ĐẦU RA)

* **Bước 1: Khám phá & Làm sạch dữ liệu (Data Cleaning & EDA):**
  * *Đầu vào:* Tệp `CEAS_08.csv` gốc (39.154 dòng).
  * *Xử lý:* Điền chuỗi rỗng cho tiêu đề khuyết; gỡ sạch thẻ HTML ngụy trang, URL và ký tự rác bằng Regex; chuyển chữ thường; khử triệt để 6.494 mẫu thư rác trùng lặp đa hình; phân tích phân bố nhãn và độ dài qua IQR.
  * *Đầu ra:* Tệp dữ liệu sạch `clean_emails.csv` (32.660 dòng) và 2 biểu đồ phân bố EDA.

* **Bước 2: Kỹ thuật đặc trưng & Phân tách tập dữ liệu (Feature Engineering & Split):**
  * *Đầu vào:* Dữ liệu sạch `clean_emails.csv`.
  * *Xử lý:* Trích xuất 7 đặc trưng miền số (`caps_ratio`, dấu `!`, ký hiệu `$`, số link URL, tiền tố `Re:`, độ dài tiêu đề và log độ dài thân thư); biến đổi văn bản qua TF-IDF (1,2-gram) và sàng lọc 5.000 từ vựng bằng Chi-Square; ghép tất cả qua `ColumnTransformer`; phân chia tập Train (80%) và Test (20%) có bảo toàn tỷ lệ nhãn (`stratify=y`).
  * *Đầu ra:* Ma trận đặc trưng đã chuẩn hóa, các tệp `train.csv`, `test.csv` và bộ nạp nhanh `train_test_data.joblib`.

* **Bước 3: Huấn luyện mô hình & Tinh chỉnh siêu tham số (Model Training & Tuning):**
  * *Đầu vào:* Tập dữ liệu huấn luyện `X_train`, `y_train` (26.128 mẫu).
  * *Xử lý:* Huấn luyện 3 thuật toán Scratch (Multinomial Naive Bayes, Logistic Regression L2, Linear SVM Pegasos); chạy GridSearchCV kết hợp 5-Fold Stratified Cross-Validation để tìm bộ siêu tham số tối ưu ($\alpha, C$); đối với Linear SVM, kết hợp hàm Sigmoid tính khoảng cách tới siêu phẳng để suy ra xác suất tin cậy.
  * *Đầu ra:* 3 mô hình đã huấn luyện tối ưu và tệp lưu trữ mô hình Quán quân `best_model.joblib`.

* **Bước 4: Đánh giá mô hình & Phân tích sai số (Model Evaluation):**
  * *Đầu vào:* 3 mô hình đã tối ưu và tập kiểm tra `X_test`, `y_test` (6.532 mẫu).
  * *Xử lý:* Tính toán 4 chỉ số (Accuracy, Precision, Recall, F1); vẽ ma trận nhầm lẫn để kiểm soát số ca bắt nhầm thư thường (FP) và bỏ lọt thư rác (FN); bóc tách Top 15 từ khóa có trọng số cảnh báo spam lớn nhất từ mô hình tốt nhất.
  * *Đầu ra:* Biểu đồ ma trận nhầm lẫn `model_01_confusion_matrices.png`, biểu đồ so sánh `model_03_performance_comparison.png` và tệp chỉ số nghiệm thu `metadata.json`.

* **Bước 5: Thử nghiệm tương tác & Dự đoán thời gian thực (Demo):**
  * *Đầu vào:* Tiêu đề và nội dung email văn bản thô bất kỳ do người dùng gõ vào.
  * *Xử lý:* Nạp tự động `preprocessor.joblib` và mô hình được chọn (mặc định Linear SVM hoặc tùy chọn Naive Bayes / Logistic Regression); tự động tính toán 7 đặc trưng miền và vector hóa TF-IDF rồi đưa ra nhãn dự đoán cùng thanh đo độ tin cậy (Confidence score).
  * *Đầu ra:* Giao diện trực quan trong `notebooks/demo.ipynb` đưa ra kết quả phân loại Ham/Spam tức thì.
