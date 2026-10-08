# THIẾT KẾ KIẾN TRÚC HỆ THỐNG TINH GỌN (SYSTEM DESIGN)
## PRACTICE 1: CLASSIFYING SPAM EMAILS

> **Nguyên tắc thiết kế:** Bám sát 4 bước Workflow trong đề bài kết hợp các mục mở rộng (Feature Engineering, Hyperparameter Tuning, Ensemble), cấu trúc module hóa chuẩn mực, tối ưu cho tập dữ liệu thực tế `data/raw/spam.csv`.  

---

## 1. Cấu trúc cây thư mục hoàn chỉnh (Project Tree)

```text
E:\DH_GTVT\NĂM 3\ML\baitap_code_Nhom_3\PRACTICE 1_CLASSIFYING_SPAM_EMAILS/
├── data/
│   ├── raw/
│   │   └── spam.csv                  # Dữ liệu nguồn (5.572 dòng, 2 cột Category & Message)
│   └── processed/                    # Dữ liệu sạch / trung gian (nếu cần cache)
├── docs/                             # Thư mục tài liệu thiết kế & lộ trình
│   ├── DEFINE.md                     # Đặc tả bài toán, dataset thực tế và tiêu chuẩn nghiệm thu
│   ├── DESIGN.md                     # Thiết kế kiến trúc, pipeline & giao diện ô chat
│   ├── TASKS.md                      # Kế hoạch phân rã 4 Milestones công việc
│   └── TEAM_WORKFLOW.md              # Quy chuẩn phân công & phối hợp nhóm 7 người trên GitHub
├── notebooks/                        # Jupyter Notebook phục vụ làm báo cáo & thuyết trình
│   └── eda_and_report.ipynb          # Phân tích dữ liệu (EDA), biểu đồ và so sánh mô hình
├── models/
│   ├── naive_bayes.joblib            # [Model 1] Pipeline Naive Bayes đã train & tuning
│   ├── logistic_reg.joblib           # [Model 2] Pipeline Logistic Regression đã train & tuning
│   ├── svm.joblib                    # [Model 3] Pipeline Calibrated LinearSVC đã train & tuning
│   ├── best_model.joblib             # Pipeline tối ưu nhất (hoặc Ensemble)
│   └── metadata.json                 # Kết quả so sánh 4 chỉ số (Acc, Prec, Rec, F1) của 3 mô hình
├── scripts/
│   ├── run_train.py                  # Thực thi Workflow: Preprocessing -> Train 3 Models -> Tuning -> Eval
│   └── run_predict.py                # Thực thi Bước 4: Mở ô Chat Console tương tác thời gian thực
├── src/
│   ├── __init__.py
│   ├── config.py                     # Quản lý đường dẫn tương đối và siêu tham số mặc định
│   ├── preprocessing.py              # [Bước 1] Làm sạch văn bản
│   ├── features.py                   # [Bước 1] Trích xuất đặc trưng bổ trợ & TF-IDF
│   ├── models/                       # [Bước 2] Phân tách mô hình cho nhiều thành viên code song song
│   │   ├── __init__.py
│   │   ├── naive_bayes.py            # Naive Bayes + Tuning
│   │   ├── logistic_reg.py           # Logistic Regression + Tuning
│   │   ├── svm.py                    # Calibrated LinearSVC + Tuning
│   │   └── ensemble.py               # VotingClassifier & RandomForest
│   ├── evaluation.py                 # [Bước 3] Đánh giá 4 chỉ số & Confusion Matrix
│   └── inference.py                  # [Bước 4] Nạp 3 pipeline model và dự đoán so sánh thời gian thực
├── app.py                            # [Bước 4] Web Demo Streamlit so sánh đồng thời 3 mô hình đã train
├── tests/
│   ├── __init__.py
│   └── test_workflow.py              # Bộ kiểm thử tự động chứng minh toàn trình chạy trơn tru
├── requirements.txt                  # Danh sách thư viện (scikit-learn, pandas, numpy, joblib, streamlit...)
└── README.md                         # Hướng dẫn cài đặt và chạy nhanh hệ thống
```

---

## 2. Kiến trúc luồng xử lý toàn trình (End-to-End Workflow Pipeline)

```mermaid
flowchart TD
    subgraph W1["Bước 1: Data Preprocessing & Feature Engineering"]
        Raw["data/raw/spam.csv\n(Category, Message)"] --> Clean["Làm sạch: Bóc HTML,\nchữ thường, stop words"]
        Raw --> FE["Feature Engineering:\n- Đếm dấu cảm thán (!)\n- Ký hiệu tiền tệ ($, £, €)\n- Tỷ lệ viết HOA\n- Độ dài văn bản"]
        Clean --> TFIDF["TfidfVectorizer\n(ngram 1-2, 5000 feats)"]
        FE & TFIDF --> Comb["Ghép đặc trưng\n(FeatureUnion / ColumnTransformer)"]
        Comb --> Split["train_test_split (80/20 Stratified)"]
    end

    subgraph W2["Bước 2: Training, Tuning & Ensemble"]
        Split --> T1["GridSearchCV (Logistic Regression)"]
        Split --> T2["GridSearchCV (Calibrated LinearSVC)"]
        Split --> T3["GridSearchCV (Multinomial Naive Bayes)"]
        Split --> T4["Ensemble Model (VotingClassifier / RandomForest)"]
    end

    subgraph W3["Bước 3: Model Evaluation"]
        T1 & T2 & T3 & T4 --> Eval["Đánh giá trên tập Test 20%:\n- Accuracy, Precision, Recall, F1\n- Confusion Matrix Heatmap"]
        Eval --> SaveModels["Xuất toàn bộ 3 mô hình đã train:\n-> models/naive_bayes.joblib\n-> models/logistic_reg.joblib\n-> models/svm.joblib\n-> models/metadata.json"]
    end

    subgraph W4["Bước 4: Deployment & Comparative Demo"]
        SaveModels -.->|"Load cả 3 model vào RAM"| App["app.py (Web Streamlit)\nSo sánh song song 3 Cột"]
        SaveModels -.->|"Load Best Model"| ChatLoop["scripts/run_predict.py\n(Terminal Chat Loop)"]
        User["Người dùng nhập email"] --> App & ChatLoop
        App --> Comp["Hiển thị 3 Cột song song:\nNaive Bayes vs Logistic vs SVM"]
        ChatLoop --> Output["Hiển thị: [SPAM] / [HAM]\nkèm Confidence % & Latency (ms)"]
    end
```

---

## 3. Thiết kế kỹ thuật chi tiết các thành phần

### 3.1. Kỹ thuật đặc trưng kết hợp (Hybrid Feature Pipeline)
Để vừa tận dụng sức mạnh của biểu diễn từ ngữ vừa khai thác các tín hiệu spam theo đúng đề bài, Pipeline sử dụng kiến trúc ghép nối:
* **Nhánh Text (TF-IDF):** Làm sạch HTML, chuyển chữ thường, loại bỏ stop words, sinh ma trận n-gram (1, 2) cho từ khóa.
* **Nhánh Dense Features (Ký tự & Thống kê):** 
  * Số lượng dấu chấm cảm `!`.
  * Số lượng ký hiệu tiền tệ `$`, `£`, `€`.
  * Tỷ lệ ký tự in hoa / tổng số ký tự.
  * Tổng số ký tự trong thông điệp.
* Toàn bộ được đóng gói trong một `scikit-learn Pipeline` duy nhất, giúp các file `.joblib` có thể nhận trực tiếp văn bản thô đầu vào mà không cần bước tiền xử lý thủ công bên ngoài khi suy luận.

### 3.2. Xử lý xác suất (Probability Calibration) cho SVM
* **Vấn đề:** `LinearSVC` của Scikit-learn chỉ cung cấp `decision_function()` (khoảng cách đến siêu phẳng), không hỗ trợ phương thức `predict_proba()` mặc định.
* **Giải pháp:** Sử dụng `CalibratedClassifierCV(estimator=LinearSVC(C=...), cv=3)` hoặc dùng ánh xạ Sigmoid trên giá trị hàm quyết định. Điều này đảm bảo khi người dùng kiểm tra trên web hoặc chat terminal, cả 3 mô hình đều hiển thị độ tin cậy % (Confidence) mượt mà, không gặp lỗi `AttributeError`.

### 3.3. Thiết kế Giao diện Web Demo so sánh đồng thời 3 Mô hình (`app.py` Streamlit)
Web App nạp đồng thời cả 3 mô hình đã huấn luyện (`naive_bayes.joblib`, `logistic_reg.joblib`, `svm.joblib`). Khi người dùng dán nội dung email vào ô nhập và bấm **"Phân tích & So sánh"**, cả 3 mô hình sẽ cùng dự đoán và hiển thị kết quả song song trong 3 cột (`col1, col2, col3`):

```text
=================================================================================================
             🛡️ HỆ THỐNG SO SÁNH & PHÂN LOẠI EMAIL SPAM — NHÓM 3 (UTH)
=================================================================================================
[ Ô nhập văn bản email / tin nhắn cần kiểm tra:                                                ]
[ "WINNER!! You have won a $1000 cash prize! Claim code KL341. Valid 12 hours only. Call now!"   ]

                              [ 🚀 PHÂN TÍCH & SO SÁNH 3 MÔ HÌNH ]

-------------------------------------------------------------------------------------------------
     CỘT 1: NAIVE BAYES       │   CỘT 2: LOGISTIC REGRESSION  │    CỘT 3: LINEAR SVM
-------------------------------------------------------------------------------------------------
       [ 🚨 SPAM ]            │         [ 🚨 SPAM ]           │       [ 🚨 SPAM ]
   Độ tin cậy: 99.12%         │     Độ tin cậy: 98.45%        │   Độ tin cậy: 99.80%
   Thanh đo: [█████████░]     │     Thanh đo: [████████░░]    │   Thanh đo: [██████████]
   Độ trễ: 1.1 ms             │     Độ trễ: 0.9 ms            │   Độ trễ: 1.4 ms
-------------------------------------------------------------------------------------------------
                      📊 KẾT LUẬN CHUNG: ĐỒNG THUẬN 3/3 MÔ HÌNH LÀ SPAM!
```

#### Ưu điểm vượt trội của Giao diện 3 Cột:
1. **So sánh trực quan tuyệt đối:** Giúp Giảng viên thấy ngay sự đồng thuận hoặc khác biệt giữa 3 thuật toán trên các câu test thực tế.
2. **Khai thác toàn diện công sức nhóm:** Sử dụng đồng thời cả 3 mô hình đã train và tuning, không bỏ phí bất kỳ mô hình nào.
3. **Thanh tiến trình (Progress Bar):** Thể hiện trực quan % xác suất từ 0% đến 100%.

### 3.4. Thiết kế Ô Chat tương tác Terminal trong `scripts/run_predict.py`
Khi người dùng chạy `python scripts/run_predict.py`, giao diện Terminal Chat hiển thị như sau:

```text
======================================================================
🤖 SPAMGUARD-ML: BỘ LỌC EMAIL SPAM TƯƠNG TÁC THỜI GIAN THỰC
======================================================================
• Mô hình tối ưu: Calibrated LinearSVC (F1-score: 98.6%)
• Hướng dẫn: Nhập nội dung email/tin nhắn cần kiểm tra rồi nhấn [Enter].
• Thoát chương trình: Gõ 'exit' hoặc 'quit'.
======================================================================

💬 Bạn: Free entry in 2 a wkly comp to win FA Cup final tickets! Text FA to 87121
🤖 AI: [ 🚨 SPAM ] | Độ tin cậy: 99.85% | Độ trễ: 1.4 ms

💬 Bạn: Hi team, please find attached the meeting notes for tomorrow.
🤖 AI: [ ✅ HAM  ] | Độ tin cậy: 99.30% | Độ trễ: 0.9 ms

💬 Bạn: WINNER!! As a valued customer you have been selected to receive $1000 prize!
🤖 AI: [ 🚨 SPAM ] | Độ tin cậy: 99.92% | Độ trễ: 1.1 ms

💬 Bạn: exit
Tạm biệt! Kết thúc phiên làm việc.
```

### 3.5. Nguyên lý an toàn và hiệu năng:
1. **Nạp 1 lần (Pre-load):** Cả 3 file `.joblib` được nạp vào RAM ngay khi khởi động Web/Console.
2. **Xử lý ngoại lệ (Graceful Exit):** Bắt chuỗi rỗng và ngoại lệ mềm dẻo, không bao giờ văng lỗi traceback ngoài ý muốn.
3. **Độ trễ phản hồi cực thấp:** Suy luận trực tiếp trên RAM đạt $< 5\text{ms}$ cho mỗi lượt kiểm tra.
