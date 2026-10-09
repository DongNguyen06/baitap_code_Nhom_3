# KẾ HOẠCH PHÂN RÃ CÔNG VIỆC (TASKS BREAKDOWN) — SPAMGUARD-ML
## PRACTICE 1: CLASSIFYING SPAM EMAILS — NHÓM 3 (UTH)
> **Giảng viên hướng dẫn:** Tiến sĩ Nguyễn Thị Khánh Tiên  
> **Nguyên tắc phân công:** Toàn bộ nhóm cùng đóng góp vào **1 Notebook duy nhất** (`notebooks/spam_classification.ipynb`) theo cơ chế phân chia mục và gối đầu nhánh Git.

---

## 1. BẢNG PHÂN CÔNG 7 THÀNH VIÊN THEO CÁC MỤC TRONG NOTEBOOK

| STT | Thành viên & MSSV | Vai trò chính | Phần phụ trách trong Notebook / Repo | Nội dung nhiệm vụ chi tiết |
| :---: | :--- | :--- | :--- | :--- |
| **TV 1** | **Nguyễn Xuân Đông**<br>`94206000207` | **Trưởng nhóm (Leader)**<br>Kiến trúc & Điều phối | **Mục 1 & 7** trong Notebook<br>`README.md`, `.gitignore` | • Khởi tạo kho mã nguồn, tạo Notebook khung sườn, phân nhánh Git<br>• Khai báo thư viện dùng chung, cấu hình random seed tái lập kết quả<br>• Điều phối thứ tự merge nhánh và review code |
| **TV 2** | **Mai Hoàng Danh**<br>`066206005311` | **Data Cleaning Engineer** | **Mục 2** trong Notebook (`spam_classification.ipynb`) | • Tải `CEAS_08.csv`, khám phá dữ liệu, tạo cột `text = subject + " " + body`<br>• Kiểm tra và loại bỏ các bản ghi trùng lặp (nếu có), xử lý missing values<br>• Xây dựng hàm bóc tách HTML, chuẩn hóa chữ thường, lọc stop words<br>• Xuất tập dữ liệu sạch ra `data/processed/clean_emails.csv` |
| **TV 3** | **Trương Thành Công**<br>`87206002694` | **Feature Engineer** | **Mục 3** trong Notebook (`spam_classification.ipynb`) | • Trích xuất các đặc trưng số đếm (dấu !, $, liên kết URL, độ dài, chữ hoa)<br>• Cấu hình TF-IDF Vectorizer (ngram, giới hạn max_features)<br>• Phân chia tập dữ liệu Train/Test (80/20) có phân tầng nhãn `stratify` |
| **TV 4** | **Võ Khôi Nguyên**<br>`79206001370` | **ML Engineer 1**<br>(NB & Logistic Regression) | **Mục 4** trong Notebook (`spam_classification.ipynb`) | • Xây dựng Pipeline huấn luyện Naive Bayes (`MultinomialNB`)<br>• Xây dựng Pipeline huấn luyện Logistic Regression<br>• Tinh chỉnh siêu tham số với 5-Fold Cross-Validation và lưu file mô hình |
| **TV 5** | **Võ Thanh Phú**<br>`89206003078` | **ML Engineer 2**<br>(Support Vector Machine) | **Mục 5** trong Notebook (`spam_classification.ipynb`) | • Xây dựng Pipeline huấn luyện Support Vector Machine (`LinearSVC`)<br>• Áp dụng CalibratedClassifierCV để hiệu chuẩn xác suất dự đoán<br>• Tinh chỉnh siêu tham số, lưu mô hình và chọn ra `best_model.joblib` |
| **TV 6** | **Nguyễn Minh Nhật**<br>`70206006644` | **Evaluation Analyst** | **Mục 6** trong Notebook (`spam_classification.ipynb`) | • Đánh giá độc lập 3 mô hình trên tập Test: Accuracy, Precision, Recall, F1<br>• Vẽ biểu đồ nhiệt Ma trận nhầm lẫn (Confusion Matrix Heatmap)<br>• Xuất báo cáo điểm số chi tiết ra file `models/metadata.json` |
| **TV 7** | **Mai An Thịnh**<br>`052206005348` | **Deployment & Demo Engineer** | **Mục 7** trong Notebook<br>`scripts/run_predict.py`<br>`app.py` (Streamlit UI) | • Thử nghiệm dự đoán email mẫu, đo lường thời gian xử lý (Latency)<br>• Xây dựng giao diện Ô Chat dòng lệnh tương tác (`run_predict.py`)<br>• Xây dựng giao diện Web Demo Streamlit trực quan (`app.py`) |

---

## 2. KẾ HOẠCH THỰC HIỆN THEO 4 MILESTONES

### 🟢 Milestone 1: Tiền xử lý dữ liệu & Trích xuất đặc trưng (Tuần 1 - 2)
- **Nhiệm vụ:** Hoàn thành **Mục 1, Mục 2, Mục 3** trong Notebook.
- **Đầu ra:** Tập dữ liệu sạch sau tiền xử lý và bộ trích xuất đặc trưng TF-IDF + số đếm hoạt động ổn định.
- **Phụ trách:** TV1, TV2, TV3.

### 🟢 Milestone 2: Huấn luyện 3 mô hình học máy (Tuần 3)
- **Nhiệm vụ:** Hoàn thành **Mục 4, Mục 5** trong Notebook.
- **Đầu ra:** 3 mô hình (Naive Bayes, Logistic Regression, Linear SVM) được huấn luyện và lưu vào thư mục `models/`.
- **Phụ trách:** TV4, TV5.

### 🟢 Milestone 3: Đánh giá & So sánh mô hình (Tuần 4)
- **Nhiệm vụ:** Hoàn thành **Mục 6** trong Notebook.
- **Đầu ra:** Bảng đối chiếu các chỉ số đánh giá của 3 mô hình, biểu đồ ma trận nhầm lẫn và file `metadata.json`.
- **Phụ trách:** TV6.

### 🟢 Milestone 4: Thử nghiệm dự đoán, Ô Chat & Web Demo (Tuần 5)
- **Nhiệm vụ:** Hoàn thành **Mục 7** trong Notebook, `scripts/run_predict.py`, `app.py`.
- **Đầu ra:** Kiểm thử thời gian phản hồi đạt yêu cầu, giao diện Ô Chat và Web Demo sẵn sàng trình chiếu báo cáo.
- **Phụ trách:** TV7, TV1.
