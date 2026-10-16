# SPAMGUARD-ML: PHÂN LOẠI EMAIL SPAM BẰNG HỌC MÁY
## PRACTICE 1: CLASSIFYING SPAM EMAILS — NHÓM 3 (UTH)

> **Môn học:** Học máy (Machine Learning) — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** Tiến sĩ Nguyễn Thị Khánh Tiên  
> **Tập dữ liệu:** `data/raw/CEAS_08.csv` (39.154 email thực tế từ hội thảo quốc tế CEAS)

---

## 📌 1. GIỚI THIỆU DỰ ÁN
Hệ thống giải quyết bài toán phân loại email nhị phân: Tự động phát hiện email là **Spam** (Thư rác - nhãn 1) hay **Ham** (Hợp lệ - nhãn 0) dựa trên nội dung văn bản tổng hợp từ tiêu đề và thân email.

Dự án áp dụng phương pháp luận **Vibe Coding** với AI: Toàn bộ quy trình ML đi từ 5 bước EDA toàn diện, thực thi làm sạch dữ liệu, 5 trụ cột Feature Engineering (chống rò rỉ dữ liệu) đến huấn luyện 3 mô hình cốt lõi (**Naive Bayes, Logistic Regression, Linear Support Vector Machine**) được tích hợp trong **1 file Jupyter Notebook duy nhất** (`notebooks/spam_classification.ipynb`). Nhóm phối hợp theo quy trình Git nhánh gối đầu, xóa output trước khi commit để không bao giờ bị xung đột mã nguồn.

---

## 👥 2. THÀNH VIÊN VÀ PHÂN CÔNG NHIỆM VỤ

| STT | Họ và tên | Vai trò | Đóng góp | Phần phụ trách trong Notebook / Repo |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **Nguyễn Xuân Đông** | **Trưởng nhóm** | 15% | Mục 1: EDA Nền tảng (Cấu trúc, Chất lượng & Đơn biến), Quản trị Git |
| 2 | *[Chưa chỉ định]* | Thành viên | 14% | Mục 2: EDA Chuyên sâu (Đa biến, Ngoại lai & Đúc kết định hướng) |
| 3 | *[Chưa chỉ định]* | Thành viên | 14% | Mục 3: Tiền xử lý dữ liệu & Làm sạch văn bản |
| 4 | *[Chưa chỉ định]* | Thành viên | 14% | Mục 4: FE Biến đổi (Log, Binning) & Tạo biến Domain |
| 5 | *[Chưa chỉ định]* | Thành viên | 14% | Mục 5: FE Trích xuất (TF-IDF), Lọc Chi-square & Pipeline chống Leak |
| 6 | *[Chưa chỉ định]* | Thành viên | 14% | Mục 6: Huấn luyện 3 mô hình (Naive Bayes, Logistic, Linear SVM) |
| 7 | *[Chưa chỉ định]* | Thành viên | 15% | Mục 7: Đánh giá 3 mô hình, Phân tích lỗi & Triển khai Web Demo |
| | **Tổng cộng** | | **100%** | |

### Sơ đồ luồng thực hiện tuần tự (Pipeline Workflow):
```text
[CEAS_08.csv] ──> TV 1 (EDA Nền tảng: Cấu trúc, Chất lượng & Đơn biến)
              ──> TV 2 (EDA Chuyên sâu: Đa biến, Ngoại lai & Đúc kết định hướng)
              ──> TV 3 (Làm sạch dữ liệu: Khử trùng, điền missing, clean_text)
              ──> TV 4 (FE Biến đổi: Log, Binning, Tạo biến Domain, Mã hóa)
              ──> TV 5 (FE Lọc & Pipeline: Chia Train/Test chống Leak, TF-IDF, Chi2)
              ──> TV 6 (Huấn luyện 3 mô hình: Naive Bayes, Logistic L1, Linear SVM)
              ──> TV 7 (Đánh giá kiểm thử, Phân tích lỗi & Web Demo Streamlit)
```

---

## 🚀 3. HƯỚNG DẪN KHỞI CHẠY (QUICK START)

### Bước 1: Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
pip install -r requirements.txt
```

### Bước 2: Mở và thực thi Jupyter Notebook
Khởi chạy Jupyter Notebook hoặc mở trong VS Code để thực hiện các mục theo phân công:
```bash
jupyter notebook notebooks/spam_classification.ipynb
```

### Bước 3: Mở Ô Chat thử nghiệm hoặc Web Demo
- **Ô Chat tương tác trên dòng lệnh (Console Chat):**
```bash
python scripts/run_predict.py
```
- **Web Demo giao diện Streamlit trực quan:**
```bash
streamlit run app.py
```

---

## 📂 4. CẤU TRÚC THƯ MỤC CHUẨN
```text
PRACTICE 1_CLASSIFYING_SPAM_EMAILS/
├── data/
│   ├── raw/CEAS_08.csv                       # Dữ liệu nguồn (39.154 email, 7 cột)
│   └── processed/clean_emails.csv            # Dữ liệu sạch (sau khi bóc HTML, bỏ trùng lặp)
├── docs/                                     # Bộ tài liệu đặc tả dự án
│   ├── project_description.md                # 7 mục đặc tả dự án chuẩn (dưới 200 từ)
│   ├── project_design.md                     # Thiết kế hệ thống theo 4 tính năng (dưới 200 từ)
│   └── TASKS.md                              # Phân công 7 thành viên bám sát 4 tính năng
├── models/                                   # Nơi lưu trữ các mô hình sau huấn luyện
│   ├── best_model.joblib                     # Mô hình tối ưu nhất dùng cho ô chat/web
│   ├── naive_bayes.joblib                    # Mô hình 1: Naive Bayes
│   ├── logistic_reg.joblib                   # Mô hình 2: Logistic Regression (L1 Lasso)
│   ├── svm.joblib                            # Mô hình 3: Linear Support Vector Machine (Platt Scaling)
│   └── metadata.json                         # Kết quả đánh giá các chỉ số & ma trận nhầm lẫn
├── notebooks/                                # KHÔNG GIAN LÀM VIỆC CHÍNH CỦA NHÓM
│   └── spam_classification.ipynb             # 1 Notebook duy nhất chứa toàn bộ 7 mục (khung sườn)
├── scripts/
│   └── run_predict.py                        # [TV7] Ô Chat dòng lệnh tương tác
├── app.py                                    # [TV7] Web Demo Streamlit trực quan 3 cột
├── requirements.txt                          # Danh sách thư viện cần thiết
└── README.md                                 # Hướng dẫn tổng quan bài thực hành (file này)
```
