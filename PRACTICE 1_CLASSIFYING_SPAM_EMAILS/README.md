# SPAMGUARD-ML: PHÂN LOẠI EMAIL SPAM BẰNG HỌC MÁY
## PRACTICE 1: CLASSIFYING SPAM EMAILS — NHÓM 3 (UTH)

> **Môn học:** Học máy (Machine Learning) — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** Tiến sĩ Nguyễn Thị Khánh Tiên  
> **Tập dữ liệu:** `data/raw/CEAS_08.csv` (39.154 email thực tế từ hội thảo quốc tế CEAS)

---

## 📌 1. GIỚI THIỆU DỰ ÁN
Hệ thống giải quyết bài toán phân loại email nhị phân: Tự động phát hiện email là **Spam** (Thư rác - nhãn 1) hay **Ham** (Hợp lệ - nhãn 0) dựa trên nội dung văn bản tổng hợp từ tiêu đề và thân email.

Dự án áp dụng phương pháp luận **Vibe Coding** với AI: Toàn bộ quy trình ML từ tiền xử lý đến đánh giá 3 mô hình (Naive Bayes, Logistic Regression, Linear SVM) được tích hợp trong **1 file Jupyter Notebook duy nhất** (`notebooks/spam_classification.ipynb`). Nhóm phối hợp theo quy trình Git nhánh gối đầu, xóa output trước khi commit để không bao giờ bị xung đột mã nguồn.

---

## 👥 2. THÀNH VIÊN VÀ PHÂN CÔNG NHIỆM VỤ

| STT | Họ và tên | MSSV | Vai trò | Phần phụ trách trong Notebook / Repo |
| :---: | :--- | :---: | :--- | :--- |
| 1 | **Nguyễn Xuân Đông** | `94206000207` | **Trưởng nhóm** | Mục 1 (Cấu hình) & Mục 7 (Demo), Quản trị Git |
| 2 | **Mai Hoàng Danh** | `066206005311` | Data Cleaning | Mục 2: Tiền xử lý & Làm sạch dữ liệu văn bản |
| 3 | **Trương Thành Công** | `87206002694` | Feature Engineer | Mục 3: Kỹ thuật đặc trưng (TF-IDF & Đặc trưng số) |
| 4 | **Võ Khôi Nguyên** | `79206001370` | ML Engineer 1 | Mục 4: Huấn luyện Naive Bayes & Logistic Regression |
| 5 | **Võ Thanh Phú** | `89206003078` | ML Engineer 2 | Mục 5: Huấn luyện Support Vector Machine (Linear SVM) |
| 6 | **Nguyễn Minh Nhật** | `70206006644` | Evaluation Analyst | Mục 6: Đánh giá & So sánh 3 mô hình, vẽ Confusion Matrix |
| 7 | **Mai An Thịnh** | `052206005348` | Deployment & Demo | Mục 7: Ô Chat tương tác (`run_predict.py`), Web (`app.py`) |

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
│   ├── DEFINE.md                             # 7 mục đặc tả chuẩn theo vở ghi
│   ├── DESIGN.md                             # Thiết kế kiến trúc & luồng dữ liệu
│   ├── TASKS.md                              # Phân công 7 thành viên theo các mục của Notebook
│   └── TEAM_WORKFLOW.md                      # 4 nguyên tắc Vibe Coding & Git nhánh gối đầu
├── models/                                   # Nơi lưu trữ 3 mô hình sau huấn luyện
│   ├── best_model.joblib                     # Mô hình tối ưu nhất dùng cho ô chat/web
│   ├── naive_bayes.joblib                    # Mô hình 1: Naive Bayes
│   ├── logistic_reg.joblib                   # Mô hình 2: Logistic Regression
│   ├── svm.joblib                            # Mô hình 3: Linear Support Vector Machine
│   └── metadata.json                         # Kết quả đánh giá các chỉ số & ma trận nhầm lẫn
├── notebooks/                                # KHÔNG GIAN LÀM VIỆC CHÍNH CỦA NHÓM
│   └── spam_classification.ipynb             # 1 Notebook duy nhất chứa toàn bộ 7 mục (khung sườn)
├── scripts/
│   └── run_predict.py                        # [TV7] Ô Chat dòng lệnh tương tác (đang để trống)
├── app.py                                    # [TV7] Web Demo Streamlit trực quan 3 cột (đang để trống)
├── requirements.txt                          # Danh sách thư viện cần thiết
└── README.md                                 # Hướng dẫn tổng quan bài thực hành (file này)
```
