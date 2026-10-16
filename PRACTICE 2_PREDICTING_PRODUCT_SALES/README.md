# SALESPREDICT-ML: DỰ ĐOÁN DOANH SỐ BÁN HÀNG BẰNG HỌC MÁY
## PRACTICE 2: PREDICTING PRODUCT SALES — NHÓM 3 (UTH)

> **Môn học:** Học máy (Machine Learning) — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** Tiến sĩ Nguyễn Thị Khánh Tiên  
> **Bài toán:** Hồi quy dự đoán doanh số bán hàng sản phẩm (Product Sales Prediction)

---

## 📌 1. GIỚI THIỆU DỰ ÁN
Hệ thống giải quyết bài toán hồi quy liên tục: Dự đoán doanh số bán hàng của các dòng sản phẩm thương mại điện tử dựa trên thông tin sản phẩm, nhân khẩu học khách hàng, lịch sử mua sắm, chi phí quảng cáo marketing và các yếu tố mùa vụ.

Dự án áp dụng phương pháp luận **Vibe Coding** với AI: Toàn bộ quy trình ML từ 5 bước EDA toàn diện, tiền xử lý dữ liệu, 5 trụ cột Feature Engineering (chống rò rỉ dữ liệu) đến huấn luyện 3 mô hình hồi quy cốt lõi (**Linear Regression, Decision Tree Regressor, Random Forest Regressor**) được tích hợp trong **1 file Jupyter Notebook duy nhất** (`notebooks/product_sales_prediction.ipynb`). Nhóm phối hợp theo quy trình Git nhánh gối đầu, xóa output trước khi commit để không bao giờ bị xung đột mã nguồn.

---

## 👥 2. THÀNH VIÊN VÀ PHÂN CÔNG NHIỆM VỤ

| STT | Họ và tên | Vai trò | Đóng góp | Phần phụ trách trong Notebook / Repo |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **Nguyễn Xuân Đông** | **Trưởng nhóm** | 15% | Mục 1: EDA Nền tảng (Cấu trúc, Chất lượng & Đơn biến), Quản trị Git |
| 2 | *[Chưa chỉ định]* | Thành viên | 14% | Mục 2: EDA Chuyên sâu (Đa biến, Tương quan, Ngoại lai & Kết luận) |
| 3 | *[Chưa chỉ định]* | Thành viên | 14% | Mục 3: Tiền xử lý dữ liệu (Khử trùng, Điền missing trung vị/mode) |
| 4 | *[Chưa chỉ định]* | Thành viên | 14% | Mục 4: FE Biến đổi (Log, Binning) & Tạo biến Domain (Mùa vụ, Vòng đời) |
| 5 | *[Chưa chỉ định]* | Thành viên | 14% | Mục 5: FE Mã hóa, Chuẩn hóa Scaling, Lọc đặc trưng & Pipeline |
| 6 | *[Chưa chỉ định]* | Thành viên | 14% | Mục 6: Huấn luyện mô hình hồi quy (Linear, Decision Tree, Random Forest) |
| 7 | *[Chưa chỉ định]* | Thành viên | 15% | Mục 7: Đánh giá RMSE/R2, Biểu đồ phần dư & Web Demo Streamlit |
| | **Tổng cộng** | | **100%** | |

### Sơ đồ luồng thực hiện tuần tự (Pipeline Workflow):
```text
[Sales_Data.csv] ──> TV 1 (EDA Nền tảng: Cấu trúc, Chất lượng & Đơn biến Doanh số)
                 ──> TV 2 (EDA Chuyên sâu: Đa biến, Tương quan, Ngoại lai & Kết luận)
                 ──> TV 3 (Làm sạch dữ liệu: Khử trùng lặp, Điền khuyết theo trung vị)
                 ──> TV 4 (FE Biến đổi & Domain: Log, Binning, Mùa vụ & Vòng đời)
                 ──> TV 5 (FE Mã hóa, Chuẩn hóa & Lọc F-regression trong Pipeline)
                 ──> TV 6 (Huấn luyện các mô hình hồi quy: Linear, Tree, Random Forest)
                 ──> TV 7 (Đánh giá RMSE/R2, Biểu đồ phần dư & Web Demo Streamlit)
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
jupyter notebook notebooks/product_sales_prediction.ipynb
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
PRACTICE 2_PREDICTING_PRODUCT_SALES/
├── data/
│   ├── raw/                                  # Dữ liệu nguồn về doanh số bán hàng
│   └── processed/clean_product_sales.csv     # Dữ liệu sạch sau tiền xử lý
├── docs/                                     # Bộ tài liệu đặc tả dự án
│   ├── project_description.md                # 7 mục đặc tả dự án chuẩn (dưới 200 từ)
│   ├── project_design.md                     # Thiết kế hệ thống theo 4 tính năng (dưới 200 từ)
│   └── TASKS.md                              # Phân công 7 thành viên bám sát 4 tính năng
├── models/                                   # Nơi lưu trữ các mô hình sau huấn luyện
│   ├── best_model.joblib                     # Mô hình tối ưu nhất dùng cho web app
│   ├── linear_regression.joblib              # Mô hình 1: Linear Regression
│   ├── decision_tree.joblib                  # Mô hình 2: Decision Tree Regressor
│   ├── random_forest.joblib                  # Mô hình 3: Random Forest Regressor
│   └── metadata.json                         # Kết quả đánh giá MSE, RMSE, MAE, R2
├── notebooks/                                # KHÔNG GIAN LÀM VIỆC CHÍNH CỦA NHÓM
│   └── product_sales_prediction.ipynb        # 1 Notebook duy nhất chứa toàn bộ 7 mục (khung sườn)
├── scripts/
│   └── run_predict.py                        # [TV7] Ô Chat dòng lệnh tương tác
├── app.py                                    # [TV7] Web Demo Streamlit trực quan dự đoán doanh số
├── requirements.txt                          # Danh sách thư viện cần thiết
└── README.md                                 # Hướng dẫn tổng quan bài thực hành (file này)
```
