# SALESPREDICT-ML: DỰ ĐOÁN DOANH SỐ BÁN HÀNG BẰNG HỌC MÁY
## PRACTICE 2: PREDICTING PRODUCT SALES — NHÓM 3 (UTH)

> **Môn học:** Học máy (Machine Learning) — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** Tiến sĩ Nguyễn Thị Khánh Tiên  
> **Bài toán:** Hồi quy dự đoán doanh số bán hàng sản phẩm (Product Sales Prediction)  
> **Kiến trúc triển khai:** 3 Notebooks Walkthrough chuyên sâu theo chuẩn của Will Koehrsen & TS. Nguyễn Thị Khánh Tiên

---

## 📌 1. GIỚI THIỆU DỰ ÁN
Hệ thống giải quyết bài toán hồi quy liên tục: Dự đoán doanh số bán hàng của các sản phẩm thương mại điện tử dựa trên giá sản phẩm, chi phí quảng cáo marketing, danh mục sản phẩm và các yếu tố mùa vụ.

Toàn bộ quy trình ML được tổ chức theo **kiến trúc 3 Notebooks tuần tự (Modular Walkthrough)**:
1. `01_data_cleaning_and_eda.ipynb`: Khám phá dữ liệu 5 bước EDA toàn diện và làm sạch dữ liệu.
2. `02_feature_engineering_and_pipeline.ipynb`: Biến đổi log, lọc đa cộng tuyến ($r > 0.6$) và đóng gói Scikit-Learn Pipeline chống rò rỉ dữ liệu.
3. `03_model_training_evaluation_and_explain.ipynb`: Huấn luyện 3 mô hình hồi quy (Linear Regression, Decision Tree, Random Forest), kiểm định chéo 5-Fold CV, đánh giá trên Test set và trích xuất độ quan trọng đặc trưng (Feature Importances).

---

## 👥 2. THÀNH VIÊN VÀ PHÂN CÔNG NHIỆM VỤ (TỔNG: 100%)

| STT | Họ và tên | Vai trò | Đóng góp | Vị trí phụ trách trong dự án / Repo |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **Nguyễn Xuân Đông** | **Trưởng nhóm** | 15% | **Notebook 1 — Mục 1:** EDA Nền tảng, Quản trị Git, Web Demo Streamlit (`app.py`) & CLI |
| 2 | *[Chưa chỉ định]* | Thành viên | 14% | **Notebook 1 — Mục 2:** EDA Chuyên sâu (Đa biến, Tương quan, Ngoại lai & Đúc kết) |
| 3 | *[Chưa chỉ định]* | Thành viên | 14% | **Notebook 1 — Mục 3:** Tiền xử lý dữ liệu (Khử trùng, Điền median/mode) |
| 4 | *[Chưa chỉ định]* | Thành viên | 14% | **Notebook 2 — Mục 4:** FE Biến đổi (Log np.log1p, Binning) & Tạo biến Domain |
| 5 | *[Chưa chỉ định]* | Thành viên | 14% | **Notebook 2 — Mục 5:** FE Lọc đa cộng tuyến, OneHot + Scaler Pipeline chống Leak |
| 6 | *[Chưa chỉ định]* | Thành viên | 14% | **Notebook 3 — Mục 6:** Huấn luyện mô hình hồi quy (Linear, Decision Tree, Random Forest) |
| 7 | *[Chưa chỉ định]* | Thành viên | 15% | **Notebook 3 — Mục 7:** Đánh giá RMSE/R2, Biểu đồ phần dư & Feature Importance |
| | **Tổng cộng** | | **100%** | **Hoàn thành toàn diện 4 tính năng hệ thống** |

### Sơ đồ luồng thực hiện tuần tự qua 3 Notebooks:
```text
[sales_data.csv]
      │
      ▼
📓 NOTEBOOK 1: 01_data_cleaning_and_eda.ipynb (Khám phá & Làm sạch dữ liệu)
   ├── TV 1 (15%): Nguyễn Xuân Đông (Trưởng nhóm) — EDA 1 (Cấu trúc), EDA 2 (Chất lượng), EDA 3 (Đơn biến)
   ├── TV 2 (14%): *[Chưa chỉ định]* — EDA 4 (Đa biến tương quan, Ngoại lai IQR) & EDA 5 (Bảng đúc kết EDA)
   └── TV 3 (14%): *[Chưa chỉ định]* — Tiền xử lý (Khử trùng, Điền missing Median/Mode) ──> [clean_product_sales.csv]
      │
      ▼
📓 NOTEBOOK 2: 02_feature_engineering_and_pipeline.ipynb (Kỹ thuật đặc trưng & Pipeline)
   ├── TV 4 (14%): *[Chưa chỉ định]* — FE Biến đổi (Log np.log1p, Binning) & Tạo biến Domain (Tỷ lệ chi tiêu, Mùa vụ)
   └── TV 5 (14%): *[Chưa chỉ định]* — Chia Train/Test, Lọc đa cộng tuyến (r > 0.6), OneHot + Scaler Pipeline
      │                             └──> [train_test_data.joblib & preprocessor.joblib]
      ▼
📓 NOTEBOOK 3: 03_model_training_evaluation_and_explain.ipynb (Huấn luyện & Đánh giá chuyên sâu)
   ├── TV 6 (14%): *[Chưa chỉ định]* — Huấn luyện 3 mô hình (Linear, Tree, Random Forest) + 5-Fold Cross Validation
   └── TV 7 (15%): *[Chưa chỉ định]* — Đánh giá Test, Residual Plot, Feature Importance (Giải thích mô hình)
      │                             └──> [best_model.joblib & metadata.json]
      ▼
🚀 TV 1: NGUYỄN XUÂN ĐÔNG TRIỂN KHAI WEB DEMO STREAMLIT (app.py) & CLI TEST (scripts/run_predict.py)
```

---

## 🚀 3. HƯỚNG DẪN KHỞI CHẠY (QUICK START)

### Bước 1: Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy tuần tự 3 Notebooks
Mở và thực thi các Notebook theo đúng thứ tự chuyển giao dữ liệu:
1. **Chạy Notebook 1:** Khám phá dữ liệu và sinh ra tệp `data/processed/clean_product_sales.csv`.
   ```bash
   jupyter notebook notebooks/01_data_cleaning_and_eda.ipynb
   ```
2. **Chạy Notebook 2:** Tạo đặc trưng, chia Train/Test, lọc đa cộng tuyến và lưu Pipeline.
   ```bash
   jupyter notebook notebooks/02_feature_engineering_and_pipeline.ipynb
   ```
3. **Chạy Notebook 3:** Huấn luyện mô hình, kiểm định chéo 5-Fold, đánh giá Test và xuất `best_model.joblib`.
   ```bash
   jupyter notebook notebooks/03_model_training_evaluation_and_explain.ipynb
   ```

### Bước 3: Khởi chạy Ứng dụng Web Demo Streamlit
```bash
streamlit run app.py
```

---

## 📂 4. CẤU TRÚC THƯ MỤC CHUẨN
```text
PRACTICE 2_PREDICTING_PRODUCT_SALES/
├── data/
│   ├── raw/sales_data.csv                    # Dữ liệu nguồn về doanh số bán hàng
│   └── processed/clean_product_sales.csv     # Dữ liệu sạch sau tiền xử lý (Output NB1)
├── docs/                                     # Bộ tài liệu đặc tả dự án
│   ├── project_description.md                # 7 mục đặc tả dự án chuẩn (dưới 200 từ)
│   ├── project_design.md                     # Thiết kế hệ thống theo 4 tính năng (dưới 200 từ)
│   └── TASKS.md                              # Phân công 7 thành viên theo 3 Notebooks
├── models/                                   # Nơi lưu trữ pipeline và mô hình
│   ├── preprocessor.joblib                   # Pipeline tiền xử lý đã fit trên tập Train (Output NB2)
│   ├── best_model.joblib                     # Mô hình tối ưu nhất dùng cho web app (Output NB3)
│   ├── linear_regression.joblib              # Mô hình 1: Linear Regression
│   ├── decision_tree.joblib                  # Mô hình 2: Decision Tree Regressor
│   ├── random_forest.joblib                  # Mô hình 3: Random Forest Regressor
│   └── metadata.json                         # Kết quả đối sánh MSE, RMSE, MAE, R2
├── notebooks/                                # KHÔNG GIAN 3 NOTEBOOKS WALKTHROUGH
│   ├── 01_data_cleaning_and_eda.ipynb        # [TV1, TV2, TV3] Khám phá & Làm sạch
│   ├── 02_feature_engineering_and_pipeline.ipynb # [TV4, TV5] Tạo biến & Pipeline
│   └── 03_model_training_evaluation_and_explain.ipynb # [TV6, TV7] Huấn luyện, Đánh giá & Giải thích
├── scripts/
│   └── run_predict.py                        # [TV1] Ô Chat dòng lệnh tương tác suy luận nhanh
├── app.py                                    # [TV1] Web Demo Streamlit trực quan dự đoán doanh số
├── requirements.txt                          # Danh sách thư viện cần thiết
└── README.md                                 # Hướng dẫn tổng quan bài thực hành (file này)
```
