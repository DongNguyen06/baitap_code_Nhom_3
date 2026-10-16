# KẾ HOẠCH PHÂN RÃ CÔNG VIỆC (TASKS BREAKDOWN) — SALESPREDICT-ML
## PRACTICE 2: PREDICTING PRODUCT SALES — NHÓM 3 (UTH)

> **Nguyên tắc:** Phân công bám sát dòng chảy khoa học dữ liệu thẳng tắp cho bài toán Hồi quy (EDA toàn diện $\rightarrow$ Làm sạch $\rightarrow$ Kỹ thuật đặc trưng $\rightarrow$ Huấn luyện $\rightarrow$ Đánh giá & Triển khai) và thực hiện qua **3 Notebooks chuyên sâu tuần tự** theo chuẩn *Machine Learning Walkthrough* của TS. Nguyễn Thị Khánh Tiên.

---

## SƠ ĐỒ LUỒNG THỰC HIỆN TUẦN TỰ (3 NOTEBOOKS PIPELINE)

```text
[sales_data.csv]
      │
      ▼
📓 NOTEBOOK 1: 01_data_cleaning_and_eda.ipynb (Khám phá & Làm sạch dữ liệu)
   ├── Nguyễn Xuân Đông (Trưởng nhóm - 15%): EDA 1 (Cấu trúc), EDA 2 (Chất lượng), EDA 3 (Đơn biến)
   ├── Thành viên (14%): *[Chưa chỉ định]* — EDA 4 (Đa biến tương quan, Ngoại lai IQR) & EDA 5 (Bảng đúc kết EDA)
   └── Thành viên (14%): *[Chưa chỉ định]* — Tiền xử lý (Khử trùng, Điền missing Median/Mode) ──> [clean_product_sales.csv]
      │
      ▼
📓 NOTEBOOK 2: 02_feature_engineering_and_pipeline.ipynb (Kỹ thuật đặc trưng & Pipeline)
   ├── Thành viên (14%): *[Chưa chỉ định]* — FE Biến đổi (Log np.log1p, Binning) & Tạo biến Domain (Tỷ lệ chi tiêu, Mùa vụ)
   └── Thành viên (14%): *[Chưa chỉ định]* — Chia Train/Test, Lọc đa cộng tuyến (r > 0.6), OneHot + Scaler Pipeline
      │                             └──> [train_test_data.joblib & preprocessor.joblib]
      ▼
📓 NOTEBOOK 3: 03_model_training_evaluation_and_explain.ipynb (Huấn luyện & Đánh giá chuyên sâu)
   ├── Thành viên (14%): *[Chưa chỉ định]* — Huấn luyện 3 mô hình (Linear, Tree, Random Forest) + 5-Fold Cross Validation
   └── Thành viên (15%): *[Chưa chỉ định]* — Đánh giá Test, Residual Plot, Feature Importance (Giải thích mô hình)
      │                             └──> [best_model.joblib & metadata.json]
      ▼
🚀 NGUYỄN XUÂN ĐÔNG (TRƯỞNG NHÓM): TRIỂN KHAI WEB DEMO STREAMLIT (app.py) & CLI TEST (scripts/run_predict.py)
```

---

## BẢNG PHÂN CÔNG NHIỆM VỤ & MỨC ĐỘ ĐÓNG GÓP (TỔNG: 100%)

| STT | Thành viên | Vai trò | Đóng góp | Tính năng (Design) | Vị trí thực hiện | Tóm tắt nhiệm vụ chính |
| :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| **1** | **Nguyễn Xuân Đông** | **Trưởng nhóm** | **15%** | **Tính năng 1 & 4:**<br>EDA Nền tảng & Web Demo | **Notebook 1 — Mục 1**<br>kèm `app.py` & CLI | • Khởi tạo dự án, thiết lập môi trường và cấu hình `RANDOM_SEED = 42`.<br>• Khảo sát cấu trúc tổng quan dữ liệu thô (`shape`, `info`, `describe`).<br>• Đánh giá chất lượng dữ liệu: Thống kê tỷ lệ khuyết thiếu và dữ liệu trùng lặp.<br>• Phân tích đơn biến toàn diện: Phân phối doanh số bán hàng và các biến số độc lập.<br>• Quản trị phân nhánh Git, rà soát mã nguồn và điều phối tiến độ toàn nhóm.<br>• **Xây dựng và hoàn thiện ứng dụng Web Demo Streamlit (`app.py`) và công cụ dòng lệnh CLI (`scripts/run_predict.py`) phục vụ người dùng cuối.** |
| **2** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 1:**<br>EDA Chuyên sâu | **Notebook 1 — Mục 2** | • Phân tích đa biến: Đo lường tương quan Pearson (Heatmap) và biểu đồ phân tán (Scatter plot).<br>• Phân tích phân phối độ lệch của doanh số và nhận diện hiện tượng đuôi dài.<br>• Xác định giá trị ngoại lai theo công thức khoảng tứ phân vị IQR ($1.5 \times IQR$) và Boxplot.<br>• Đúc kết các bằng chứng EDA thành Bảng kế hoạch hành động định hướng cho các bước tiếp theo. |
| **3** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 1:**<br>Làm sạch dữ liệu | **Notebook 1 — Mục 3** | • Thực thi xử lý chất lượng: Khử trùng lặp hoàn toàn các dòng dữ liệu giao dịch.<br>• Xử lý giá trị khuyết: Điền trung vị (Median) cho các biến số lệch và điền mode cho biến danh mục.<br>• Kiểm tra tính toàn vẹn (đảm bảo 0 ô khuyết).<br>• Xuất dữ liệu sạch sang `data/processed/clean_product_sales.csv` bàn giao cho Notebook 2. |
| **4** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 2:**<br>FE Biến đổi & Domain | **Notebook 2 — Mục 4** | • Kỹ thuật biến đổi đặc trưng: Áp dụng biến đổi log `np.log1p` để nén độ lệch phân phối của giá và chi phí.<br>• Rời rạc hóa (Binning): Gom nhóm các biến liên tục như khoảng giá thành các nhóm thứ bậc.<br>• Tạo đặc trưng mới theo ngữ cảnh: Tỷ lệ chi tiêu tiếp thị / giá, chỉ số mùa vụ và phân khúc sản phẩm.<br>• Viết nhận xét đối sánh phân phối trước và sau biến đổi log. |
| **5** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 2:**<br>FE Mã hóa, Lọc & Pipeline | **Notebook 2 — Mục 5** | • Thiết lập hàng rào phân chia Train/Test 80/20 nghiêm ngặt chống rò rỉ dữ liệu (Anti-Leakage).<br>• **Kỹ thuật Cô Tiên:** Lập trình hàm lọc và loại bỏ đặc trưng đa cộng tuyến ($r > 0.6$).<br>• Đóng gói tiền xử lý trong `ColumnTransformer`: OneHotEncoder cho danh mục và StandardScaler cho biến số.<br>• Lưu trữ tập dữ liệu và đối tượng `preprocessor.joblib` bàn giao cho Notebook 3. |
| **6** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 3:**<br>Huấn luyện mô hình | **Notebook 3 — Mục 6** | • Xây dựng 3 mô hình hồi quy hoàn chỉnh: Linear Regression (Baseline), Decision Tree Regressor và Random Forest Regressor.<br>• Áp dụng kiểm định chéo **5-Fold Cross-Validation** trên tập Train để đo lường độ ổn định.<br>• Tinh chỉnh siêu tham số cơ bản bằng `GridSearchCV` cho Random Forest.<br>• Lưu trữ các mô hình đã huấn luyện vào thư mục `models/`. |
| **7** | *[Chưa chỉ định]* | Thành viên | **15%** | **Tính năng 4:**<br>Đánh giá & Phân tích lỗi | **Notebook 3 — Mục 7** | • Đánh giá độc lập trên tập kiểm thử Test set qua 4 chỉ số: MSE, RMSE, MAE và hệ số $R^2$.<br>• Trực quan hóa biểu đồ Thực tế vs Dự đoán và biểu đồ phân tán phần dư (Residual Plot).<br>• **Kỹ thuật Cô Tiên:** Trích xuất và trực quan hóa Top 10 đặc trưng quan trọng nhất (Feature Importances).<br>• Xuất mô hình tối ưu `best_model.joblib` và tệp `metadata.json` bàn giao cho Trưởng nhóm (Nguyễn Xuân Đông) tích hợp ứng dụng. |
| **TỔNG** | **7 thành viên** | | **100%** | | | **Hoàn thành toàn diện 4 tính năng hệ thống qua 3 Notebooks** |
