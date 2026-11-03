# BÀI THỰC HÀNH 2: DỰ ĐOÁN DOANH SỐ SẢN PHẨM — NHÓM 3
## PRACTICE 2: PREDICTING PRODUCT SALES — GROUP 3

> **Môn học:** Học máy (Machine Learning) — UTH  
> **Bài toán:** Hồi quy dự đoán doanh số bán hàng sản phẩm ($y = \text{Amount} \in \mathbb{R}$, đơn vị INR)  
> **Kiến trúc:** 4 Jupyter Notebooks, 3 thuật toán Scratch tự viết bằng NumPy thuần túy và Pipeline tự động hóa khép kín.

---

## 📌 1. GIỚI THIỆU DỰ ÁN
Dự án giải quyết bài toán hồi quy dự đoán doanh thu bán hàng thương mại điện tử bằng Học máy trên bộ dữ liệu `Amazon Sale Report.csv` (128.975 dòng thô $\rightarrow$ 102.852 dòng sạch sau khi lọc đơn hủy, đơn lỗi và khử trùng lặp).

Quy trình thực hiện được chia thành 4 Notebooks chuyên sâu:
1. `notebooks/01_data_cleaning_and_eda.ipynb`: Khám phá dữ liệu toàn diện (EDA), phân tích phân phối lệch phải của Amount, biến đổi Log, quét ngoại lai IQR và làm sạch dữ liệu (`clean_amazon_sales.csv`).
2. `notebooks/02_feature_engineering_and_pipeline.ipynb`: Trích xuất đặc trưng thời gian mùa vụ, gom nhóm địa lý Top 10 bang, tạo cờ khuyến mãi/B2B, lọc đa cộng tuyến ($|r| > 0.6$), chia Train/Test 80/20 và thiết lập `ColumnTransformer`.
3. `notebooks/03_model_training_evaluation_and_explain.ipynb`: Huấn luyện 3 mô hình Scratch, kiểm định chéo 5-Fold Cross Validation trên 82.281 mẫu Train, đánh giá độc lập trên 20.571 mẫu Test set và trích xuất Top 15 đặc trưng quan trọng.
4. `notebooks/demo.ipynb`: Giao diện thử nghiệm suy luận doanh số tương tác thời gian thực kèm cơ chế phòng vệ Guardrails.

> **Phân công nhiệm vụ:** Chi tiết công việc của 7 thành viên được quản lý tại [docs/description/TASKS.md](docs/description/TASKS.md).  
> **Thiết kế chi tiết 5W1H:** Xem tại [docs/description/project_design.md](docs/description/project_design.md).

---

## 👥 2. THÀNH VIÊN VÀ PHÂN CÔNG ĐÓNG GÓP

| STT | Thành viên | Vai trò | Module / Tệp phụ trách | Sản phẩm nghiệm thu chính |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Nguyễn Xuân Đông** | **Trưởng nhóm** | `data/scripts/eda.py`<br>`notebooks/demo.ipynb`<br>`scripts/test_all_components.py` | Quản trị Git, cấu trúc dự án, đơn biến Amount, Demo suy luận và bộ test tự động. |
| **2** | **Mai An Thịnh** | Thành viên | `data/scripts/eda.py`<br>`notebooks/01_data_cleaning_and_eda.ipynb` | Phân tích đa biến, tương quan địa lý/mùa vụ, quét ngoại lai IQR và Bảng kết luận EDA. |
| **3** | **Nguyễn Minh Nhật** | Thành viên | `data/scripts/data_preprocessing.py`<br>`notebooks/01_data_cleaning_and_eda.ipynb` | Lọc đơn hủy, khử rò rỉ dữ liệu, xử lý missing và xuất `clean_amazon_sales.csv` (102.852 dòng). |
| **4** | **Trương Thành Công** | Thành viên | `scripts/feature_engineering.py`<br>`notebooks/02_feature_engineering_and_pipeline.ipynb` | Trích xuất đặc trưng mùa vụ (`Date`), cờ khuyến mãi (`is_promoted`), khách sỉ (`is_b2b`), Top 10 bang. |
| **5** | **Võ Thanh Phú** | Thành viên | `scripts/feature_engineering.py`<br>`notebooks/02_feature_engineering_and_pipeline.ipynb` | Tách Train/Test 80/20, quét đa cộng tuyến ($|r| > 0.6$), `ColumnTransformer` (Scaler + OneHot). |
| **6** | **Võ Khôi Nguyên** | Thành viên | `scripts/model_training.py`<br>`notebooks/03_model_training_evaluation_and_explain.ipynb` | Đóng gói Pipeline hồi quy, kiểm định chéo 5-Fold Cross Validation trên tập Train. |
| **7** | **Mai Hoàng Danh** | Thành viên | `scripts/model_evaluation.py`<br>`notebooks/03_model_training_evaluation_and_explain.ipynb` | Đánh giá độc lập 20.571 mẫu Test set (nghịch đảo expm1), Residual Plot, Feature Importances. |

---

## 📊 3. BẢNG KẾT QUẢ NGHIỆM THU (TEST SET: 20.571 MẪU)

| Thuật toán (Scratch NumPy) | $R^2$ (Log) | $R^2$ (Thực tế) | RMSE (INR) | MAE (INR) | Đánh giá & Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 **Decision Tree Regressor** | **0.5623** | **0.4901** | **195.77** | **139.53** | **Quán quân (Best Model)** — Tối ưu hóa phân vị nhanh, giải thích rõ ràng |
| 🥈 **Random Forest Regressor** | 0.5464 | 0.4554 | 202.32 | 141.61 | Học kết hợp Bagging, ổn định phương sai |
| 🥉 **Linear Regression** | 0.5566 | -0.7194 | 359.49 | 142.00 | Mô hình đối chứng cơ sở (OLS Normal Equation) |

---

## 🚀 4. HƯỚNG DẪN KHỞI CHẠY (QUICK START)

### Bước 1: Cài đặt thư viện phụ thuộc
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
pip install -r requirements.txt
```

### Bước 2: Thực thi dự án
Có 2 phương thức chạy linh hoạt:

* **Cách A — Chạy tự động toàn bộ Pipeline (Khuyến nghị cho CLI):**
  ```bash
  python scripts/run_pipeline.py
  ```

* **Cách B — Mở và chạy từng Notebook (Khuyến nghị cho nghiên cứu trực quan):**
  1. `jupyter notebook notebooks/01_data_cleaning_and_eda.ipynb`
  2. `jupyter notebook notebooks/02_feature_engineering_and_pipeline.ipynb`
  3. `jupyter notebook notebooks/03_model_training_evaluation_and_explain.ipynb`

### Bước 3: Thử nghiệm tương tác với Demo
Mở notebook demo để nhập đơn hàng bất kỳ và kiểm tra khả năng dự đoán doanh số tức thì:
```bash
jupyter notebook notebooks/demo.ipynb
```

### Bước 4: Chạy bộ kiểm thử tự động toàn diện
Hỗ trợ kiểm tra tính toàn vẹn cú pháp và logic của toàn bộ 8 nhóm thành phần trong dự án:

* **Chạy kiểm thử riêng từng bước theo nhiệm vụ từng thành viên (1–8):**
  ```bash
  python scripts/test_all_components.py --step 1   # Tiền xử lý (Nguyễn Minh Nhật)
  python scripts/test_all_components.py --step 4   # 3 thuật toán Scratch NumPy (Đã hoàn thiện)
  ```
* **Chạy kiểm thử toàn bộ hệ thống (Kiểm tra tích hợp):**
  ```bash
  python scripts/test_all_components.py
  ```

---

## 📁 5. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
PRACTICE 2_PREDICTING_PRODUCT_SALES/
├── data/
│   ├── raw/                      # Dữ liệu gốc (Amazon Sale Report.csv, .zip)
│   ├── processed/                # Dữ liệu sạch (clean_amazon_sales.csv, train.csv, test.csv, train_test_data.joblib)
│   └── scripts/                  # Module tiền xử lý (data_preprocessing.py, eda.py)
├── docs/
│   ├── description/              # Tài liệu thiết kế (project_description.md, project_design.md, TASKS.md)
│   ├── algorithms/               # Lý thuyết 3 thuật toán Scratch (linear_regression.md, decision_tree_regressor.md, ...)
│   ├── images/                   # Biểu đồ trực quan hóa EDA và kết quả mô hình
│   └── report/                   # Báo cáo kỹ thuật độc lập
├── models/
│   ├── algorithms/               # Mã nguồn 3 thuật toán Scratch (linear_regression.py, decision_tree.py, random_forest.py)
│   └── saved_models/             # Mô hình tối ưu đã lưu (best_model.joblib, preprocessor.joblib, metadata.json)
├── notebooks/                    # 4 Notebooks thực hành tuần tự và Demo
├── scripts/                      # Các kịch bản chạy tự động (run_pipeline.py, feature_engineering.py, test_all_components.py)
├── README.md
└── requirements.txt
```
