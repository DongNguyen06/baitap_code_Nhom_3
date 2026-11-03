# ĐẶC TẢ BÀI THỰC HÀNH 2: DỰ ĐOÁN DOANH SỐ SẢN PHẨM — NHÓM 3

## 1. Project Name
Dự đoán Doanh số Bán hàng Thương mại Điện tử (E-Commerce Product Sales Prediction)

## 2. Problem
Dự đoán giá trị doanh số thực tế ($y = \text{Amount} \in \mathbb{R}$, đơn vị INR) của từng đơn hàng dựa trên thuộc tính sản phẩm, số lượng, phương thức vận chuyển và yếu tố thời gian địa lý.
- **Dữ liệu thực nghiệm:** `data/raw/Amazon Sale Report.csv` (128.975 giao dịch ban đầu $\rightarrow$ 102.852 đơn hàng hợp lệ sau khi làm sạch, loại đơn hủy và khử trùng lặp).
- **Phạm vi bài lab:** Tiền xử lý, kỹ thuật đặc trưng chống rò rỉ dữ liệu, tự cài đặt 3 thuật toán hồi quy Scratch thuần NumPy, huấn luyện kiểm định 5-Fold CV, đánh giá trên tập kiểm tra độc lập và demo suy luận thời gian thực.

## 3. Target User
- Giảng viên và sinh viên bộ môn Học máy (đánh giá năng lực tự cài đặt thuật toán và xây dựng pipeline hồi quy).
-  Về ý tưởng ứng dụng: Nhà bán hàng trực tuyến và bộ phận kinh doanh sàn thương mại điện tử cần công cụ dự báo doanh số, tối ưu định giá và kế hoạch tồn kho.

## 4. Input / Output
- **Input:** Thông tin chi tiết một đơn hàng gồm: Ngành hàng (`Category`), Kích cỡ (`Size`), Số lượng (`Qty`), Cấp dịch vụ giao hàng (`ship-service-level`), Kênh hoàn tất (`Fulfilment`), Bang nhận hàng (`ship-state`), Ngày giao dịch (`Date`), Trạng thái khuyến mãi (`promotion-ids`) và Loại khách hàng (`B2B`).
- **Output:** Doanh số dự đoán cụ thể ($\hat{y}$ tính bằng INR) kèm thời gian phản hồi tức thì.

## 5. Core Features
- **Khám phá dữ liệu (EDA):** Khảo sát cấu trúc, phân tích phân phối lệch phải của Amount, biến đổi Log, phân tích tương quan hai biến với ngành hàng và địa phương, quét ngoại lai IQR.
- **Tiền xử lý & Khử rò rỉ:** Lọc đơn hàng hủy, loại đơn có Amount $\le 0$ hoặc Qty $\le 0$, điền khuyết thiếu theo logic nghiệp vụ, loại bỏ các cột định danh ngẫu nhiên và phương sai bằng 0; khử triệt để trùng lặp.
- **Kỹ thuật đặc trưng (Feature Engineering):** Trích xuất yếu tố mùa vụ từ `Date` (`month`, `day_of_week`, `is_weekend`), gom nhóm địa lý Top 10 bang lớn + Other, tạo cờ nhị phân `is_promoted` và `is_b2b`; quét và loại bỏ đa cộng tuyến ($|r| > 0.6$); chuẩn hóa và mã hóa thống nhất qua `ColumnTransformer`.
- **Demo tương tác đa mô hình:** Notebook thử nghiệm `notebooks/demo.ipynb` cho phép nhập tham số đơn hàng bất kỳ và đối chiếu tức thì kết quả dự báo của 3 mô hình hồi quy.

## 6. Technology Constraints
- **Môi trường & Thư viện:** Python 3.10+, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Joblib.
- **Thực thi:** 4 Jupyter Notebooks tuần tự (EDA $\rightarrow$ Kỹ thuật đặc trưng $\rightarrow$ Huấn luyện & Đánh giá $\rightarrow$ Demo) hoặc tự động hóa hoàn toàn bằng lệnh `python scripts/run_pipeline.py`.

## 7. Algorithm / Evaluation
- **Thuật toán Scratch:** Tự cài đặt 3 thuật toán hồi quy bằng 100% Python/NumPy thuần túy:
  1. *Linear Regression Scratch* (tối ưu nghiệm đóng Normal Equation kết hợp Gradient Descent L2).
  2. *Decision Tree Regressor Scratch* (tối ưu tiêu chí giảm phương sai Variance Reduction / MSE, kết hợp chia ngưỡng nhanh qua Quantile Splits).
  3. *Random Forest Regressor Scratch* (học kết hợp Bagging, chọn ngẫu nhiên không gian đặc trưng Random Subspace, tổng hợp trung bình dự đoán).
- **Chia dữ liệu:** Tách tập Train (80% — 82.281 mẫu) và Test (20% — 20.571 mẫu) độc lập với `random_state=42`.
- **Đánh giá:** 4 chỉ số hồi quy tiêu chuẩn: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), và Hệ số xác định ($R^2$); trực quan hóa biểu đồ Thực tế vs Dự đoán, biểu đồ phần dư Residual Plot và Top 15 đặc trưng quan trọng.
