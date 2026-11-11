# ĐẶC TẢ BÀI THỰC HÀNH 3: PHÂN KHÚC KHÁCH HÀNG BÁN LẺ — NHÓM 3

## 1. Project Name
Phân Cụm Khách Hàng Bán Lẻ (Retail Store Customer Segmentation & Clustering)

## 2. Problem
Phân cụm dữ liệu không giám sát: tự động nhóm khách hàng trung tâm thương mại thành các phân khúc hành vi mua sắm riêng biệt.
- **Dữ liệu thực nghiệm:** `data/raw/Mall_Customers.csv` (nguồn Kaggle). Tập dữ liệu chuẩn gồm 200 khách hàng, 5 trường thuộc tính hoàn chỉnh, 100% dữ liệu sạch, không khuyết thiếu và không trùng lặp.
- **Phạm vi bài lab:** Khám phá dữ liệu EDA Read-Only, tiền xử lý và chuẩn hóa thang đo, tự cài đặt 3 thuật toán phân cụm Scratch thuần NumPy, tối ưu siêu tham số, đối chuẩn toán học với Scikit-learn, vẽ cây Dendrogram, lập hồ sơ 5 Personas và demo suy luận phân cụm tương tác.

## 3. Target User
- Người học và giảng viên môn Học máy (đánh giá năng lực tự cài đặt thuật toán phân cụm Scratch và xây dựng pipeline hoàn chỉnh).
- Về ý tưởng ứng dụng: Bộ phận Tiếp thị (Marketing) và Quản trị Quan hệ Khách hàng (CRM) của trung tâm thương mại cần phân khúc khách hàng tự động để cá nhân hóa ưu đãi và tối ưu doanh thu.

## 4. Input / Output
- **Input:** Thông tin khách hàng gồm: Mã định danh (`CustomerID`), Giới tính (`Gender`), Độ tuổi (`Age`), Thu nhập hàng năm (`Annual Income (k$)`) và Điểm thói quen chi tiêu (`Spending Score (1-100)`).
- **Output:** Nhãn phân cụm gán cho từng khách hàng ($0 \dots 4$), hồ sơ định lượng 5 nhóm Personas, 4 biểu đồ chuyên sâu (Dendrogram, 2D K-Means, DBSCAN) và mô hình lưu trữ `.joblib`.

## 5. Core Features
- **Khám phá dữ liệu (EDA):** Khảo sát cấu trúc bảng, phân tích phân phối đơn biến (Age, Income, Spending), tương quan Pearson trực giao giữa Thu nhập và Chi tiêu ($r \approx 0.0099$), quét ngoại lai IQR theo cơ chế Read-Only 100%.
- **Tiền xử lý dữ liệu:** Chuẩn hóa tên cột thống nhất, mã hóa nhị phân `Gender` (Female: 0, Male: 1) và xuất lưu trữ dữ liệu sạch `data/processed/clean_customers.csv`.
- **Kỹ thuật đặc trưng & Scaling:** Thiết lập không gian 2D cốt lõi Thu nhập - Chi tiêu, áp dụng `StandardScaler` đưa các thuộc tính về phân phối chuẩn tắc ($\mu=0, \sigma=1$) chống thiên lệch khoảng cách Euclidean.
- **Demo tương tác:** Thử nghiệm suy luận phân cụm tương tác thời gian thực tại `notebooks/demo.ipynb`, tự động nạp mô hình tốt nhất đã lưu để phân loại khách hàng mới và đưa ra chiến lược bán lẻ.

## 6. Technology Constraints
- **Môi trường & Thư viện:** Python 3.10+, NumPy, Pandas, Scikit-learn, SciPy, Matplotlib, Seaborn, Joblib.
- **Thực thi:** Tổ chức qua 4 Jupyter Notebooks tuần tự (EDA $\rightarrow$ Tiền xử lý & Đặc trưng $\rightarrow$ Huấn luyện & Đánh giá $\rightarrow$ Demo tương tác) hoặc chạy tự động hóa toàn trình qua lệnh `python scripts/run_pipeline.py`.

## 7. Algorithm / Evaluation
- **Thuật toán Scratch:** Tự cài đặt 3 thuật toán phân cụm From Scratch bằng NumPy:
  1. *K-Means Scratch* (Khởi tạo $K$-Means++, tối ưu WCSS Inertia).
  2. *Hierarchical Agglomerative Scratch* (Ward Linkage, xây dựng cây phân cấp Dendrogram).
  3. *DBSCAN Scratch* (Phân cụm theo mật độ $\varepsilon, \text{min\_samples}$, cô lập điểm nhiễu ngoại lai).
- **Không gian phân cụm:** Không gian 2D trực giao cốt lõi `Annual Income (k$)` vs `Spending Score (1-100)` kết hợp chuẩn hóa StandardScaler.
- **Đánh giá:** Bộ 3 chỉ số phân cụm nội tại: Silhouette Score ($\uparrow$), Davies-Bouldin Index ($\downarrow$) và Calinski-Harabasz Index ($\uparrow$); kiểm soát đối chuẩn toán học độ sai lệch tuyệt đối $< 10^{-10}$ so với Scikit-learn.
- **Tiêu chí nghiệm thu:** Mô hình Quán quân K-Means Scratch đạt $\text{Silhouette Score} \ge 0.55$ tại $K=5$, phân tách rõ rệt 5 phân khúc khách hàng tự nhiên.
