# THIẾT KẾ HỆ THỐNG THEO TÍNH NĂNG (SYSTEM DESIGN) — PRACTICE 2
> 📝 **HƯỚNG DẪN DÀNH CHO BẠN:** Hãy viết nội dung thiết kế theo 4 tính năng chính của bài toán **Predicting Product Sales**.  
> ⚠️ **LƯU Ý QUAN TRỌNG:** Tổng độ dài file sau khi viết xong **BẮT BUỘC $\le 200$ TỪ** (Tôi sẽ review, đếm từ và tối ưu lại giúp bạn sau khi bạn viết xong).

## 1. Khám phá & Làm sạch dữ liệu (Data Cleaning & EDA)
- [EDA: Phân tích phân phối doanh số bán hàng, đo lường tương quan các biến số và nhận diện ngoại lai qua IQR/Boxplot]
- [Xử lý chất lượng: Khử dữ liệu trùng lặp, điền giá trị khuyết (trung vị cho biến số, mode cho biến danh mục)]

## 2. Kỹ thuật đặc trưng & Pipeline (Feature Engineering)
- [Biến đổi & Tạo biến: Biến đổi log nén độ lệch phải, tạo biến độ phổ biến sản phẩm, giá trị vòng đời khách hàng và mùa vụ]
- [Mã hóa & Chuẩn hóa: Mã hóa One-Hot cho danh mục, chuẩn hóa StandardScaler và lọc đặc trưng F-regression]
- [Đóng gói Pipeline chống rò rỉ dữ liệu (Data Leakage) khi kiểm định chéo]

## 3. Huấn luyện mô hình (Model Building)
- [Huấn luyện các mô hình hồi quy cốt lõi: Linear Regression, Decision Tree Regressor, Random Forest Regressor]
- [Áp dụng 5-Fold Cross-Validation trên tập huấn luyện và tinh chỉnh siêu tham số]

## 4. Đánh giá & Triển khai (Evaluation & Deployment)
- [Đánh giá trên tập Test độc lập qua các chỉ số: MSE, RMSE, MAE và hệ số xác định R2]
- [Trực quan hóa biểu đồ Thực tế vs Dự đoán, biểu đồ phần dư và phân tích mẫu có sai số lớn nhất]
- [Lưu mô hình tối ưu nhất và xây dựng Web Demo tương tác dự đoán doanh số thời gian thực]
