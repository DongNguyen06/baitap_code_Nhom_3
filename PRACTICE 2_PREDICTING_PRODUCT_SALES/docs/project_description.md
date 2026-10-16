# ĐẶC TẢ DỰ ÁN (PROJECT SPECIFICATION) — PRACTICE 2
> 📝 **HƯỚNG DẪN DÀNH CHO BẠN:** Hãy viết nội dung nháp vào 7 mục bên dưới theo bài toán **Predicting Product Sales**.  
> ⚠️ **LƯU Ý QUAN TRỌNG:** Tổng độ dài file sau khi viết xong **BẮT BUỘC $\le 200$ TỪ** (Tôi sẽ review, đếm từ và tối ưu lại giúp bạn sau khi bạn viết xong).

## 1. Project Name
[Điền tên hệ thống: Ví dụ - Hệ thống dự đoán doanh số bán hàng sản phẩm thương mại điện tử (SalesPredict-ML)]

## 2. Problem
[Mô tả bài toán hồi quy: Dự đoán doanh số bán hàng (Product Sales) liên tục nhằm tối ưu quản lý tồn kho và chiến dịch marketing]

## 3. Target User
[Người dùng mục tiêu: Doanh nghiệp thương mại điện tử, nhà bán hàng hoặc bộ phận quản lý chuỗi cung ứng]

## 4. Input / Output
- **Input:** [Các thuộc tính đầu vào: Thông tin sản phẩm, nhân khẩu học khách hàng, lịch sử mua hàng, chi phí marketing, mùa vụ]
- **Output:** [Doanh số bán hàng dự đoán (Product Sales) theo thời gian thực]

## 5. Core Features
- [Khám phá dữ liệu (EDA): Phân tích phân phối doanh số, đo lường tương quan và phát hiện ngoại lai IQR]
- [Tiền xử lý dữ liệu: Khử trùng lặp, xử lý giá trị khuyết và mã hóa danh mục]
- [Kỹ thuật đặc trưng: Biến đổi log nén lệch phải, tạo biến độ phổ biến và xu hướng mùa vụ]
- [Dự đoán và mô phỏng doanh số qua giao diện Web trực quan]

## 6. Technology Constraints
[Python 3.10+, Scikit-learn, Pandas; tổ chức quy trình thực thi qua 3 Jupyter Notebooks tuần tự]

## 7. Algorithm / Evaluation
- **Thuật toán:** [Linear Regression, Decision Tree Regressor, Random Forest Regressor]
- **Đánh giá:** [Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared (R2)]
