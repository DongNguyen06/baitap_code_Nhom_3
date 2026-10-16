# KẾ HOẠCH PHÂN RÃ CÔNG VIỆC (TASKS BREAKDOWN) — SALESPREDICT-ML
## PRACTICE 2: PREDICTING PRODUCT SALES — NHÓM 3 (UTH)

> **Nguyên tắc:** Phân công bám sát dòng chảy khoa học dữ liệu thẳng tắp cho bài toán Hồi quy (EDA toàn diện $\rightarrow$ Làm sạch $\rightarrow$ Kỹ thuật đặc trưng $\rightarrow$ Huấn luyện $\rightarrow$ Đánh giá & Triển khai) và thực hiện tuần tự trên 1 Notebook duy nhất (`notebooks/product_sales_prediction.ipynb`).

---

## SƠ ĐỒ LUỒNG THỰC HIỆN TUẦN TỰ (PIPELINE WORKFLOW)

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

## BẢNG PHÂN CÔNG NHIỆM VỤ & MỨC ĐỘ ĐÓNG GÓP (TỔNG: 100%)

| STT | Thành viên | Vai trò | Đóng góp | Tính năng (Design) | Mục Notebook | Tóm tắt nhiệm vụ chính |
| :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| **TV 1** | **Nguyễn Xuân Đông** | **Trưởng nhóm** | **15%** | **Tính năng 1:**<br>EDA Nền tảng | **Mục 1** | • Khởi tạo dự án và thiết lập môi trường làm việc chung.<br>• Khảo sát cấu trúc tổng quan, các cột thuộc tính và kiểu dữ liệu.<br>• Đánh giá chất lượng dữ liệu: Thống kê tỷ lệ khuyết thiếu và dữ liệu trùng lặp.<br>• Phân tích đơn biến toàn diện: Biến mục tiêu doanh số bán hàng, phân phối các biến nhân khẩu học và chi phí marketing.<br>• Quản trị phân nhánh Git, rà soát mã nguồn và điều phối tiến độ. |
| **TV 2** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 1:**<br>EDA Chuyên sâu | **Mục 2** | • Phân tích đa biến: Khảo sát mối tương quan giữa chi phí quảng cáo, giá sản phẩm với doanh số.<br>• Phân tích phân phối độ lệch của doanh số và nhận diện hiện tượng đuôi dài.<br>• Xác định giá trị ngoại lai theo công thức khoảng tứ phân vị và đo lường ma trận tương quan.<br>• Đúc kết các bằng chứng EDA và lập bảng kế hoạch hành động định hướng cho tiền xử lý và đặc trưng. |
| **TV 3** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 1:**<br>Làm sạch dữ liệu | **Mục 3** | • Thực thi xử lý chất lượng: Khử trùng lặp hoàn toàn các bản ghi giao dịch.<br>• Xử lý dữ liệu khuyết: Điền trung vị cho các biến số có phân phối lệch và điền mode cho biến danh mục.<br>• Chuẩn hóa các trường định dạng ngày tháng và thời gian mua hàng.<br>• Xuất dữ liệu sạch sẵn sàng cho giai đoạn tạo đặc trưng. |
| **TV 4** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 2:**<br>FE Biến đổi & Domain | **Mục 4** | • Kỹ thuật biến đổi đặc trưng: Áp dụng biến đổi log để nén độ lệch phân phối của giá và chi phí.<br>• Rời rạc hóa: Gom nhóm các biến liên tục như tuổi khách hàng thành các khoảng nhóm.<br>• Tạo đặc trưng mới theo ngữ cảnh: Điểm phổ biến sản phẩm, giá trị vòng đời khách hàng và chỉ số mùa vụ.<br>• Đo lường ma trận tương quan của các biến mới tạo với doanh số. |
| **TV 5** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 2:**<br>FE Mã hóa, Chuẩn hóa & Lọc | **Mục 5** | • Thiết lập hàng rào phân chia tập huấn luyện và kiểm thử theo chuẩn chống rò rỉ dữ liệu.<br>• Kỹ thuật mã hóa: Mã hóa One-Hot cho các danh mục sản phẩm và nhân khẩu học.<br>• Chuẩn hóa dữ liệu số học: Đưa các thuộc tính về cùng thang đo chuẩn trên tập huấn luyện.<br>• Chọn lọc đặc trưng: Lọc các biến quan trọng nhất và đóng gói vào pipeline chuẩn. |
| **TV 6** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 3:**<br>Huấn luyện mô hình | **Mục 6** | • Huấn luyện mô hình hồi quy tuyến tính (Linear Regression) làm đường cơ sở (Baseline).<br>• Huấn luyện mô hình Cây quyết định (Decision Tree Regressor).<br>• Huấn luyện mô hình Rừng ngẫu nhiên (Random Forest Regressor) bọc trong Pipeline.<br>• Áp dụng 5-Fold Cross-Validation, tinh chỉnh siêu tham số và lưu các mô hình. |
| **TV 7** | *[Chưa chỉ định]* | Thành viên | **15%** | **Tính năng 4:**<br>Đánh giá & Triển khai | **Mục 7** | • Đánh giá độc lập trên tập kiểm thử qua 4 chỉ số: MSE, RMSE, MAE và hệ số R2.<br>• Trực quan hóa biểu đồ Thực tế vs Dự đoán và biểu đồ phân tán phần dư (Residual Plot).<br>• Phân tích lỗi chuyên sâu: Mổ xẻ các sản phẩm có sai số dự đoán lớn nhất.<br>• Đo lường độ trễ suy luận, xuất mô hình tốt nhất và hoàn thiện Web Demo Streamlit. |
| **TỔNG** | **7 thành viên** | | **100%** | | | **Hoàn thành toàn diện 4 tính năng hệ thống** |
