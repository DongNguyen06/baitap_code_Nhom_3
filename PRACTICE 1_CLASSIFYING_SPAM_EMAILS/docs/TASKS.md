# KẾ HOẠCH PHÂN RÃ CÔNG VIỆC (TASKS BREAKDOWN) — SPAMGUARD-ML
## PRACTICE 1: CLASSIFYING SPAM EMAILS — NHÓM 3 (UTH)

> **Nguyên tắc:** Phân công bám sát dòng chảy khoa học dữ liệu thẳng tắp (EDA toàn diện $\rightarrow$ Làm sạch $\rightarrow$ Kỹ thuật đặc trưng $\rightarrow$ Huấn luyện $\rightarrow$ Đánh giá & Triển khai) và thực hiện tuần tự trên 1 Notebook duy nhất (`notebooks/spam_classification.ipynb`).

---

## SƠ ĐỒ LUỒNG THỰC HIỆN TUẦN TỰ (PIPELINE WORKFLOW)

```text
[CEAS_08.csv] ──> TV 1 (EDA Nền tảng: Cấu trúc, Chất lượng & Đơn biến)
              ──> TV 2 (EDA Chuyên sâu: Đa biến, Ngoại lai & Đúc kết định hướng)
              ──> TV 3 (Làm sạch dữ liệu: Khử trùng, điền missing, clean_text)
              ──> TV 4 (FE Biến đổi: Log, Binning, Tạo biến Domain, Mã hóa)
              ──> TV 5 (FE Lọc & Pipeline: Chia Train/Test chống Leak, TF-IDF, Chi2)
              ──> TV 6 (Huấn luyện 3 mô hình: Naive Bayes, Logistic L1, Linear SVM)
              ──> TV 7 (Đánh giá kiểm thử, Phân tích lỗi & Web Demo Streamlit)
```

---

## BẢNG PHÂN CÔNG NHIỆM VỤ & MỨC ĐỘ ĐÓNG GÓP (TỔNG: 100%)

| STT | Thành viên | Vai trò | Đóng góp | Tính năng (Design) | Mục Notebook | Tóm tắt nhiệm vụ chính |
| :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| **TV 1** | **Nguyễn Xuân Đông** | **Trưởng nhóm** | **15%** | **Tính năng 1:**<br>EDA Nền tảng | **Mục 1** | • Khởi tạo dự án và cấu hình môi trường làm việc chung.<br>• Khảo sát cấu trúc tổng quan, kiểu dữ liệu và từ điển dữ liệu.<br>• Đánh giá chất lượng dữ liệu: Thống kê tỷ lệ khuyết thiếu và dữ liệu trùng lặp.<br>• Phân tích đơn biến toàn diện: Biến mục tiêu, biến số liên kết, độ dài tiêu đề và tên miền người gửi.<br>• Quản trị phân nhánh Git, rà soát mã nguồn và điều phối tiến độ. |
| **TV 2** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 1:**<br>EDA Chuyên sâu | **Mục 2** | • Phân tích đa biến: Khảo sát mối quan hệ giữa các thuộc tính với nhãn mục tiêu.<br>• Phân tích phân phối độ dài văn bản nội dung và nhận diện hiện tượng lệch phải.<br>• Xác định giá trị ngoại lai theo công thức khoảng tứ phân vị và đo lường ma trận tương quan.<br>• Đúc kết các bằng chứng EDA và lập bảng kế hoạch hành động cho tiền xử lý và đặc trưng. |
| **TV 3** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 1:**<br>Làm sạch dữ liệu | **Mục 3** | • Thực thi xử lý chất lượng: Khử trùng lặp hoàn toàn và xử lý các email thiếu tiêu đề.<br>• Gộp tiêu đề và nội dung thành chuỗi văn bản hoàn chỉnh.<br>• Tiền xử lý văn bản chuyên sâu: Bóc tách mã HTML, loại bỏ liên kết và ký tự đặc biệt, chuẩn hóa chữ thường và lọc từ dừng.<br>• Xuất dữ liệu sạch sẵn sàng cho giai đoạn tạo đặc trưng. |
| **TV 4** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 2:**<br>FE Biến đổi & Domain | **Mục 4** | • Kỹ thuật biến đổi đặc trưng: Áp dụng biến đổi log để nén độ lệch phân phối của độ dài email.<br>• Rời rạc hóa: Gom nhóm độ dài văn bản liên tục thành các khoảng thứ bậc.<br>• Kỹ thuật mã hóa: Thực hiện mã hóa các nhóm đặc trưng phân loại.<br>• Tạo đặc trưng mới theo ngữ cảnh: Tỷ lệ viết hoa, tần suất ký tự cảm thán và ký tự tiền tệ. |
| **TV 5** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 2:**<br>FE Trích xuất & Lọc | **Mục 5** | • Thiết lập hàng rào phân chia tập huấn luyện và kiểm thử theo chuẩn chống rò rỉ dữ liệu.<br>• Trích xuất đặc trưng văn bản: Chuyển đổi nội dung email sang không gian vector trọng số từ.<br>• Chọn lọc đặc trưng: Lọc và xếp hạng các từ khóa mang tính phân biệt cao nhất giữa thư rác và thư hợp lệ.<br>• Đóng gói luồng tiền xử lý vào pipeline chuẩn để kiểm định chéo an toàn. |
| **TV 6** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 3:**<br>Huấn luyện 3 mô hình | **Mục 6** | • Huấn luyện mô hình Naive Bayes kết hợp đánh giá chéo phân tầng trên tập huấn luyện.<br>• Huấn luyện mô hình Logistic Regression kết hợp cơ chế tự động triệt tiêu trọng số từ vô dụng.<br>• Huấn luyện mô hình Linear SVM và hiệu chuẩn xác suất dự đoán phục vụ người dùng cuối.<br>• Tinh chỉnh siêu tham số điều chuẩn và lưu trữ cả 3 mô hình đã tối ưu. |
| **TV 7** | *[Chưa chỉ định]* | Thành viên | **15%** | **Tính năng 4:**<br>Đánh giá & Triển khai | **Mục 7** | • Đánh giá độc lập và đối chiếu hiệu năng 3 mô hình trên tập kiểm thử độc lập.<br>• Trực quan hóa ma trận nhầm lẫn và thực hiện phân tích lỗi các trường hợp dự đoán sai.<br>• Tối ưu hóa ngưỡng xác suất phân loại nhằm hạn chế tối đa việc bỏ sót email quan trọng.<br>• Đo lường độ trễ suy luận, xuất mô hình tốt nhất và hoàn thiện giao diện Web Demo tương tác. |
| **TỔNG** | **7 thành viên** | | **100%** | | | **Hoàn thành toàn diện 4 tính năng hệ thống** |
