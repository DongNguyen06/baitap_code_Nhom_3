# 🏛️ ĐẠI HỌC GIAO THÔNG VẬN TẢI TPHCM (UTH)
## BỘ MÔN: HỌC MÁY (MACHINE LEARNING) — NHÓM 3

> **Học kỳ:** Năm 3 — Học kỳ 1  
> **Repository:** `ML_Nhom_3` (Monorepo tổng quản lý toàn bộ các bài thực hành và đồ án môn học)  
> **Trưởng nhóm (Leader):** Nguyễn Xuân Đông  

---

## 👥 1. DANH SÁCH THÀNH VIÊN NHÓM 3 (7 THÀNH VIÊN)

| STT | Mã sinh viên (MSSV) | Họ và tên | Vai trò phụ trách chung | Ký hiệu (Tag) |
| :---: | :---: | :--- | :--- | :---: |
| **1** | `94206000207` | **Nguyễn Xuân Đông** | Quản trị Repo, Kiến trúc, Review PR, Slide báo cáo | `TV1` |
| **2** | `066206005311` | **Mai Hoàng Danh** | Tiền xử lý dữ liệu (Data Preprocessing, Cleaning, Regex) | `TV2` |
| **3** | `87206002694` | **Trương Thành Công** | Kỹ thuật đặc trưng (Feature Engineering, TF-IDF, Vectorization) | `TV3` |
| **4** | `79206001370` | **Võ Khôi Nguyên** | Kỹ sư ML 1 (Xây dựng & Tinh chỉnh Naive Bayes, Logistic Regression) | `TV4` |
| **5** | `89206003078` | **Võ Thanh Phú** | Kỹ sư ML 2 (Xây dựng & Tinh chỉnh Linear SVM, Ensemble Methods) | `TV5` |
| **6** | `70206006644` | **Nguyễn Minh Nhật** | Đánh giá mô hình (Metrics, Confusion Matrix, Error Analysis, Notebook) | `TV6` |
| **7** | `052206005348` | **Mai An Thịnh** | Triển khai & Kiểm thử (Console Chat, Web Demo Streamlit, Unit Tests) | `TV7` |

---

## 📚 2. DANH MỤC CÁC BÀI THỰC HÀNH & ĐỒ ÁN (PROJECT ROADMAP)

Monorepo này được thiết kế để chứa tất cả các dự án trong suốt học kỳ. Mỗi bài thực hành là một thư mục con độc lập:

| Mã bài | Tên bài thực hành / Dự án | Thư mục mã nguồn | Báo cáo & Tài liệu | Trạng thái |
| :---: | :--- | :--- | :---: | :---: |
| **Practice 1** | **Classifying Spam Emails**<br>*(Phân loại email/tin nhắn rác)* | [`PRACTICE 1_CLASSIFYING_SPAM_EMAILS/`](./PRACTICE%201_CLASSIFYING_SPAM_EMAILS/) | [Xem Docs](./PRACTICE%201_CLASSIFYING_SPAM_EMAILS/docs/) | 🟢 **Đang thực hiện** |
| **Practice 2** | **Predicting Product Sales**<br>*(Dự đoán doanh số bán hàng)* | `PRACTICE 2_PREDICTING_PRODUCT_SALES/` | *Chờ cập nhật* | ⏳ Sắp triển khai |
| **Practice 3** | **Machine Learning Project 3** | *Chờ đề bài* | *Chờ cập nhật* | ⏳ Sắp triển khai |
| **Final Project**| **Đồ án cuối kỳ môn Học máy** | `FINAL_PROJECT/` | *Chờ cập nhật* | ⏳ Kế hoạch cuối kỳ |

---

## 🛠️ 3. QUY ƯỚC LÀM VIỆC GIT TRÊN MONOREPO

Để tránh xung đột khi 7 thành viên cùng làm việc trên nhiều dự án khác nhau:

### 3.1. Cấu trúc nhánh (Branching Strategy):
* **`main`:** Nhánh bảo vệ (Protected Branch) — Chỉ merge vào cuối kỳ hoặc khi mỗi bài đã hoàn thiện 100%.
* **`dev`:** Nhánh tích hợp trung tâm của cả nhóm.
* **Nhánh cá nhân cho từng bài:** Tạo từ `dev` theo tiền tố:
  * Khi làm **Practice 1**: `p1/feature/tv<số>-<tên-chức-năng>` (Ví dụ: `p1/feature/tv2-data-preprocessing`)
  * Khi làm **Practice 2**: `p2/feature/tv<số>-<tên-chức-năng>` (Ví dụ: `p2/feature/tv2-eda-sales`)

### 3.2. Cú pháp Commit có gắn thẻ (Tagging Convention):
Mỗi lần commit cần ghi rõ mã bài và mã thành viên để minh bạch đóng góp cá nhân:
```text
[P<mã-bài>][TV<số>] <Loại hành động>: <Mô tả việc đã làm bằng tiếng Việt>
```
* **Ví dụ:** `git commit -m "[P1][TV1] Khoi tao: Cau hinh he thong va tai lieu dac ta"`
* **Ví dụ:** `git commit -m "[P1][TV2] Them: Ham boc tach the HTML va loc dau cau"`

---

## 💻 4. CÀI ĐẶT MÔI TRƯỜNG CHUNG

Cài đặt tất cả thư viện cần thiết cho các bài thực hành bằng một lệnh duy nhất:
```bash
pip install -r requirements.txt
```
