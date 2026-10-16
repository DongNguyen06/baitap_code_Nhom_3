# 🏛️ TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM (UTH)
## BỘ MÔN: HỌC MÁY (MACHINE LEARNING) — NHÓM 3

> **Học kỳ:** Năm 3 — Học kỳ 1  
> **Giảng viên hướng dẫn (GVHD):** Tiến sĩ Nguyễn Thị Khánh Tiên  
> **Kho lưu trữ:** Tổng hợp các bài thực hành và đồ án môn học  

---

## 👥 1. DANH SÁCH THÀNH VIÊN NHÓM 3 (7 THÀNH VIÊN)

| STT | Họ và tên | Vai trò |
| :---: | :--- | :---: |
| **1** | **Nguyễn Xuân Đông** | **Trưởng nhóm** |
| **2** | **Mai Hoàng Danh** | Thành viên |
| **3** | **Trương Thành Công** | Thành viên |
| **4** | **Võ Khôi Nguyên** | Thành viên |
| **5** | **Võ Thanh Phú** | Thành viên |
| **6** | **Nguyễn Minh Nhật** | Thành viên |
| **7** | **Mai An Thịnh** | Thành viên |

---

## 📚 2. DANH MỤC CÁC BÀI THỰC HÀNH & ĐỒ ÁN (PROJECT ROADMAP)

Monorepo này được thiết kế để chứa tất cả các dự án trong suốt học kỳ. Mỗi bài thực hành là một thư mục con độc lập:

| Mã bài | Tên bài thực hành / Dự án | Thư mục mã nguồn | Báo cáo & Tài liệu | Trạng thái |
| :---: | :--- | :--- | :---: | :---: |
| **Practice 1** | **Classifying Spam Emails**<br>*(Phân loại email/tin nhắn rác)* | [`PRACTICE 1_CLASSIFYING_SPAM_EMAILS/`](./PRACTICE%201_CLASSIFYING_SPAM_EMAILS/) | [Xem Docs](./PRACTICE%201_CLASSIFYING_SPAM_EMAILS/docs/) | 🟢 **Đang thực hiện** |
| **Practice 2** | **Predicting Product Sales**<br>*(Dự đoán doanh số bán hàng)* | [`PRACTICE 2_PREDICTING_PRODUCT_SALES/`](./PRACTICE%202_PREDICTING_PRODUCT_SALES/) | [Xem Docs](./PRACTICE%202_PREDICTING_PRODUCT_SALES/docs/) | 🟢 **Đang thực hiện** |
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
