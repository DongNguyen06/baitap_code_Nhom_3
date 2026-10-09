# QUY CHUẨN LÀM VIỆC VỚI AI & PHỐI HỢP NHÓM (TEAM WORKFLOW) — SPAMGUARD-ML
## PRACTICE 1: CLASSIFYING SPAM EMAILS — NHÓM 3 (UTH)
> **Giảng viên hướng dẫn:** Tiến sĩ Nguyễn Thị Khánh Tiên  
> **Phương pháp luận:** Áp dụng toàn diện 4 nguyên tắc Vibe Coding thực chiến theo giáo trình môn học.

---

## 1. BẢN QUY CHUẨN 4 NGUYÊN TẮC VIBE CODING VỚI AI

### 1.1. Tư duy cốt lõi (Core Mindset)
> *"AI chỉ là trợ lý (phân tích, sinh mã, debug, viết tài liệu); bạn chịu trách nhiệm 100% về kiến trúc, tính đúng đắn toán học và kết quả của dự án."*

### 1.2. Sáu nguyên tắc làm việc với AI (6 Golden Principles)
1. **Hiểu bài toán trước khi code:** Nắm rõ bản chất email spam và luồng dữ liệu trước khi yêu cầu AI viết mã.
2. **Lên kế hoạch, chia nhỏ module/milestone:** Bám sát từng ô (cell) và từng mục cụ thể trong file Notebook.
3. **Prompt ngắn gọn:** Mỗi prompt xử lý đúng 1 task duy nhất, không đưa yêu cầu chung chung.
4. **Chạy và kiểm tra (verify):** Bấm `Shift + Enter` chạy thử ngay sau mỗi lần nhận code từ AI.
5. **Không tin tưởng tuyệt đối output của AI:** Luôn tự xác minh kết quả, kiểm tra ma trận kích thước, nhãn dự đoán và điểm số.
6. **Phải hiểu và giải thích được code:** Tuyệt đối không dùng code mù quáng; phải nắm được lý thuyết để bảo vệ trước giảng viên.

### 1.3. Vòng lặp tương tác chuẩn (Core Loop)
```text
Describe ──> Generate ──> Review ──> Run ──> Test ──> Refactor ──> Commit
 (Mô tả)      (AI sinh)    (Xem lại)   (Chạy)  (Kiểm tra)  (Tối ưu)   (Lưu Git)
```

### 1.4. Quy chuẩn lưu trữ & Tài liệu
- **Hệ thống Git:** Quản lý mã nguồn tập trung, có lịch sử rõ ràng; mỗi thành viên rẽ nhánh riêng theo quy trình chuẩn.
- **Tài liệu dự án:** Viết đầy đủ `README.md`, thư mục `docs/` (nêu rõ 7 mục ở khâu Define).
- **Cấu trúc thực thi chuẩn:** Tổ chức thư mục chuẩn gồm `docs/`, `scripts/`, `models/`, và file notebook (`notebooks/spam_classification.ipynb`).

---

## 2. QUY TRÌNH GIT NHÁNH NỐI TIẾP (CHO 1 NOTEBOOK DUY NHẤT)

Khi cả 7 thành viên cùng đóng góp vào **1 file Notebook duy nhất**, các nhánh Git được thực hiện theo cơ chế **gối đầu nối tiếp** (nhánh của bạn sau chỉ rẽ ra từ `dev` sau khi nhánh của bạn trước đã merge thành công).

```text
dev (gốc) 
  └─► Nhánh TV2 (Mục 2: Tiền xử lý) ──► Merge vào dev
                                            └─► TV3 kéo dev về ──► Nhánh TV3 (Mục 3: TF-IDF) ──► Merge vào dev
                                                                                                    └─► TV4...
```

### Quy trình 5 bước thực hiện của từng thành viên:

#### Bước 1: Kéo bản mới nhất của nhánh `dev` về máy
```bash
# 1. Chuyển về nhánh dev
git checkout dev

# 2. Kéo code mới nhất từ GitHub về
git pull origin dev
```

#### Bước 2: Rẽ nhánh cá nhân để thực hiện mục của mình
```bash
git checkout -b feature/tv<số>-<tên_mục>
# Ví dụ TV2: git checkout -b feature/tv2-data-cleaning
# Ví dụ TV3: git checkout -b feature/tv3-feature-engineering
```

#### Bước 3: Mở Notebook, thực hiện code và kiểm tra
- Mở file `notebooks/spam_classification.ipynb`.
- Tìm đến đúng **Mục (Section)** được phân công trong `TASKS.md`.
- Vận dụng AI theo vòng lặp Vibe Coding để hoàn thiện code của mục đó.
- Nhấn `Shift + Enter` chạy kiểm thử, đảm bảo không có lỗi phát sinh.

#### Bước 4: ⚠️ QUY TẮC BẮT BUỘC: XÓA OUTPUT TRƯỚC KHI COMMIT
Trước khi lưu file để commit lên Git, bắt buộc phải xóa toàn bộ kết quả chạy tạm thời nhằm tránh xung đột file JSON:
* Trong Jupyter/VS Code: Chọn **Kernel / Edit** $\rightarrow$ Chọn **Clear All Outputs** (Xóa tất cả kết quả đầu ra).
* Nhấn `Ctrl + S` để lưu file Notebook sạch.

#### Bước 5: Commit, Push và Tạo Pull Request (PR)
```bash
git add notebooks/spam_classification.ipynb
git commit -m "[TV<số>] Hoàn thành Mục <số>: <Tên_mục>"
git push -u origin feature/tv<số>-<tên_mục>
```
Lên GitHub mở Pull Request (PR) vào nhánh `dev`. Trưởng nhóm (TV1) review và duyệt merge vào `dev`. Thành viên tiếp theo pull `dev` về và làm tiếp mục sau!
