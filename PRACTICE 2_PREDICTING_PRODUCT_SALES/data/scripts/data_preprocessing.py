"""
Module: data_preprocessing.py
Dự án: Practice 2: Predicting Product Sales — Group 3
Phụ trách chính: Nguyễn Minh Nhật (14%)
Nhiệm vụ: Tiền xử lý, lọc đơn hủy, chuẩn hóa dữ liệu và khử trùng lặp triệt để (Zero Duplicate Leakage).
"""
from pathlib import Path
import sys
from typing import Optional, Union
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def handle_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Khử trùng lặp bản ghi và điền khuyết thiếu cho các trường dữ liệu theo ngữ cảnh nghiệp vụ.
    Phụ trách chính: Nguyễn Minh Nhật (14%)

    Quy tắc nghiệp vụ:
    - Khử trùng lặp triệt để: drop_duplicates() trước khi xử lý khuyết thiếu (Zero Duplicate Leakage).
    - promotion-ids: Điền chuỗi 'No_Promotion' cho các ô bị khuyết (NaN).
    - Courier Status: Điền chuỗi 'Unshipped' cho các ô bị khuyết (NaN).
    - ship-state: Điền giá trị xuất hiện nhiều nhất (mode) hoặc 'Unknown' nếu không có mode.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu sau khi lọc đơn hợp lệ.

    Trả về:
        pd.DataFrame: DataFrame đã được khử trùng lặp và điền khuyết thiếu hoàn toàn.
    """
    # TODO: [Nguyễn Minh Nhật - 14%] Lập trình phần khử trùng lặp và điền khuyết:
    # 1. Khử trùng lặp: df_clean = df.drop_duplicates().reset_index(drop=True).copy()
    # 2. Xử lý 'promotion-ids': df_clean['promotion-ids'] = df_clean['promotion-ids'].fillna('No_Promotion')
    # 3. Xử lý 'Courier Status': df_clean['Courier Status'] = df_clean['Courier Status'].fillna('Unshipped')
    # 4. Xử lý 'ship-state': tìm mode của cột ship-state không rỗng, fillna bằng giá trị mode này.
    # 5. Trả về df_clean.
    raise NotImplementedError(
        "Chưa cài đặt - Nguyễn Minh Nhật (14%) cần hoàn thiện hàm 'handle_missing_and_duplicates' theo phân công trong docs/description/TASKS.md"
    )


def clean_amazon_sales_data(
    raw_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Hàm đọc dữ liệu thô Amazon Sale Report, làm sạch và xuất dữ liệu chuẩn (102.852 dòng sạch).
    Phụ trách chính: Nguyễn Minh Nhật (14%)

    Quy trình làm sạch:
    1. Đọc file CSV dữ liệu thô từ raw_path.
    2. Chỉ giữ 12 cột thông tin hợp lệ:
       ['Date', 'Status', 'Fulfilment', 'ship-service-level',
        'Category', 'Size', 'Courier Status', 'Qty', 'Amount',
        'ship-state', 'promotion-ids', 'B2B']
    3. Loại bỏ đơn hàng bị hủy: Status chứa 'Cancelled' (không phân biệt hoa thường).
    4. Lọc đơn có doanh số dương Amount > 0 và số lượng Qty > 0.
    5. Gọi hàm handle_missing_and_duplicates(df) để khử trùng lặp và điền khuyết thiếu.
    6. Nếu output_path được chỉ định, lưu kết quả ra file CSV (encoding='utf-8', index=False).
    7. Trả về DataFrame dữ liệu sạch.

    Tham số:
        raw_path (str | Path): Đường dẫn đến tệp CSV dữ liệu thô.
        output_path (str | Path, optional): Đường dẫn lưu tệp CSV kết quả sạch.

    Trả về:
        pd.DataFrame: DataFrame dữ liệu sạch hoàn thiện.
    """
    # TODO: [Nguyễn Minh Nhật - 14%] Lập trình quy trình làm sạch toàn diện:
    # 1. Kiểm tra raw_path có tồn tại không (Path(raw_path).exists())
    # 2. Đọc file raw: df_raw = pd.read_csv(raw_path, low_memory=False)
    # 3. Lọc 12 cột hợp lệ: keep_cols = [...]
    # 4. Lọc đơn hủy: ~df['Status'].astype(str).str.contains('Cancelled', case=False, na=False)
    # 5. Lọc Amount > 0 và Qty > 0
    # 6. Gọi handle_missing_and_duplicates(df)
    # 7. Xuất file nếu output_path is not None
    # 8. Trả về df_clean
    raise NotImplementedError(
        "Chưa cài đặt - Nguyễn Minh Nhật (14%) cần hoàn thiện hàm 'clean_amazon_sales_data' theo phân công trong docs/description/TASKS.md"
    )


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_file = base_dir / "data" / "raw" / "Amazon Sale Report.csv"
    out_file = base_dir / "data" / "processed" / "clean_amazon_sales.csv"
    print("Bắt đầu tiền xử lý dữ liệu Amazon Sale Report...")
    try:
        df_result = clean_amazon_sales_data(raw_file, out_file)
        print(f"Hoàn thành làm sạch: {len(df_result):,} dòng dữ liệu sạch được lưu tại: {out_file}")
    except NotImplementedError as e:
        print(f"⚠️ {e}")
