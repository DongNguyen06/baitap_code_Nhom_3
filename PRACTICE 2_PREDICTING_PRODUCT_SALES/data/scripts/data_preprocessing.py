"""
Module: data_preprocessing.py
Dự án: Practice 2: Predicting Product Sales — Group 3
Phụ trách chính: Nguyễn Minh Nhật
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

# Danh sách 12 cột cần giữ lại cho các bước sau
KEEP_COLUMNS = [
    "Date", "Status", "Fulfilment", "ship-service-level", "Category", "Size",
    "Courier Status", "Qty", "Amount", "ship-state", "promotion-ids", "B2B",
]


def handle_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Khử trùng lặp bản ghi và điền khuyết thiếu cho các trường dữ liệu.
    """
    # Xóa các cột rác (index, Unnamed) để drop_duplicates hoạt động chính xác
    df = df.drop(columns=[c for c in df.columns if c == "index" or c.startswith("Unnamed")])
    
    # Chuẩn hóa text: Xóa khoảng trắng thừa và viết IN HOA cột ship-state
    df["ship-state"] = df["ship-state"].str.strip().str.upper()
    
    # Khử trùng lặp
    df = df.drop_duplicates()

    # Điền giá trị mặc định cho 2 cột đặc thù
    df["promotion-ids"] = df["promotion-ids"].fillna("No_Promotion")
    df["Courier Status"] = df["Courier Status"].fillna("Unshipped")

    # Tự động điền khuyết (Số -> trung vị, Chữ -> mode)
    for col in df.columns[df.isna().any()]:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df.reset_index(drop=True)


def clean_amazon_sales_data(
    raw_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Làm sạch toàn diện dữ liệu thô Amazon Sale Report và xuất tệp sạch.
    """
    # 1. Đọc dữ liệu
    df = pd.read_csv(raw_path, low_memory=False)
    
    # 2. Lọc đơn hủy và đơn rác (Amount, Qty <= 0)
    df = df[~df["Status"].str.contains("Cancelled", case=False, na=False)]
    df = df[(df["Amount"] > 0) & (df["Qty"] > 0)]

    # 3. Làm sạch (Xóa trùng lặp, điền khuyết thiếu)
    df = handle_missing_and_duplicates(df)
    
    # 4. Chỉ giữ lại 12 cột cần thiết
    df = df[KEEP_COLUMNS]

    # 5. Kiểm tra an toàn (không còn ô trống)
    assert df.isna().sum().sum() == 0, "Dữ liệu sạch vẫn còn ô khuyết!"

    # 6. Xuất file nếu có đường dẫn
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
    return df


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