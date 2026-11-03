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


def handle_missing_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Khử trùng lặp bản ghi và điền khuyết thiếu cho các trường dữ liệu.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu đầu vào.

    Trả về:
        pd.DataFrame: DataFrame đã được khử trùng lặp và điền khuyết thiếu.
    """
    # TODO: [Nguyễn Minh Nhật] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Nguyễn Minh Nhật cần hoàn thiện hàm 'handle_missing_and_duplicates' theo phân công trong docs/description/TASKS.md"
    )


def clean_amazon_sales_data(
    raw_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Làm sạch toàn diện dữ liệu thô Amazon Sale Report và xuất tệp sạch.

    Tham số:
        raw_path (str | Path): Đường dẫn đến tệp CSV dữ liệu thô.
        output_path (str | Path, optional): Đường dẫn lưu tệp CSV kết quả sạch.

    Trả về:
        pd.DataFrame: DataFrame dữ liệu sạch hoàn thiện.
    """
    # TODO: [Nguyễn Minh Nhật] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Nguyễn Minh Nhật cần hoàn thiện hàm 'clean_amazon_sales_data' theo phân công trong docs/description/TASKS.md"
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
