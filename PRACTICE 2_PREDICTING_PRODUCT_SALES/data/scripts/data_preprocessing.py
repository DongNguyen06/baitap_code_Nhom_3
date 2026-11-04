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
    """Khử trùng lặp bản ghi và điền khuyết thiếu cho các trường dữ liệu."""
    df_clean = df.drop_duplicates().copy()

    if "promotion-ids" in df_clean.columns:
        df_clean["promotion-ids"] = df_clean["promotion-ids"].fillna("No_Promotion")

    if "Courier Status" in df_clean.columns:
        df_clean["Courier Status"] = df_clean["Courier Status"].fillna("Unshipped")

    if "ship-state" in df_clean.columns:
        mode_series = df_clean["ship-state"].mode().dropna()
        mode_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
        df_clean["ship-state"] = df_clean["ship-state"].fillna(mode_val)

    return df_clean


def clean_amazon_sales_data(
    raw_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """Làm sạch toàn diện dữ liệu thô Amazon Sale Report và xuất tệp sạch."""
    df_raw = pd.read_csv(raw_path, low_memory=False)

    cols = [
        'Date', 'Status', 'Fulfilment', 'ship-service-level', 'Category',
        'Size', 'Courier Status', 'Qty', 'Amount', 'ship-state',
        'promotion-ids', 'B2B'
    ]
    cols_present = [c for c in cols if c in df_raw.columns]

    df_filtered = df_raw.copy()
    if 'Status' in df_filtered.columns:
        df_filtered = df_filtered[~df_filtered['Status'].astype(str).str.contains('Cancelled', case=False, na=False)]
    if 'Amount' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Amount'] > 0]
    if 'Qty' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Qty'] > 0]

    if cols_present:
        df_filtered = df_filtered[cols_present]

    df_clean = handle_missing_and_duplicates(df_filtered)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(p, index=False)

    return df_clean


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_file = base_dir / "data" / "raw" / "Amazon Sale Report.csv"
    out_file = base_dir / "data" / "processed" / "clean_amazon_sales.csv"
    print("Bắt đầu tiền xử lý dữ liệu Amazon Sale Report...")
    df_result = clean_amazon_sales_data(raw_file, out_file)
    print(f"Hoàn thành làm sạch: {len(df_result):,} dòng dữ liệu sạch được lưu tại: {out_file}")
