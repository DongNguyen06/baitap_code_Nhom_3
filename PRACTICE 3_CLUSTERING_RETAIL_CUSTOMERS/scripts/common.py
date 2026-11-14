"""
Module: common.py
Dự án: Practice 3: Clustering Retail Customers — Group 3
Mục tiêu: Quản lý tập trung đường dẫn hệ thống, cấu hình hằng số, tiện ích lưu/nạp JSON
và các hàm hỗ trợ định dạng dữ liệu cho toàn bộ quy trình.
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union, List, Optional

# Cấu hình UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# -----------------------------------------------------------------------------
# 1. ĐỊNH NGHĨA ĐƯỜNG DẪN TẬP TRUNG (CENTRALIZED PATHS)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_CSV = RAW_DIR / "Mall_Customers.csv"

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_CSV = PROCESSED_DIR / "clean_customers.csv"

DOCS_DIR = BASE_DIR / "docs"
IMAGES_DIR = DOCS_DIR / "images"
REPORTS_DIR = DOCS_DIR / "report"

LOGS_DIR = BASE_DIR / "logs"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"

# -----------------------------------------------------------------------------
# 2. HẰNG SỐ TOÀN DỰ ÁN (PROJECT CONSTANTS)
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
OPTIMAL_K = 5
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 5

FEATURE_COLUMNS_2D = ["Annual_Income_k$", "Spending_Score"]
FEATURE_COLUMNS_3D = ["Age", "Annual_Income_k$", "Spending_Score"]
FEATURE_COLUMNS_4D = ["Age", "Annual_Income_k$", "Spending_Score", "Gender_Code"]

# -----------------------------------------------------------------------------
# 3. TIỆN ÍCH LƯU & NẠP DỮ LIỆU HIỆN VẬT JSON
# -----------------------------------------------------------------------------
def save_json(data: Any, path: Union[str, Path]) -> Path:
    """Lưu dữ liệu sang tệp JSON hỗ trợ Unicode đầy đủ và định dạng tinh gọn."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw_str = json.dumps(data, indent=2, ensure_ascii=False)
    # Nén các mảng số ngắn và mảng 2-6 phần tử trên 1 dòng
    raw_str = re.sub(r'\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', r'[\1, \2]', raw_str)
    raw_str = re.sub(r'\[\s*("[^"]*")\s*,\s*("[^"]*")\s*\]', r'[\1, \2]', raw_str)
    raw_str = re.sub(r'\[\s*("[^"]*")\s*,\s*("[^"]*")\s*,\s*("[^"]*")\s*\]', r'[\1, \2, \3]', raw_str)
    raw_str = re.sub(r'\[\s*("[^"]*")\s*,\s*("[^"]*")\s*,\s*("[^"]*")\s*,\s*("[^"]*")\s*\]', r'[\1, \2, \3, \4]', raw_str)
    raw_str = re.sub(r'\[\s*("[^"]*")\s*,\s*("[^"]*")\s*,\s*("[^"]*")\s*,\s*("[^"]*")\s*,\s*("[^"]*")\s*,\s*("[^"]*")\s*\]', r'[\1, \2, \3, \4, \5, \6]', raw_str)
    # Nén các dictionary 1 phần tử nhỏ: {"n_clusters": 5}
    raw_str = re.sub(r'\{\s*"([a-zA-Z0-9_\-]+)":\s*([0-9\.\-]+|"[^"]*")\s*\}', r'{"\1": \2}', raw_str)
    # Nén các dictionary elbow và silhouette_by_k trên 1 dòng
    raw_str = re.sub(
        r'\{\s*"(?:2|3|4|5|6|7|8|9|10)":\s*[\d\.\-]+(?:,\s*"(?:2|3|4|5|6|7|8|9|10)":\s*[\d\.\-]+){8,9}\s*\}',
        lambda m: re.sub(r'\s+', ' ', m.group(0)),
        raw_str
    )
    with open(p, "w", encoding="utf-8") as f:
        f.write(raw_str)
    return p


def load_json(path: Union[str, Path]) -> Any:
    """Nạp dữ liệu từ tệp JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp JSON tại: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_stage_result(stage_id: int, data: Dict[str, Any]) -> Path:
    """Lưu kết quả của từng chặng vào outputs/stage_0X.json."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"stage_0{stage_id}.json"
    data["stage_id"] = stage_id
    data["timestamp"] = datetime.now().isoformat()
    return save_json(data, out_path)


def load_stage_result(stage_id: int) -> Optional[Dict[str, Any]]:
    """Đọc kết quả của chặng từ outputs/stage_0X.json."""
    stage_path = OUTPUTS_DIR / f"stage_0{stage_id}.json"
    if stage_path.exists():
        return load_json(stage_path)
    return None


def require_files(*paths: Union[str, Path]) -> None:
    """Kiểm tra sự tồn tại bắt buộc của các tệp tiền đề."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu các tệp bắt buộc: {', '.join(missing)}")


def to_rel_path(path: Union[str, Path, None]) -> str:
    """
    Chuyển đổi đường dẫn tuyệt đối sang dạng tương đối so với thư mục gốc dự án
    (ví dụ: data/raw/Mall_Customers.csv, docs/images/eda_01.png).
    """
    if path is None:
        return ""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(BASE_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


# -----------------------------------------------------------------------------
# 4. TIỆN ÍCH ĐỊNH DẠNG DỮ LIỆU
# -----------------------------------------------------------------------------
def format_currency(val: float) -> str:
    """Định dạng thu nhập (k$) thành chuỗi dễ đọc."""
    return f"${val:,.1f}k"


def format_number(val: float, decimals: int = 2) -> str:
    """Định dạng số thực với số chữ số thập phân chỉ định."""
    return f"{val:,.{decimals}f}"
