"""
Gói các kịch bản xử lý dữ liệu và EDA — Practice 3
"""

from .data_preprocessing import load_raw_data, check_data_quality, clean_customer_data
from .eda import run_eda, compute_eda_summary, generate_eda_figures

__all__ = [
    "load_raw_data", "check_data_quality", "clean_customer_data",
    "run_eda", "compute_eda_summary", "generate_eda_figures"
]
