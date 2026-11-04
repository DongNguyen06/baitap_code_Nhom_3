from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def inspect_data_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Khảo sát cấu trúc tổng quan dữ liệu (kích thước, kiểu dữ liệu, tỷ lệ khuyết thiếu)."""
    n_rows, n_cols = df.shape
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = {k: round(v / max(1, n_rows) * 100, 2) for k, v in missing_counts.items()}

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_counts": missing_counts,
        "missing_pct": missing_pct,
        "n_duplicates": int(df.duplicated().sum())
    }


def get_univariate_stats(df: pd.DataFrame, col: str = 'Amount') -> Dict[str, Any]:
    """Tính toán thống kê mô tả đơn biến cho biến số (mean, median, std, min, max, skewness, iqr)."""
    if col not in df.columns:
        raise KeyError(f"Cột '{col}' không tồn tại trong DataFrame")

    s = df[col].dropna()
    skew_val = float(s.skew()) if len(s) > 2 else 0.0
    kurt_val = float(s.kurtosis()) if len(s) > 3 else 0.0

    return {
        "column": col,
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "skewness": round(skew_val, 4),
        "kurtosis": round(kurt_val, 4),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75))
    }

get_amount_univariate_stats = get_univariate_stats


def detect_iqr_outliers(series: pd.Series, k: float = 1.5) -> Dict[str, Any]:
    """Nhận diện giá trị ngoại lai theo quy tắc khoảng tứ phân vị IQR (Interquartile Range)."""
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        raise ValueError("Chuỗi dữ liệu rỗng hoặc không chứa giá trị số hợp lệ")

    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask_low = s < lower
    mask_high = s > upper
    mask = mask_low | mask_high
    outliers = s[mask]
    n_total = int(len(s))

    return {
        "column": series.name,
        "n_total": n_total,
        "k": float(k),
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "n_outliers": int(mask.sum()),
        "outlier_pct": round(float(mask.sum()) / n_total * 100, 2),
        "n_lower_outliers": int(mask_low.sum()),
        "n_upper_outliers": int(mask_high.sum()),
        "outlier_min": float(outliers.min()) if len(outliers) else None,
        "outlier_max": float(outliers.max()) if len(outliers) else None,
        "outlier_index": outliers.index.tolist()
    }

detect_sales_outliers_iqr = detect_iqr_outliers


def get_bivariate_analysis(
    df: pd.DataFrame,
    target: str = 'Amount',
    cat_cols: Optional[List[str]] = None,
    top_n_states: int = 10
) -> Dict[str, Any]:
    """Phân tích quan hệ hai biến giữa doanh số với các biến phân loại chính."""
    if target not in df.columns:
        raise KeyError(f"Cột '{target}' không tồn tại trong DataFrame")

    data = df.copy()
    data[target] = pd.to_numeric(data[target], errors='coerce')
    data = data.dropna(subset=[target])

    if 'promotion-ids' in data.columns and 'is_promoted' not in data.columns:
        data['is_promoted'] = data['promotion-ids'].notna().astype(int)

    if cat_cols is None:
        cat_cols = ['Category', 'ship-state', 'Fulfilment', 'ship-service-level',
                    'B2B', 'Size', 'is_promoted']
    cat_cols = [c for c in cat_cols if c in data.columns]
    if not cat_cols:
        raise KeyError("Không tìm thấy biến phân loại nào để phân tích")

    y = data[target]
    ss_total = float(((y - y.mean()) ** 2).sum())

    result: Dict[str, Any] = {}
    eta: Dict[str, float] = {}

    for col in cat_cols:
        grp = data.groupby(col, dropna=False)[target]

        table = grp.agg(count='count', mean='mean', median='median',
                        std='std', total='sum').reset_index()
        table['share_pct'] = (table['total'] / table['total'].sum() * 100).round(2)
        stat_cols = ['mean', 'median', 'std', 'total']
        table[stat_cols] = table[stat_cols].round(2)
        table = table.sort_values('total', ascending=False).reset_index(drop=True)
        if col == 'ship-state':
            table = table.head(top_n_states)

        stat_key = f"{col.lower().replace('-', '_')}_stats"
        result[stat_key] = table
        result[col] = table  # Hỗ trợ cả key gốc để linh hoạt tuyệt đối

        if ss_total == 0:
            eta[col] = 0.0
        else:
            ss_between = float((grp.count() * (grp.mean() - y.mean()) ** 2).sum())
            eta[col] = round(ss_between / ss_total, 4)

    result["eta_squared"] = dict(sorted(eta.items(), key=lambda x: x[1], reverse=True))
    result["ranking"] = list(result["eta_squared"].keys())

    if 'Qty' in data.columns:
        qty = pd.to_numeric(data['Qty'], errors='coerce')
        ok = qty.notna()
        result["numeric_corr"] = {
            "Qty_vs_target_pearson": round(float(qty[ok].corr(y[ok])), 4),
            "Qty_vs_target_spearman": round(float(qty[ok].corr(y[ok], method='spearman')), 4),
        }

    return result

analyze_bivariate_sales = get_bivariate_analysis
