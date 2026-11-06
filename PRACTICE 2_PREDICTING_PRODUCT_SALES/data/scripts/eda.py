from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


# -----------------------------------------------------------------------------
# CÁC HÀM TRỰC QUAN HÓA DỮ LIỆU EDA (PLOTTING FUNCTIONS)
# -----------------------------------------------------------------------------
def plot_amount_distribution(df: pd.DataFrame, save_path=None) -> Any:
    """Trực quan hóa đối chiếu phân phối trước và sau biến đổi Log1p."""
    s = pd.to_numeric(df['Amount'], errors='coerce').dropna()
    s = s[s > 0]
    s_log = np.log1p(s)
    skew_raw = float(s.skew())
    skew_log = float(s_log.skew())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(s, bins=50, kde=True, ax=axes[0], color='#2563eb')
    axes[0].set_title(f'Phân phối Doanh số Thô (Skewness: {skew_raw:.2f})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Amount (INR)')
    axes[0].set_ylabel('Tần suất')

    sns.histplot(s_log, bins=50, kde=True, ax=axes[1], color='#059669')
    axes[1].set_title(f'Phân phối Doanh số sau Biến đổi Log1p (Skewness: {skew_log:.2f})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('log1p(Amount)')
    axes[1].set_ylabel('Tần suất')

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path
    return fig


def plot_bivariate_category_fulfilment(df: pd.DataFrame, save_path=None) -> Any:
    """Trực quan hóa doanh số theo Top ngành hàng và Kênh hoàn tất."""
    df_sub = df.dropna(subset=['Amount', 'Category', 'Fulfilment'])
    res = get_bivariate_analysis(df_sub, target='Amount', cat_cols=['Category', 'Fulfilment'])
    cat_df = res.get('category_stats', pd.DataFrame())
    ful_df = res.get('fulfilment_stats', pd.DataFrame())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    top_cats = cat_df.head(6)
    axes[0].barh(top_cats['Category'][::-1], top_cats['total'][::-1] / 1e6, color='#2563eb')
    axes[0].set_title('Top Ngành hàng theo Doanh số (Triệu INR)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Doanh số (Triệu INR)')

    axes[1].bar(ful_df['Fulfilment'], ful_df['total'] / 1e6, color=['#059669', '#d97706'], width=0.5)
    axes[1].set_title('Doanh số theo Kênh Hoàn tất (Triệu INR)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Doanh số (Triệu INR)')

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path
    return fig


def plot_outliers_boxplot(df: pd.DataFrame, save_path=None) -> Any:
    """Trực quan hóa quét ngoại lai bằng biểu đồ Boxplot IQR."""
    s = pd.to_numeric(df['Amount'], errors='coerce').dropna()
    s = s[s > 0]
    outliers_info = detect_iqr_outliers(s)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(x=s, ax=ax, color='#93c5fd', flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3, 'markerfacecolor': '#ef4444'})
    ax.axvline(outliers_info['upper_bound'], color='#dc2626', linestyle='--', linewidth=1.5, label=f"Ngưỡng ngoại lai trên: {outliers_info['upper_bound']:.2f} INR")
    ax.set_title(f'Biểu đồ Boxplot Quét Ngoại lai Doanh số (IQR = {outliers_info["iqr"]:.2f} INR)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Amount (INR)')
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path
    return fig


def plot_temporal_sales_trend(df: pd.DataFrame, save_path=None) -> Any:
    """Biểu đồ xu hướng doanh số theo Thời gian & Tính mùa vụ (Monthly & Day of Week Trend)."""
    data = df.copy()
    if 'Date' in data.columns:
        dates = pd.to_datetime(data['Date'], format='mixed', errors='coerce')
        data['Month'] = dates.dt.strftime('%B')
        data['DayName'] = dates.dt.day_name()
    else:
        data['Month'] = 'Tháng 4'
        data['DayName'] = 'Thứ Hai'

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Doanh số theo Tháng
    month_order = ['April', 'May', 'June']
    month_sales = data.groupby('Month')['Amount'].sum() / 1e6
    month_sales = month_sales.reindex([m for m in month_order if m in month_sales.index])
    axes[0].plot(month_sales.index, month_sales.values, marker='o', linewidth=2.5, markersize=8, color='#2563eb')
    axes[0].set_title('Xu Hướng Doanh Số Theo Tháng (Triệu INR)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Tháng')
    axes[0].set_ylabel('Tổng Doanh Thu (Triệu INR)')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Doanh số trung bình theo Ngày trong tuần
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_sales = data.groupby('DayName')['Amount'].mean()
    day_sales = day_sales.reindex([d for d in day_order if d in day_sales.index])
    colors = ['#3b82f6' if d not in ['Saturday', 'Sunday'] else '#f59e0b' for d in day_sales.index]
    axes[1].bar(day_sales.index, day_sales.values, color=colors, width=0.6)
    axes[1].set_title('Doanh Số Trung Bình Theo Ngày Trong Tuần (Weekend Effect)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Ngày Trong Tuần')
    axes[1].set_ylabel('Doanh Số Trung Bình (INR)')
    axes[1].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path
    return fig


def plot_geographic_sales_distribution(df: pd.DataFrame, save_path=None) -> Any:
    """Biểu đồ phân bổ doanh số theo Top 10 bang tại Ấn Độ."""
    data = df.dropna(subset=['Amount', 'ship-state'])
    state_sales = data.groupby('ship-state')['Amount'].agg(['count', 'sum']).reset_index()
    state_sales['total_m'] = state_sales['sum'] / 1e6
    state_sales = state_sales.sort_values('total_m', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(state_sales['ship-state'][::-1], state_sales['total_m'][::-1], color='#3b82f6', height=0.65)
    ax.set_title('Top 10 Bang Có Doanh Thu Cao Nhất Tại Ấn Độ (Customer Geography)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Tổng Doanh Thu (Triệu INR)', fontsize=11)
    ax.set_ylabel('Bang Nhận Hàng (Ship-State)', fontsize=11)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f'{w:.1f}M', (w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path=None) -> Any:
    """Biểu đồ ma trận nhiệt tương quan giữa các đặc trưng số và log_Amount."""
    data = df.copy()
    if 'Amount' in data.columns:
        data['log_Amount'] = np.log1p(pd.to_numeric(data['Amount'], errors='coerce').fillna(0))

    if 'Date' in data.columns:
        dates = pd.to_datetime(data['Date'], format='mixed', errors='coerce')
        data['month'] = dates.dt.month.fillna(4).astype(int)
        data['day_of_week'] = dates.dt.dayofweek.fillna(5).astype(int)
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    if 'promotion-ids' in data.columns and 'is_promoted' not in data.columns:
        data['is_promoted'] = data['promotion-ids'].notna().astype(int)

    if 'B2B' in data.columns and 'is_b2b' not in data.columns:
        data['is_b2b'] = data['B2B'].astype(int)

    num_cols = ['Qty', 'month', 'day_of_week', 'is_weekend', 'is_promoted', 'is_b2b', 'log_Amount']
    num_cols = [c for c in num_cols if c in data.columns]
    corr = data[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, ax=ax, vmin=-1, vmax=1,
                linewidths=0.5, annot_kws={'size': 10, 'weight': 'bold'})
    ax.set_title('Ma Trận Tương Quan Đặc Trưng Số & Log(Amount)', fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path
    return fig
