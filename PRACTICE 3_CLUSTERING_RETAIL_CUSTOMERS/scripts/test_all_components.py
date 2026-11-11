"""
Bộ Kiểm thử Tự động Toàn diện Hệ thống — PRACTICE 3: CLUSTERING RETAIL CUSTOMERS
Học máy (Machine Learning Lab) — Nhóm 3 (UTH)

8 Nhóm thành phần kiểm thử:
1. data/scripts/data_preprocessing.py (Làm sạch, định dạng, kiểm tra missing)
2. data/scripts/eda.py (Thống kê mô tả, phân phối, tương quan, trực quan hóa)
3. scripts/feature_engineering.py (Mã hóa, co giãn thang đo, ma trận đặc trưng)
4. models/algorithms/ (3 thuật toán Scratch: K-Means, DBSCAN, Hierarchical)
5. scripts/model_training.py (Huấn luyện Scratch & Sklearn, Elbow/Silhouette tìm K tối ưu)
6. scripts/model_evaluation.py (Silhouette, Davies-Bouldin, Calinski-Harabasz, Dendrogram, Scatter 2D/3D)
7. models/saved_models/ & Demo tương tác (Lưu trữ mô hình, metadata, pipeline)
8. Cú pháp AST toàn bộ file Python và code cells Notebooks (Toàn dự án)
"""

import argparse
import ast
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_stage_1_scaffolding():
    """Kiểm tra tính đầy đủ của khung sườn dự án và dữ liệu thô ban đầu (Stage 1)."""
    print("\n[Giai đoạn 1] Kiểm tra Khung sườn cấu trúc thư mục, tệp rỗng & Dữ liệu thô...")
    
    expected_dirs = [
        "data/raw", "data/processed", "data/scripts",
        "docs/description/algorithms", "docs/images", "docs/report",
        "logs", "models/algorithms", "models/saved_models",
        "notebooks", "outputs", "scripts"
    ]
    for d in expected_dirs:
        p = BASE_DIR / d
        assert p.exists() and p.is_dir(), f"Thiếu thư mục bắt buộc: {d}"
    print("  ✓ Toàn bộ 12 thư mục chức năng đã tồn tại đầy đủ.")

    expected_files = [
        ".gitignore", "requirements.txt",
        "data/raw/Mall_Customers.csv",
        "data/scripts/__init__.py", "data/scripts/data_preprocessing.py", "data/scripts/eda.py",
        "docs/description/project_description.md", "docs/description/project_design.md",
        "docs/description/algorithms/k_means_clustering.md",
        "docs/description/algorithms/dbscan_clustering.md",
        "docs/description/algorithms/hierarchical_clustering.md",
        "models/__init__.py", "models/algorithms/__init__.py",
        "models/algorithms/k_means.py", "models/algorithms/dbscan.py", "models/algorithms/hierarchical.py",
        "notebooks/01_exploratory_data_analysis.ipynb",
        "notebooks/02_data_preprocessing_and_feature_engineering.ipynb",
        "notebooks/03_model_training_evaluation_and_explain.ipynb",
        "notebooks/demo.ipynb",
        "scripts/__init__.py", "scripts/common.py", "scripts/logger_utils.py",
        "scripts/feature_engineering.py", "scripts/model_training.py", "scripts/model_evaluation.py",
        "scripts/report_builder.py", "scripts/run_01_eda.py", "scripts/run_02_fe.py",
        "scripts/run_03_train.py", "scripts/run_04_eval.py", "scripts/run_pipeline.py",
        "scripts/test_all_components.py"
    ]
    for f in expected_files:
        p = BASE_DIR / f
        assert p.exists() and p.is_file(), f"Thiếu tệp bắt buộc: {f}"
    print(f"  ✓ Toàn bộ {len(expected_files)} tệp khung sườn (placeholders) đã tồn tại hợp lệ.")

    # Kiểm tra dữ liệu thô
    raw_path = BASE_DIR / "data" / "raw" / "Mall_Customers.csv"
    df_raw = pd.read_csv(raw_path)
    assert df_raw.shape == (200, 5), f"Kích thước dữ liệu thô sai lệch: {df_raw.shape} (kỳ vọng 200, 5)"
    expected_cols = ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
    assert list(df_raw.columns) == expected_cols, f"Tên cột dữ liệu thô sai lệch: {df_raw.columns.tolist()}"
    print(f"  ✓ Tệp dữ liệu thô 'Mall_Customers.csv' đạt chuẩn 100%: 200 dòng, 5 cột.")

    return "PASSED"


def test_01_data_preprocessing():
    print("\n[1/8] Kiểm thử module data/scripts/data_preprocessing.py...")
    prep_path = BASE_DIR / "data" / "scripts" / "data_preprocessing.py"
    code = prep_path.read_text(encoding="utf-8")
    if "def clean_customer_data" not in code:
        print("  ⏳ [TODO] Module chưa được cài đặt logic đầy đủ (đang ở dạng khung sườn).")
        return "TODO"
    
    from data.scripts.data_preprocessing import load_raw_data, clean_customer_data
    df = load_raw_data()
    assert df is not None and len(df) == 200, "Lỗi load_raw_data"
    df_clean, report = clean_customer_data(df)
    assert "Annual_Income_k$" in df_clean.columns or "Annual_Income" in df_clean.columns, "Chưa chuẩn hóa tên cột"
    print("  ✓ [PASSED] Module data_preprocessing hoạt động chính xác.")
    return "PASSED"


def test_02_eda():
    print("\n[2/8] Kiểm thử module data/scripts/eda.py...")
    eda_path = BASE_DIR / "data" / "scripts" / "eda.py"
    code = eda_path.read_text(encoding="utf-8")
    if "def generate_eda_figures" not in code and "def run_eda" not in code:
        print("  ⏳ [TODO] Module chưa được cài đặt logic đầy đủ (đang ở dạng khung sườn).")
        return "TODO"
    
    from data.scripts.eda import run_eda
    summary = run_eda()
    assert summary is not None, "run_eda không trả về kết quả"
    print("  ✓ [PASSED] Module eda hoạt động chính xác.")
    return "PASSED"


def test_03_feature_engineering():
    print("\n[3/8] Kiểm thử module scripts/feature_engineering.py...")
    fe_path = BASE_DIR / "scripts" / "feature_engineering.py"
    code = fe_path.read_text(encoding="utf-8")
    if "def prepare_clustering_features" not in code and "def build_feature_pipeline" not in code:
        print("  ⏳ [TODO] Module chưa được cài đặt logic đầy đủ (đang ở dạng khung sườn).")
        return "TODO"
    
    from scripts.feature_engineering import prepare_clustering_features
    X, scaler, feature_names = prepare_clustering_features()
    assert X.shape[0] == 200, "Số dòng ma trận đặc trưng không khớp 200"
    print("  ✓ [PASSED] Module feature_engineering hoạt động chính xác.")
    return "PASSED"


def test_04_algorithms():
    print("\n[4/8] Kiểm thử các thuật toán Gom cụm Scratch trong models/algorithms/...")
    km_path = BASE_DIR / "models" / "algorithms" / "k_means.py"
    db_path = BASE_DIR / "models" / "algorithms" / "dbscan.py"
    hc_path = BASE_DIR / "models" / "algorithms" / "hierarchical.py"
    
    km_code = km_path.read_text(encoding="utf-8")
    db_code = db_path.read_text(encoding="utf-8")
    hc_code = hc_path.read_text(encoding="utf-8")
    
    if "class KMeansClustering" not in km_code or "class DBSCANClustering" not in db_code or "class HierarchicalClustering" not in hc_code:
        print("  ⏳ [TODO] Các lớp thuật toán Scratch chưa được định nghĩa đầy đủ.")
        return "TODO"
    
    from models.algorithms.k_means import KMeansClustering
    from models.algorithms.dbscan import DBSCANClustering
    from models.algorithms.hierarchical import HierarchicalClustering
    
    # Dữ liệu giả lập 2D
    np.random.seed(42)
    X_dummy = np.vstack([
        np.random.normal(loc=[0, 0], scale=0.5, size=(30, 2)),
        np.random.normal(loc=[5, 5], scale=0.5, size=(30, 2))
    ])
    
    # Test KMeans
    km = KMeansClustering(n_clusters=2, random_state=42)
    km.fit(X_dummy)
    assert len(km.labels_) == 60, "KMeans nhãn không khớp độ dài dữ liệu"
    assert km.cluster_centers_.shape == (2, 2), "KMeans tâm cụm sai shape"
    
    # Test DBSCAN
    db = DBSCANClustering(eps=1.0, min_samples=5)
    db.fit(X_dummy)
    assert len(db.labels_) == 60, "DBSCAN nhãn không khớp độ dài dữ liệu"
    
    # Test Hierarchical
    hc = HierarchicalClustering(n_clusters=2, linkage="ward")
    hc.fit(X_dummy)
    assert len(hc.labels_) == 60, "Hierarchical nhãn không khớp độ dài dữ liệu"
    
    print("  ✓ [PASSED] Toàn bộ 3 thuật toán Scratch (K-Means, DBSCAN, Hierarchical) đạt chuẩn 100%.")
    return "PASSED"


def test_05_model_training():
    print("\n[5/8] Kiểm thử module scripts/model_training.py...")
    train_path = BASE_DIR / "scripts" / "model_training.py"
    code = train_path.read_text(encoding="utf-8")
    if "def train_all_models" not in code:
        print("  ⏳ [TODO] Module chưa được cài đặt logic đầy đủ (đang ở dạng khung sườn).")
        return "TODO"
    
    from scripts.model_training import train_all_models
    results = train_all_models()
    assert "k_means" in results or "kmeans" in results, "Chưa huấn luyện K-Means"
    print("  ✓ [PASSED] Module model_training hoạt động chính xác.")
    return "PASSED"


def test_06_model_evaluation():
    print("\n[6/8] Kiểm thử module scripts/model_evaluation.py...")
    eval_path = BASE_DIR / "scripts" / "model_evaluation.py"
    code = eval_path.read_text(encoding="utf-8")
    if "def evaluate_clustering" not in code:
        print("  ⏳ [TODO] Module chưa được cài đặt logic đầy đủ (đang ở dạng khung sườn).")
        return "TODO"
    
    from scripts.model_evaluation import evaluate_clustering
    eval_res = evaluate_clustering()
    assert eval_res is not None, "Kết quả đánh giá không hợp lệ"
    print("  ✓ [PASSED] Module model_evaluation hoạt động chính xác.")
    return "PASSED"


def test_07_saved_models_and_demo():
    print("\n[7/8] Kiểm thử hiện vật lưu trữ models/saved_models/ và Demo suy luận...")
    saved_dir = BASE_DIR / "models" / "saved_models"
    best_model_path = saved_dir / "best_model.joblib"
    meta_path = saved_dir / "metadata.json"
    
    if not best_model_path.exists() or not meta_path.exists():
        print("  ⏳ [TODO] Chưa xuất hiện best_model.joblib hoặc metadata.json (sẽ tạo ở Bước 3.4 & 5).")
        return "TODO"
    
    print("  ✓ [PASSED] Các hiện vật mô hình đã được lưu trữ đầy đủ.")
    return "PASSED"


def test_08_ast_syntax_integrity():
    """Kiểm tra cú pháp AST toàn bộ mã nguồn Python và toàn bộ code cells Notebooks."""
    print("\n[8/8] Kiểm tra cú pháp AST toàn bộ file Python và Jupyter Notebooks...")
    
    # 1. Quét tất cả file .py
    py_files = list(BASE_DIR.rglob("*.py"))
    assert len(py_files) > 0, "Không tìm thấy file python nào"
    for py_file in py_files:
        if "__pycache__" in str(py_file) or ".pytest_cache" in str(py_file):
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            ast.parse(content, filename=str(py_file))
        except SyntaxError as e:
            raise AssertionError(f"Lỗi cú pháp trong file {py_file.relative_to(BASE_DIR)}: {e}")
    print(f"  ✓ {len(py_files)} tệp Python đạt chuẩn cú pháp AST 100% không có lỗi.")

    # 2. Quét tất cả Jupyter Notebooks
    nb_files = list((BASE_DIR / "notebooks").glob("*.ipynb"))
    assert len(nb_files) == 4, f"Số lượng notebook không đúng: {len(nb_files)} (kỳ vọng 4)"
    for nb_file in nb_files:
        try:
            with open(nb_file, "r", encoding="utf-8") as f:
                nb_data = json.load(f)
            cells = nb_data.get("cells", [])
            for c_idx, cell in enumerate(cells):
                if cell.get("cell_type") == "code":
                    code_src = "".join(cell.get("source", []))
                    # Làm sạch magic commands trước khi parse AST
                    clean_lines = [l for l in code_src.splitlines() if not l.strip().startswith(("%", "!"))]
                    ast.parse("\n".join(clean_lines))
                    assert cell.get("outputs") == [], f"Ô code #{c_idx} trong {nb_file.name} chưa xóa output"
        except Exception as e:
            raise AssertionError(f"Lỗi trong notebook {nb_file.name}: {e}")
    print(f"  ✓ Toàn bộ 4 Jupyter Notebooks đạt chuẩn cú pháp AST và outputs rỗng.")

    return "PASSED"


def main():
    parser = argparse.ArgumentParser(description="Bộ kiểm thử tự động toàn diện — Practice 3")
    parser.add_argument("--step", type=int, choices=range(1, 9), help="Chạy kiểm thử riêng bước 1 đến 8")
    parser.add_argument("--stage", type=int, choices=[1], help="Chạy kiểm tra giai đoạn cụ thể (1: Khung sườn)")
    args = parser.parse_args()

    print("=" * 80)
    print("   🚀 KIỂM THỬ TỰ ĐỘNG HỆ THỐNG — PRACTICE 3: CLUSTERING RETAIL CUSTOMERS")
    print("=" * 80)

    if args.stage == 1:
        status_stage1 = test_stage_1_scaffolding()
        status_ast = test_08_ast_syntax_integrity()
        print("\n" + "-" * 80)
        if status_stage1 == "PASSED" and status_ast == "PASSED":
            print("🎉 CHÚC MỪNG: GIAI ĐOẠN 1 (KHUNG SƯỜN & DỮ LIỆU) ĐÃ ĐẠT 100% YÊU CẦU!")
        else:
            print("❌ GIAI ĐOẠN 1 CẦN ĐIỀU CHỈNH LẠI.")
        print("-" * 80)
        return

    steps = {
        1: ("Tiền xử lý dữ liệu", "data/scripts/data_preprocessing.py", test_01_data_preprocessing),
        2: ("Khám phá dữ liệu EDA", "data/scripts/eda.py", test_02_eda),
        3: ("Kỹ thuật đặc trưng", "scripts/feature_engineering.py", test_03_feature_engineering),
        4: ("3 Thuật toán Scratch", "models/algorithms/", test_04_algorithms),
        5: ("Huấn luyện mô hình", "scripts/model_training.py", test_05_model_training),
        6: ("Đánh giá gom cụm", "scripts/model_evaluation.py", test_06_model_evaluation),
        7: ("Hiện vật & Demo", "models/saved_models/", test_07_saved_models_and_demo),
        8: ("Kiểm tra AST toàn dự án", "notebooks/ & scripts/", test_08_ast_syntax_integrity),
    }

    if args.step:
        name, target, func = steps[args.step]
        print(f"Đang kiểm tra Bước {args.step}: {name} ({target})...")
        status = func()
        print("\n" + "-" * 80)
        if status == "PASSED":
            print(f"🎉 BƯỚC {args.step} ({name}) ĐÃ VƯỢT QUA KIỂM THỬ 100%!")
        elif status == "TODO":
            print(f"⏳ BƯỚC {args.step} ({name}) ĐANG CHỜ HOÀN THIỆN (ĐÃ CÓ FILE KHUNG SƯỜN).")
        else:
            print(f"❌ BƯỚC {args.step} ({name}) BỊ LỖI ASSERTION.")
        print("-" * 80)
    else:
        results = {}
        for s in range(1, 9):
            name, target, func = steps[s]
            results[s] = (name, target, func())

        print("\n" + "=" * 80)
        print("   📊 BẢNG TỔNG HỢP TIẾN ĐỘ THỰC HIỆN — PRACTICE 3 (NHÓM 3 - UTH)")
        print("=" * 80)
        print(f" {'Bước':<5} | {'Hạng mục công việc':<25} | {'Module / Tệp':<30} | {'Trạng thái'}")
        print("-" * 88)
        for s, (name, target, status) in results.items():
            if status == "PASSED":
                status_str = "✅ ĐÃ HOÀN THIỆN"
            elif status == "TODO":
                status_str = "⏳ CHỜ TRIỂN KHAI (TODO)"
            else:
                status_str = "❌ CẦN ĐIỀU CHỈNH"
            print(f" {s:<5} | {name:<25} | {target:<30} | {status_str}")
        print("=" * 88)
        print("👉 LƯU Ý TIẾN ĐỘ:")
        print("   - Chạy kiểm thử riêng từng bước: python scripts/test_all_components.py --step <1-8>")
        print("   - Chạy kiểm thử xác nhận khung sườn: python scripts/test_all_components.py --stage 1")
        print("=" * 88)


if __name__ == "__main__":
    main()
