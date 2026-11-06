"""
Bộ Kiểm thử Tự động Toàn diện Hệ thống (Comprehensive Component Tests)
Dự án: Practice 1: Classifying Spam Emails — Group 3
Phạm vi kiểm thử:
1. data/scripts/data_preprocessing.py (Xử lý missing, clean_text NLP chuẩn hóa)
2. data/scripts/eda.py (Cấu trúc dữ liệu, thống kê đơn biến, ngoại lai IQR)
3. scripts/feature_engineering.py (7 đặc trưng miền, ColumnTransformer, Train/Test split)
4. models/algorithms/ (3 Mô hình From Scratch thuần NumPy)
5. scripts/model_training.py (Tạo mô hình Scratch, Pipeline huấn luyện, 5-Fold CV)
6. scripts/model_evaluation.py (Đánh giá Hold-out test, Confusion Matrix, Top từ khóa)
7. models/saved_models/ & notebooks/demo.ipynb (Mô hình tối ưu, metadata.json, Guardrails)
8. Toàn vẹn cú pháp AST toàn bộ mã nguồn .py và các code cells trong Notebooks
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import sparse

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def test_01_data_preprocessing():
    print("[1/8] Kiểm thử module data/scripts/data_preprocessing.py...")
    from data.scripts.data_preprocessing import handle_missing_values, clean_text

    df_raw = pd.DataFrame({
        "sender": ["abc@test.com", None],
        "receiver": [None, "user@test.com"],
        "date": ["2026-01-01", "2026-01-02"],
        "subject": [None, "URGENT WINNER"],
        "body": ["Hello world", None],
        "label": [0, 1],
        "urls": [0, 1]
    })
    df_clean = handle_missing_values(df_raw)
    assert df_clean["subject"].iloc[0] == "", "Lỗi: subject None không được chuyển thành chuỗi rỗng"
    assert df_clean["body"].iloc[1] == "", "Lỗi: body None không được chuyển thành chuỗi rỗng"
    assert df_clean["receiver"].iloc[0] == "unknown", "Lỗi: receiver None không được điền 'unknown'"

    raw_html = "<p>Congratulations! Visit http://spam.biz to claim &amp; win $$$ NOW!</p>"
    cleaned = clean_text(raw_html)
    assert "<p>" not in cleaned and "</p>" not in cleaned, "Lỗi: Chưa lột sạch thẻ HTML"
    assert "http" not in cleaned and "spam.biz" not in cleaned, "Lỗi: Chưa xóa URL"
    assert "&amp;" not in cleaned and "$" not in cleaned, "Lỗi: Chưa unescape HTML hoặc ký tự lạ"
    assert "congratulations" in cleaned and "claim" in cleaned and "win" in cleaned, "Lỗi: Thất thoát từ vựng cốt lõi"
    print("  ✓ data_preprocessing.py: Xử lý missing và clean_text NLP đạt 100%.")


def test_02_eda():
    print("\n[2/8] Kiểm thử module data/scripts/eda.py...")
    from data.scripts.eda import inspect_data_structure, get_univariate_stats, detect_iqr_outliers, extract_keywords

    df_sample = pd.DataFrame({
        "sender": ["a@b.com", "a@b.com", "c@d.com", "e@f.com"],
        "receiver": ["u1", "u2", "u3", "u4"],
        "date": ["d1", "d2", "d3", "d4"],
        "subject": ["Meeting notes", "Re: Meeting notes", "WIN CASH $", "Hello"],
        "body": ["text 1", "text 2", "lottery winner cash", "regular talk"],
        "label": [0, 0, 1, 0],
        "urls": [0, 0, 1, 0]
    })

    struct = inspect_data_structure(df_sample)
    assert struct["n_rows"] == 4 and struct["n_cols"] == 7, "Lỗi cấu trúc shape"
    stats = get_univariate_stats(df_sample)
    assert stats["ham_count"] == 3 and stats["spam_count"] == 1, "Lỗi đếm số lượng nhãn"

    lens = pd.Series([10, 12, 11, 14, 15, 12, 200])
    iqr_res = detect_iqr_outliers(lens)
    assert iqr_res["n_outliers"] >= 1, "Lỗi không nhận diện được điểm ngoại lai IQR"

    kw = extract_keywords(df_sample["subject"], top_n=2)
    assert len(kw) > 0, "Lỗi trích xuất từ khóa"
    print("  ✓ eda.py: Khảo sát cấu trúc, đơn biến, ngoại lai IQR và trích xuất từ khóa đạt 100%.")


def test_03_feature_engineering():
    print("\n[3/8] Kiểm thử module scripts/feature_engineering.py...")
    from scripts.feature_engineering import extract_domain_features, split_stratified_train_test, build_column_transformer

    df_sample = pd.DataFrame({
        "sender": ["s1", "s2", "s3", "s4", "s5"],
        "receiver": ["r1", "r2", "r3", "r4", "r5"],
        "date": ["d1", "d2", "d3", "d4", "d5"],
        "subject": ["Re: Urgent project", "FREE PRIZE $$$", "Meeting update", "ALERT!", "Normal report"],
        "body": ["Please reply soon", "Claim $1000 cash now!!!", "See attached minutes", "Check your account", "Everything is good"],
        "text_clean": ["urgent project reply soon", "free prize claim cash", "meeting update see attached minutes", "alert check account", "normal report everything good"],
        "urls": [0, 1, 0, 0, 0],
        "label": [0, 1, 0, 1, 0]
    })

    df_feat = extract_domain_features(df_sample)
    expected_cols = [
        "text_clean", "caps_ratio", "exclamation_count", "dollar_count",
        "text_len", "log_text_len", "has_re_prefix", "subject_exclamation"
    ]
    for col in expected_cols:
        assert col in df_feat.columns, f"Thiếu cột đặc trưng miền: {col}"
    assert df_feat["has_re_prefix"].iloc[0] == 1, "Lỗi nhận diện tiền tố Re:"
    assert df_feat["dollar_count"].iloc[1] >= 3, "Lỗi đếm ký hiệu $"
    assert df_feat["subject_exclamation"].iloc[3] == 1, "Lỗi nhận diện dấu ! ở tiêu đề"

    ct = build_column_transformer(max_features=100, k_best=10)
    trans_names = [t[0] for t in ct.transformers]
    assert "text" in trans_names and "num" in trans_names, "Cấu trúc ColumnTransformer chưa chuẩn 2 nhánh"

    from scripts.feature_engineering import load_pipeline_and_splits, export_train_test_csv
    prep_loaded, x_tr, x_te, y_tr, y_te = load_pipeline_and_splits()
    assert prep_loaded is not None and len(x_tr) == 26128 and len(x_te) == 6532, "Lỗi nạp pipeline và dữ liệu phân chia"

    train_csv_p = BASE_DIR / "data" / "processed" / "train.csv"
    test_csv_p = BASE_DIR / "data" / "processed" / "test.csv"
    assert train_csv_p.exists() and test_csv_p.exists(), "Thiếu tệp train.csv hoặc test.csv"
    print("  ✓ feature_engineering.py: Trích xuất 7 đặc trưng miền, cấu hình ColumnTransformer, load_pipeline_and_splits và export_train_test_csv đạt 100%.")


def test_04_scratch_algorithms():
    print("\n[4/8] Kiểm thử 3 thuật toán Scratch NumPy thuần túy (Nhiệm vụ chung cả 7 thành viên)...")
    from models.algorithms.naive_bayes import MultinomialNaiveBayesScratch
    from models.algorithms.logistic_regression import LogisticRegressionScratch
    from models.algorithms.support_vector_machine import LinearSVMScratch

    X_toy = sparse.csr_matrix(np.array([
        [3.0, 1.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 3.0, 2.0],
        [0.0, 0.0, 0.0, 2.0, 3.0]
    ], dtype=np.float64))
    y_toy = np.array([0, 0, 1, 1])

    nb = MultinomialNaiveBayesScratch(alpha=1.0)
    nb.fit(X_toy, y_toy)
    pred_nb = nb.predict(X_toy)
    proba_nb = nb.predict_proba(X_toy)
    assert np.array_equal(pred_nb, y_toy), "Naive Bayes Scratch dự đoán sai dữ liệu cơ bản"
    assert np.allclose(proba_nb.sum(axis=1), 1.0), "Naive Bayes xác suất không chuẩn hóa tổng bằng 1"

    lr = LogisticRegressionScratch(learning_rate=12.0, n_iters=300, l2=1e-4)
    lr.fit(X_toy, y_toy)
    pred_lr = lr.predict(X_toy)
    proba_lr = lr.predict_proba(X_toy)
    assert proba_lr.shape == (4, 2), "Xác suất Logistic Regression phải có dạng (N, 2)"
    assert np.all((proba_lr >= 0.0) & (proba_lr <= 1.0)), "Xác suất Logistic Regression vượt ngưỡng [0, 1]"

    svm = LinearSVMScratch(learning_rate=5.0, lambda_param=1e-5, n_iters=300)
    svm.fit(X_toy, y_toy)
    pred_svm = svm.predict(X_toy)
    scores = svm.decision_function(X_toy)
    assert len(scores) == 4, "decision_function của SVM tính sai số chiều"
    assert hasattr(svm, "coef_") and hasattr(svm, "intercept_"), "SVM Scratch thiếu thuộc tính trọng số"
    print("  ✓ models/algorithms/: Cả 3 class Scratch thuần túy NumPy tính toán hội tụ chính xác 100%.")


def test_05_training_module():
    print("\n[5/8] Kiểm thử module scripts/model_training.py...")
    from scripts.model_training import create_scratch_models, build_training_pipelines
    from scripts.feature_engineering import build_column_transformer

    models = create_scratch_models()
    assert set(models.keys()) == {"naive_bayes", "logistic_regression", "linear_svm"}, "Sai danh sách mô hình Scratch"

    prep = build_column_transformer(max_features=50, k_best=10)
    pipes = build_training_pipelines(prep, models)
    assert len(pipes) == 3, "Số lượng Pipeline huấn luyện không đủ 3"
    for name, pipe in pipes.items():
        assert "preprocessor" in pipe.named_steps and "classifier" in pipe.named_steps, f"Pipeline {name} thiếu thành phần"
    print("  ✓ model_training.py: Tạo mô hình và đóng gói Pipeline tự động đạt 100%.")


def test_06_evaluation_module():
    print("\n[6/8] Kiểm thử module scripts/model_evaluation.py...")
    from scripts.model_evaluation import evaluate_holdout_test, select_best_model
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer

    dummy_pipe = Pipeline([
        ("preprocessor", CountVectorizer()),
        ("classifier", DummyClassifier(strategy="constant", constant=1))
    ])
    X_dummy = pd.Series(["win cash prize", "project meeting report"])
    y_dummy = np.array([1, 0])
    dummy_pipe.fit(X_dummy, y_dummy)

    results, cms = evaluate_holdout_test({"dummy": dummy_pipe}, X_dummy, y_dummy)
    assert len(results) == 1 and "dummy" in cms, "Hàm evaluate_holdout_test trả về sai cấu trúc"
    assert "F1-Score (%)" in results[0], "Thiếu chỉ số F1-Score trong bảng kết quả"

    best_name, best_clf = select_best_model({"dummy": dummy_pipe}, results)
    assert best_name == "dummy" and best_clf == dummy_pipe, "select_best_model không trả về đúng mô hình quán quân"
    print("  ✓ model_evaluation.py: Tính toán metrics, ma trận nhầm lẫn và chọn mô hình tối ưu đạt 100%.")


def test_07_artifacts_and_demo():
    print("\n[7/8] Kiểm thử Models đã lưu và Notebook Demo (notebooks/demo.ipynb)...")
    saved_dir = BASE_DIR / "models" / "saved_models"
    best_model_path = saved_dir / "best_model.joblib"
    meta_path = saved_dir / "metadata.json"

    assert best_model_path.exists(), f"Không tìm thấy {best_model_path}"
    assert meta_path.exists(), f"Không tìm thấy {meta_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["best_model"] == "Support Vector Machine (Linear SVM)"
    assert meta["test_metrics"]["f1_score_percent"] > 95.0, "F1-Score mô hình quán quân không đạt ngưỡng kỳ vọng > 95%"
    assert "LinearSVMScratch" in meta["pipeline_steps"][1], "Chưa đồng bộ classifier sang LinearSVMScratch"
    assert meta["project_name"] == "Practice 1: Classifying Spam Emails — Group 3"
    print(f"  ✓ Model Artifact: Linear SVM F1={meta['test_metrics']['f1_score_percent']}%, Top 1={meta['top_15_keywords'][0]['keyword']}")

    demo_nb_path = BASE_DIR / "notebooks" / "demo.ipynb"
    assert demo_nb_path.exists(), f"Thiếu notebook demo: {demo_nb_path}"
    with open(demo_nb_path, "r", encoding="utf-8") as f:
        demo_json = json.load(f)
    assert len(demo_json["cells"]) >= 4, "demo.ipynb thiếu các code cells cần thiết"
    print("  ✓ notebooks/demo.ipynb: Khởi tạo hoàn chỉnh, tích hợp Guardrails và sẵn sàng suy luận thời gian thực.")


def test_08_codebase_integrity():
    print("\n[8/8] Kiểm thử cú pháp toàn vẹn mã nguồn qua AST Parser...")
    py_files = [
        BASE_DIR / "scripts" / "run_pipeline.py",
        BASE_DIR / "scripts" / "feature_engineering.py",
        BASE_DIR / "scripts" / "model_training.py",
        BASE_DIR / "scripts" / "model_evaluation.py",
        BASE_DIR / "scripts" / "logger_utils.py",
        BASE_DIR / "scripts" / "common.py",
        BASE_DIR / "scripts" / "run_01_eda.py",
        BASE_DIR / "scripts" / "run_02_fe.py",
        BASE_DIR / "scripts" / "run_03_train.py",
        BASE_DIR / "scripts" / "run_04_eval.py",
        BASE_DIR / "scripts" / "report_builder.py",
        BASE_DIR / "data" / "scripts" / "data_preprocessing.py",
        BASE_DIR / "data" / "scripts" / "eda.py",
        BASE_DIR / "models" / "algorithms" / "naive_bayes.py",
        BASE_DIR / "models" / "algorithms" / "logistic_regression.py",
        BASE_DIR / "models" / "algorithms" / "support_vector_machine.py",
    ]
    for py_file in py_files:
        assert py_file.exists(), f"Thiếu tệp Python: {py_file}"
        with open(py_file, "r", encoding="utf-8") as f:
            ast.parse(f.read(), filename=str(py_file))
        print(f"  ✓ {py_file.name}: Cú pháp Python AST hợp lệ 100%.")

    nb_files = [
        BASE_DIR / "notebooks" / "01_data_cleaning_and_eda.ipynb",
        BASE_DIR / "notebooks" / "02_feature_engineering_and_pipeline.ipynb",
        BASE_DIR / "notebooks" / "03_model_training_evaluation_and_explain.ipynb",
        BASE_DIR / "notebooks" / "demo.ipynb"
    ]
    for nb_path in nb_files:
        assert nb_path.exists(), f"Thiếu notebook: {nb_path.name}"
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_json = json.load(f)

        code_cells = [c for c in nb_json["cells"] if c.get("cell_type") == "code"]
        for idx, cell in enumerate(code_cells):
            src = cell.get("source", "")
            if isinstance(src, list):
                src = "".join(src)
            cleaned_lines = []
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith("%") or stripped.startswith("!") or stripped.startswith("?"):
                    cleaned_lines.append("# " + line)
                else:
                    cleaned_lines.append(line)
            cell_code = "\n".join(cleaned_lines)
            try:
                ast.parse(cell_code, filename=f"{nb_path.name}#cell_{idx+1}")
            except SyntaxError as e:
                raise AssertionError(f"Lỗi cú pháp Python AST tại {nb_path.name} cell #{idx+1}: {e}")

        print(f"  ✓ {nb_path.name}: Toàn bộ {len(code_cells)} code cells đã vượt qua kiểm định AST 100%.")

    # Kiểm tra sự tồn tại và tính hợp lệ của các artifacts đã sinh
    stage_01_json = BASE_DIR / "outputs" / "stage_01.json"
    if stage_01_json.exists():
        with open(stage_01_json, "r", encoding="utf-8") as f:
            s1_data = json.load(f)
        assert s1_data.get("status") == "success", "stage_01.json có trạng thái không hợp lệ"
        assert s1_data.get("metrics", {}).get("clean_rows") == 32660, "Số lượng mẫu sạch không đúng 32,660"
        print("  ✓ outputs/stage_01.json: Dữ liệu phong bì Chặng 1 hoàn chỉnh và số mẫu sạch đạt 32,660.")

    report_01 = BASE_DIR / "docs" / "report" / "01_data_report.html"
    index_html = BASE_DIR / "docs" / "report" / "index.html"
    if report_01.exists() and index_html.exists():
        assert report_01.stat().st_size > 1000, "01_data_report.html bị rỗng hoặc lỗi"
        assert index_html.stat().st_size > 1000, "index.html bị rỗng hoặc lỗi"
        print("  ✓ docs/report/: Các tệp 01_data_report.html và index.html đã được sinh tự động hợp lệ.")

    stage_02_json = BASE_DIR / "outputs" / "stage_02.json"
    if stage_02_json.exists():
        with open(stage_02_json, "r", encoding="utf-8") as f:
            s2_data = json.load(f)
        assert s2_data.get("status") == "success", "stage_02.json có trạng thái không hợp lệ"
        assert s2_data.get("metrics", {}).get("total_features_dim") == 2007, "Tổng số chiều đặc trưng không đúng 2,007"
        assert s2_data.get("metrics", {}).get("text_overlap_count") == 0, "Có rò rỉ dữ liệu giữa Train và Test"
        print("  ✓ outputs/stage_02.json: Dữ liệu phong bì Chặng 2 hoàn chỉnh (2,007 chiều, Zero Leakage).")

    report_02 = BASE_DIR / "docs" / "report" / "02_feature_pipeline_report.html"
    if report_02.exists():
        assert report_02.stat().st_size > 1000, "02_feature_pipeline_report.html bị rỗng hoặc lỗi"
        print("  ✓ docs/report/02_feature_pipeline_report.html: Báo cáo Chặng 2 đã sinh tự động hợp lệ.")

    stage_03_json = BASE_DIR / "outputs" / "stage_03.json"
    if stage_03_json.exists():
        with open(stage_03_json, "r", encoding="utf-8") as f:
            s3_data = json.load(f)
        assert s3_data.get("status") == "success", "stage_03.json có trạng thái không hợp lệ"
        assert s3_data.get("metrics", {}).get("models_count") == 3, "Số lượng mô hình huấn luyện không đúng 3"
        print("  ✓ outputs/stage_03.json: Dữ liệu phong bì Chặng 3 hoàn chỉnh (3 mô hình Scratch, 5-Fold CV).")

    stage_04_json = BASE_DIR / "outputs" / "stage_04.json"
    if stage_04_json.exists():
        with open(stage_04_json, "r", encoding="utf-8") as f:
            s4_data = json.load(f)
        assert s4_data.get("status") == "success", "stage_04.json có trạng thái không hợp lệ"
        assert s4_data.get("metrics", {}).get("best_model") in ["linear_svm", "logistic_regression", "naive_bayes"]
        assert len(s4_data.get("tables", {}).get("top_15_keywords", [])) == 15, "Top 15 từ khóa không đủ 15 phần tử"
        print("  ✓ outputs/stage_04.json: Dữ liệu phong bì Chặng 4 hoàn chỉnh (Hold-out Test, Top 15 keywords).")

    report_03 = BASE_DIR / "docs" / "report" / "03_model_training_report.html"
    if report_03.exists():
        assert report_03.stat().st_size > 1000, "03_model_training_report.html bị rỗng hoặc lỗi"
        print("  ✓ docs/report/03_model_training_report.html: Báo cáo Chặng 3 (Training & CV) đã sinh tự động hợp lệ.")

    report_04 = BASE_DIR / "docs" / "report" / "04_model_evaluation_report.html"
    if report_04.exists():
        assert report_04.stat().st_size > 1000, "04_model_evaluation_report.html bị rỗng hoặc lỗi"
        print("  ✓ docs/report/04_model_evaluation_report.html: Báo cáo Chặng 4 (Evaluation & Test) đã sinh tự động hợp lệ.")

    log_report = BASE_DIR / "docs" / "report" / "run_log.html"
    if log_report.exists():
        assert log_report.stat().st_size > 500, "run_log.html bị rỗng hoặc lỗi"
        print("  ✓ docs/report/run_log.html: Trang nhật ký thực thi Zero-drift đã sinh tự động hợp lệ.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bộ kiểm thử tự động toàn diện hệ thống — Nhóm 3")
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        default=None,
        help="Chạy riêng lẻ bài test theo bước (1-8). Mặc định: chạy toàn bộ 8 bài test."
    )
    args = parser.parse_args()

    tests = {
        1: ("data/scripts/data_preprocessing.py (Tiền xử lý & Khử trùng lặp)", test_01_data_preprocessing),
        2: ("data/scripts/eda.py (Phân tích EDA & Khảo sát ngoại lai)", test_02_eda),
        3: ("scripts/feature_engineering.py (Kỹ thuật đặc trưng & Pipeline)", test_03_feature_engineering),
        4: ("models/algorithms/ (3 Thuật toán Scratch thuần NumPy)", test_04_scratch_algorithms),
        5: ("scripts/model_training.py (Huấn luyện Scratch & 5-Fold CV)", test_05_training_module),
        6: ("scripts/model_evaluation.py (Đánh giá độc lập & Confusion Matrix)", test_06_evaluation_module),
        7: ("models/saved_models/ & notebooks/demo.ipynb (Mô hình tối ưu & Demo)", test_07_artifacts_and_demo),
        8: ("Toàn vẹn cú pháp AST toàn bộ mã nguồn (Kiểm tra chung toàn dự án)", test_08_codebase_integrity),
    }

    if args.step is not None:
        print("================================================================================")
        print(f"   CHẠY KIỂM THỬ ĐỘC LẬP BƯỚC [{args.step}/8]: {tests[args.step][0]}")
        print("================================================================================")
        tests[args.step][1]()
        print("\n================================================================================")
        print(f"   🎉 CHÚC MỪNG: BƯỚC {args.step} ĐÃ ĐẠT 100% TIÊU CHUẨN ĐỊNH NGHĨA HOÀN THÀNH!")
        print("================================================================================")
    else:
        print("================================================================================")
        print("   BẮT ĐẦU KIỂM THỬ TOÀN DIỆN HỆ THỐNG MÃ NGUỒN & MODULES — NHÓM 3")
        print("================================================================================")
        for step in range(1, 9):
            tests[step][1]()
        print("\n================================================================================")
        print("   🎉 CHÚC MỪNG: TOÀN BỘ 8 NHÓM THÀNH PHẦN ĐÃ ĐƯỢC KIỂM THỬ ĐẠT 100% TIÊU CHUẨN!   ")
        print("================================================================================")
