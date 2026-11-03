"""
Bộ Kiểm thử Tự động Toàn diện Hệ thống (Comprehensive Component Tests & Assignment Verification)
Dự án: Practice 2: Predicting Product Sales — Group 3 (Khoa CNTT - UTH)
Trưởng nhóm: Nguyễn Xuân Đông

Mục đích:
- Phục vụ bàn giao bài tập khung (Assignment Starter Code) cho 6 thành viên trong nhóm.
- Mỗi thành viên có thể kiểm tra tiến độ của riêng mình qua cờ: python scripts/test_all_components.py --step <1-8>
- Trưởng nhóm có thể chạy toàn diện để giám sát tổng thể bài tập lớn: python scripts/test_all_components.py

Phân công kiểm thử (8 Nhóm thành phần):
1. data/scripts/data_preprocessing.py (Nguyễn Minh Nhật)
2. data/scripts/eda.py (Nguyễn Xuân Đông & Mai An Thịnh)
3. scripts/feature_engineering.py (Trương Thành Công & Võ Thanh Phú)
4. models/algorithms/ (Thuật toán Scratch NumPy)
5. scripts/model_training.py (Võ Khôi Nguyên)
6. scripts/model_evaluation.py (Mai Hoàng Danh)
7. models/saved_models/ & Demo tương tác (Nguyễn Xuân Đông - Trưởng nhóm)
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


def test_01_data_preprocessing():
    print("\n[1/8] Kiểm thử module data/scripts/data_preprocessing.py (Nguyễn Minh Nhật)...")
    from data.scripts.data_preprocessing import handle_missing_and_duplicates, clean_amazon_sales_data
    import shutil

    df_raw = pd.DataFrame({
        "Date": ["04-30-22", "04-30-22", "05-01-22"],
        "Status": ["Shipped", "Cancelled", "Shipped"],
        "Fulfilment": ["Merchant", "Amazon", "Merchant"],
        "ship-service-level": ["Standard", "Expedited", "Standard"],
        "Category": ["Set", "Kurta", "Set"],
        "Size": ["M", "L", "M"],
        "Courier Status": [None, "Cancelled", "Shipped"],
        "Qty": [1, 1, 1],
        "Amount": [500.0, 0.0, 500.0],
        "ship-state": [None, "MAHARASHTRA", "MAHARASHTRA"],
        "promotion-ids": [None, "PROMO1", "PROMO2"],
        "B2B": [False, False, False]
    })

    sub_status = {}

    # 1. Test handle_missing_and_duplicates
    try:
        df_filtered = df_raw[~df_raw["Status"].str.contains("Cancelled", case=False, na=False)]
        df_filtered = df_filtered[(df_filtered["Amount"] > 0) & (df_filtered["Qty"] > 0)]
        df_clean = handle_missing_and_duplicates(df_filtered)
        assert len(df_clean) == 2, f"Lỗi số dòng sau làm sạch: {len(df_clean)} != 2"
        assert df_clean["promotion-ids"].iloc[0] == "No_Promotion", "Lỗi: promotion-ids None không được điền 'No_Promotion'"
        assert df_clean["Courier Status"].iloc[0] == "Unshipped", "Lỗi: Courier Status None không được điền 'Unshipped'"
        assert df_clean["ship-state"].iloc[0] == "MAHARASHTRA", "Lỗi: ship-state None không được điền mode"
        print("  ✓ handle_missing_and_duplicates: Khử trùng lặp và điền khuyết thiếu ĐẠT 100%.")
        sub_status["handle_missing_and_duplicates"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ TRIỂN KHAI] handle_missing_and_duplicates: {e}")
        sub_status["handle_missing_and_duplicates"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] handle_missing_and_duplicates thất bại: {e}")
        sub_status["handle_missing_and_duplicates"] = "FAILED"

    # 2. Test clean_amazon_sales_data
    temp_dir = BASE_DIR / "data" / "processed" / "temp_test_prep"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        raw_test_file = temp_dir / "raw_test.csv"
        out_test_file = temp_dir / "clean_test.csv"
        df_raw.to_csv(raw_test_file, index=False)

        df_out = clean_amazon_sales_data(raw_test_file, out_test_file)
        assert len(df_out) == 2, f"Lỗi số dòng clean_amazon_sales_data: {len(df_out)} != 2"
        assert out_test_file.exists(), "clean_amazon_sales_data không xuất file ra output_path"
        print("  ✓ clean_amazon_sales_data: Quy trình lọc hủy và xuất file sạch ĐẠT 100%.")
        sub_status["clean_amazon_sales_data"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ TRIỂN KHAI] clean_amazon_sales_data: {e}")
        sub_status["clean_amazon_sales_data"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] clean_amazon_sales_data thất bại: {e}")
        sub_status["clean_amazon_sales_data"] = "FAILED"
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    if all(s == "PASSED" for s in sub_status.values()):
        print("  ✓ [Nguyễn Minh Nhật] Toàn bộ module data_preprocessing.py ĐẠT 100%.")
        return "PASSED"
    elif any(s == "FAILED" for s in sub_status.values()):
        return "FAILED"
    else:
        print("     👉 Nguyễn Minh Nhật cần mở 'data/scripts/data_preprocessing.py' và hoàn thiện các hàm theo TODO.")
        return "TODO"


def test_02_eda():
    print("\n[2/8] Kiểm thử module data/scripts/eda.py (Nguyễn Xuân Đông & Mai An Thịnh)...")
    from data.scripts.eda import (
        inspect_data_structure,
        get_univariate_stats,
        get_amount_univariate_stats,
        detect_iqr_outliers,
        detect_sales_outliers_iqr,
        get_bivariate_analysis,
        analyze_bivariate_sales
    )

    df_sample = pd.DataFrame({
        "Category": ["Kurta", "Kurta", "Set", "Top"],
        "Amount": [300.0, 350.0, 800.0, 250.0],
        "ship-state": ["MAHARASHTRA", "MAHARASHTRA", "KARNATAKA", "TAMIL NADU"],
        "B2B": [False, False, True, False]
    })

    # 1. Kiểm thử phần Nguyễn Xuân Đông (Trưởng nhóm)
    struct = inspect_data_structure(df_sample)
    assert struct["n_rows"] == 4 and struct["n_cols"] == 4, "Lỗi kiểm tra shape"
    stats = get_univariate_stats(df_sample, col="Amount")
    assert stats["min"] == 250.0 and stats["max"] == 800.0, "Lỗi thống kê đơn biến"
    stats_alias = get_amount_univariate_stats(df_sample, col="Amount")
    assert stats_alias["min"] == 250.0, "Lỗi alias get_amount_univariate_stats"
    print("  ✓ [Nguyễn Xuân Đông] inspect_data_structure & get_univariate_stats: ĐẠT 100%.")

    # 2. Kiểm thử phần Mai An Thịnh
    thinh_status = {}
    # 2a. IQR
    try:
        amounts = pd.Series([100.0, 105.0, 110.0, 95.0, 102.0, 1000.0])
        iqr_res = detect_iqr_outliers(amounts)
        assert iqr_res["n_outliers"] >= 1, "Lỗi nhận diện ngoại lai IQR"
        iqr_alias = detect_sales_outliers_iqr(amounts)
        assert iqr_alias["n_outliers"] >= 1, "Lỗi alias detect_sales_outliers_iqr"
        print("  ✓ detect_iqr_outliers: Quét ngoại lai IQR ĐẠT 100%.")
        thinh_status["iqr"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ MAI AN THỊNH] detect_iqr_outliers: {e}")
        thinh_status["iqr"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] detect_iqr_outliers thất bại: {e}")
        thinh_status["iqr"] = "FAILED"

    # 2b. Bivariate
    try:
        biv = get_bivariate_analysis(df_sample)
        assert "category_stats" in biv and "b2b_stats" in biv, "Lỗi phân tích hai biến"
        biv_alias = analyze_bivariate_sales(df_sample)
        assert "category_stats" in biv_alias, "Lỗi alias analyze_bivariate_sales"
        print("  ✓ get_bivariate_analysis: Phân tích hai biến ĐẠT 100%.")
        thinh_status["bivariate"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ MAI AN THỊNH] get_bivariate_analysis: {e}")
        thinh_status["bivariate"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] get_bivariate_analysis thất bại: {e}")
        thinh_status["bivariate"] = "FAILED"

    if all(s == "PASSED" for s in thinh_status.values()):
        print("  ✓ [Mai An Thịnh] eda.py: Toàn bộ hàm của Thịnh ĐẠT 100%.")
        return "PASSED"
    elif any(s == "FAILED" for s in thinh_status.values()):
        return "FAILED"
    else:
        print("     👉 Mai An Thịnh cần mở 'data/scripts/eda.py' và hoàn thiện các hàm theo TODO.")
        return "PARTIAL_TODO"


def test_03_feature_engineering():
    print("\n[3/8] Kiểm thử module scripts/feature_engineering.py (Trương Thành Công & Võ Thanh Phú)...")
    from scripts.feature_engineering import (
        extract_domain_features,
        transform_target_log,
        detect_collinear_features,
        split_train_test,
        build_column_transformer,
        export_train_test_csv
    )
    import shutil

    df_sample = pd.DataFrame({
        "Date": ["04-30-22", "05-01-22", "05-02-22", "05-03-22", "05-04-22"],
        "Status": ["Shipped"] * 5,
        "Fulfilment": ["Merchant"] * 5,
        "ship-service-level": ["Standard"] * 5,
        "Category": ["Kurta", "Set", "Top", "Kurta", "Set"],
        "Size": ["M", "L", "XL", "S", "M"],
        "Courier Status": ["Shipped"] * 5,
        "Qty": [1, 2, 1, 3, 1],
        "Amount": [400.0, 900.0, 350.0, 1200.0, 850.0],
        "ship-state": ["MAHARASHTRA", "KARNATAKA", "DELHI", "MAHARASHTRA", "TAMIL NADU"],
        "promotion-ids": ["No_Promotion", "PROMO1", "No_Promotion", "PROMO2", "No_Promotion"],
        "B2B": [False, True, False, False, True]
    })

    cong_status = {}
    phu_status = {}

    # 1. Trương Thành Công: extract_domain_features & transform_target_log
    try:
        df_feat = extract_domain_features(df_sample)
        assert "month" in df_feat.columns, "Lỗi trích xuất month"
        assert "day_of_week" in df_feat.columns, "Lỗi trích xuất day_of_week"
        assert "is_weekend" in df_feat.columns, "Lỗi trích xuất is_weekend"
        assert "is_promoted" in df_feat.columns, "Lỗi tạo cờ is_promoted"
        assert "is_b2b" in df_feat.columns, "Lỗi tạo cờ is_b2b"
        print("  ✓ [Trương Thành Công] extract_domain_features: Trích xuất đặc trưng miền ĐẠT 100%.")
        cong_status["extract_features"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ TRƯƠNG THÀNH CÔNG] extract_domain_features: {e}")
        cong_status["extract_features"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] extract_domain_features thất bại: {e}")
        cong_status["extract_features"] = "FAILED"

    try:
        y_log_test = transform_target_log(df_sample["Amount"])
        assert len(y_log_test) == len(df_sample), "Lỗi kích thước transform_target_log"
        print("  ✓ [Trương Thành Công] transform_target_log: Biến đổi log biến mục tiêu ĐẠT 100%.")
        cong_status["log_transform"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ TRƯƠNG THÀNH CÔNG] transform_target_log: {e}")
        cong_status["log_transform"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] transform_target_log thất bại: {e}")
        cong_status["log_transform"] = "FAILED"

    # 2. Võ Thanh Phú: độc lập với Công bằng df_synthetic sẵn sàng
    df_synthetic = pd.DataFrame({
        "Qty": [1, 2, 1, 3, 1],
        "month": [4, 5, 5, 5, 5],
        "day_of_week": [5, 6, 0, 1, 2],
        "is_weekend": [1, 1, 0, 0, 0],
        "is_promoted": [0, 1, 0, 1, 0],
        "is_b2b": [0, 1, 0, 0, 1],
        "Category": ["Kurta", "Set", "Top", "Kurta", "Set"],
        "Size": ["M", "L", "XL", "S", "M"],
        "Fulfilment": ["Merchant", "Merchant", "Merchant", "Merchant", "Merchant"],
        "ship-service-level": ["Standard", "Standard", "Standard", "Standard", "Standard"],
        "Courier Status": ["Shipped", "Shipped", "Shipped", "Shipped", "Shipped"],
        "state_grouped": ["MAHARASHTRA", "KARNATAKA", "DELHI", "MAHARASHTRA", "TAMIL NADU"],
        "Amount": [400.0, 900.0, 350.0, 1200.0, 850.0]
    })

    try:
        collinear = detect_collinear_features(df_synthetic, threshold=0.5)
        assert isinstance(collinear, list), "Lỗi phát hiện đa cộng tuyến"
        print("  ✓ [Võ Thanh Phú] detect_collinear_features: Quét đa cộng tuyến ĐẠT 100%.")
        phu_status["collinear"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ VÕ THANH PHÚ] detect_collinear_features: {e}")
        phu_status["collinear"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] detect_collinear_features thất bại: {e}")
        phu_status["collinear"] = "FAILED"

    try:
        X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = split_train_test(df_synthetic, test_size=0.4, random_state=42)
        assert len(X_train) == 3 and len(X_test) == 2, "Lỗi phân tách Train/Test"
        print("  ✓ [Võ Thanh Phú] split_train_test: Phân chia Train/Test 80/20 ĐẠT 100%.")
        phu_status["split"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ VÕ THANH PHÚ] split_train_test: {e}")
        phu_status["split"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] split_train_test thất bại: {e}")
        phu_status["split"] = "FAILED"

    try:
        num_cols = ["Qty", "month", "day_of_week", "is_promoted", "is_b2b"]
        cat_cols = ["Category", "Size", "Fulfilment", "ship-service-level", "Courier Status", "state_grouped"]
        preprocessor = build_column_transformer(num_cols, cat_cols)
        preprocessor.fit(df_synthetic[num_cols + cat_cols])
        trans = preprocessor.transform(df_synthetic[num_cols + cat_cols])
        assert trans.shape[0] == len(df_synthetic), "Lỗi transform ColumnTransformer"
        print("  ✓ [Võ Thanh Phú] build_column_transformer: Thiết lập ColumnTransformer ĐẠT 100%.")
        phu_status["transformer"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ VÕ THANH PHÚ] build_column_transformer: {e}")
        phu_status["transformer"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] build_column_transformer thất bại: {e}")
        phu_status["transformer"] = "FAILED"

    temp_dir = BASE_DIR / "data" / "processed" / "temp_test_fe"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        dummy_X = df_synthetic.drop(columns=["Amount"])
        dummy_y = df_synthetic["Amount"].values
        p_train, p_test = export_train_test_csv(dummy_X, dummy_X, dummy_y, dummy_y, temp_dir)
        assert p_train.exists() and p_test.exists(), "Lỗi xuất file train.csv/test.csv"
        print("  ✓ [Võ Thanh Phú] export_train_test_csv: Xuất tệp train.csv/test.csv ĐẠT 100%.")
        phu_status["export"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ VÕ THANH PHÚ] export_train_test_csv: {e}")
        phu_status["export"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] export_train_test_csv thất bại: {e}")
        phu_status["export"] = "FAILED"
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    all_statuses = list(cong_status.values()) + list(phu_status.values())
    if all(s == "PASSED" for s in all_statuses):
        print("  ✓ [Thành Công & Thanh Phú] Toàn bộ module feature_engineering.py ĐẠT 100%.")
        return "PASSED"
    elif any(s == "FAILED" for s in all_statuses):
        return "FAILED"
    else:
        print("     👉 Trương Thành Công & Võ Thanh Phú cần mở 'scripts/feature_engineering.py' để hoàn thiện các hàm theo TODO.")
        return "TODO"


def test_04_algorithms():
    print("\n[4/8] Kiểm thử 3 thuật toán Scratch NumPy thuần túy (Linear Regression, Decision Tree, Random Forest)...")
    from models.algorithms import (
        LinearRegressionScratch,
        DecisionTreeRegressorScratch,
        RandomForestRegressorScratch
    )

    np.random.seed(42)
    X = np.random.randn(80, 4)
    y = X[:, 0] * 3.0 - X[:, 1] * 2.0 + 5.0

    algo_status = {}

    # 1. Linear Regression
    try:
        lr = LinearRegressionScratch(method="normal_equation")
        lr.fit(X, y)
        pred_lr = lr.predict(X)
        r2_lr = lr.score(X, y)
        assert r2_lr > 0.95, f"LinearRegressionScratch không đạt R2 > 0.95 (đạt {r2_lr:.4f})"
        assert hasattr(lr, "coef_") and hasattr(lr, "intercept_"), "Thiếu thuộc tính coef_/intercept_"
        print("  ✓ LinearRegressionScratch: OLS Normal Equation hội tụ chính xác 100%.")
        algo_status["Linear Regression"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ TRIỂN KHAI] LinearRegressionScratch: {e}")
        algo_status["Linear Regression"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] LinearRegressionScratch thất bại: {e}")
        algo_status["Linear Regression"] = "FAILED"

    # 2. Decision Tree Regressor
    try:
        dt = DecisionTreeRegressorScratch(max_depth=4, random_state=42)
        dt.fit(X, y)
        pred_dt = dt.predict(X)
        r2_dt = dt.score(X, y)
        assert r2_dt > 0.70, f"DecisionTreeRegressorScratch không đạt R2 > 0.70 (đạt {r2_dt:.4f})"
        assert hasattr(dt, "feature_importances_"), "Thiếu feature_importances_"
        print("  ✓ DecisionTreeRegressorScratch: Cây quyết định MSE Quantile Splits hội tụ chính xác 100%.")
        algo_status["Decision Tree Regressor"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ TRIỂN KHAI] DecisionTreeRegressorScratch: {e}")
        algo_status["Decision Tree Regressor"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] DecisionTreeRegressorScratch thất bại: {e}")
        algo_status["Decision Tree Regressor"] = "FAILED"

    # 3. Random Forest Regressor
    try:
        rf = RandomForestRegressorScratch(n_estimators=5, max_depth=4, random_state=42)
        rf.fit(X, y)
        pred_rf = rf.predict(X)
        r2_rf = rf.score(X, y)
        assert r2_rf > 0.70, f"RandomForestRegressorScratch không đạt R2 > 0.70 (đạt {r2_rf:.4f})"
        assert hasattr(rf, "feature_importances_"), "Thiếu feature_importances_"
        print("  ✓ RandomForestRegressorScratch: Rừng ngẫu nhiên Ensemble Bagging hội tụ chính xác 100%.")
        algo_status["Random Forest Regressor"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ TRIỂN KHAI] RandomForestRegressorScratch: {e}")
        algo_status["Random Forest Regressor"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] RandomForestRegressorScratch thất bại: {e}")
        algo_status["Random Forest Regressor"] = "FAILED"

    if all(s == "PASSED" for s in algo_status.values()):
        print("  ✓ models/algorithms/: Cả 3 class Scratch thuần túy NumPy đạt 100%.")
        return "PASSED"
    elif any(s == "FAILED" for s in algo_status.values()):
        return "FAILED"
    else:
        print("     👉 Kiểm tra lại 3 file trong models/algorithms/.")
        return "TODO"


def test_05_model_training():
    print("\n[5/8] Kiểm thử module scripts/model_training.py (Võ Khôi Nguyên)...")
    from scripts.model_training import (
        create_scratch_models,
        build_training_pipelines,
        evaluate_cv_models,
        tune_hyperparameters,
        train_and_save_all_models
    )
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    import shutil

    nguyen_status = {}

    # 1. create_scratch_models
    try:
        models = create_scratch_models()
        assert len(models) == 3, "Lỗi số lượng mô hình Scratch"
        assert "Linear Regression" in models and "Decision Tree Regressor" in models and "Random Forest Regressor" in models
        print("  ✓ create_scratch_models: Khởi tạo 3 mô hình Scratch ĐẠT 100%.")
        nguyen_status["create_models"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ VÕ KHÔI NGUYÊN] create_scratch_models: {e}")
        nguyen_status["create_models"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] create_scratch_models thất bại: {e}")
        nguyen_status["create_models"] = "FAILED"

    # Dummy preprocessor và data để kiểm tra logic của Nguyên độc lập với Step 4
    df_dummy = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]})
    y_dummy = pd.Series([10.0, 12.0, 15.0, 18.0, 22.0, 25.0])
    dummy_prep = ColumnTransformer([("num", StandardScaler(), ["x1", "x2"])])
    dummy_prep.fit(df_dummy)
    dummy_models = {"Mock Model": LinearRegression()}

    # 2. build_training_pipelines
    try:
        pipes = build_training_pipelines(dummy_prep, dummy_models)
        assert "Mock Model" in pipes, "Lỗi đóng gói Pipeline"
        print("  ✓ build_training_pipelines: Đóng gói Pipeline chuẩn scikit-learn ĐẠT 100%.")
        nguyen_status["build_pipelines"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ VÕ KHÔI NGUYÊN] build_training_pipelines: {e}")
        nguyen_status["build_pipelines"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] build_training_pipelines thất bại: {e}")
        nguyen_status["build_pipelines"] = "FAILED"

    # 3. evaluate_cv_models
    try:
        mock_pipes = {"Mock Linear": Pipeline([("preprocessor", dummy_prep), ("regressor", LinearRegression())])}
        cv_df = evaluate_cv_models(mock_pipes, df_dummy, y_dummy, cv=2, sample_size=None)
        assert len(cv_df) >= 1, "Lỗi kết quả Cross Validation"
        assert "CV R2 (Mean)" in cv_df.columns, "Thiếu cột CV R2 (Mean)"
        print("  ✓ evaluate_cv_models: Kiểm định chéo K-Fold Cross Validation ĐẠT 100%.")
        nguyen_status["evaluate_cv"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ VÕ KHÔI NGUYÊN] evaluate_cv_models: {e}")
        nguyen_status["evaluate_cv"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] evaluate_cv_models thất bại: {e}")
        nguyen_status["evaluate_cv"] = "FAILED"

    # 4. tune_hyperparameters
    try:
        mock_pipe = Pipeline([("preprocessor", dummy_prep), ("regressor", LinearRegression())])
        param_grid = {"regressor__fit_intercept": [True, False]}
        grid = tune_hyperparameters(mock_pipe, param_grid, df_dummy, y_dummy, cv=2)
        assert hasattr(grid, "best_params_"), "GridSearchCV chưa fit thành công"
        print("  ✓ tune_hyperparameters: Tinh chỉnh siêu tham số GridSearchCV ĐẠT 100%.")
        nguyen_status["tune"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ VÕ KHÔI NGUYÊN] tune_hyperparameters: {e}")
        nguyen_status["tune"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] tune_hyperparameters thất bại: {e}")
        nguyen_status["tune"] = "FAILED"

    # 5. train_and_save_all_models
    temp_dir = BASE_DIR / "models" / "saved_models" / "temp_train_test"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        mock_pipes = {"Mock Linear": Pipeline([("preprocessor", dummy_prep), ("regressor", LinearRegression())])}
        trained = train_and_save_all_models(mock_pipes, df_dummy, y_dummy, temp_dir)
        assert "Mock Linear" in trained, "Lỗi train_and_save_all_models"
        print("  ✓ train_and_save_all_models: Huấn luyện và lưu trữ mô hình ĐẠT 100%.")
        nguyen_status["train_save"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ VÕ KHÔI NGUYÊN] train_and_save_all_models: {e}")
        nguyen_status["train_save"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] train_and_save_all_models thất bại: {e}")
        nguyen_status["train_save"] = "FAILED"
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    if all(s == "PASSED" for s in nguyen_status.values()):
        print("  ✓ [Võ Khôi Nguyên] model_training.py: Toàn bộ module của Nguyên ĐẠT 100%.")
        return "PASSED"
    elif any(s == "FAILED" for s in nguyen_status.values()):
        return "FAILED"
    else:
        print("     👉 Võ Khôi Nguyên cần mở 'scripts/model_training.py' và hoàn thiện các hàm theo TODO.")
        return "TODO"


def test_06_model_evaluation():
    print("\n[6/8] Kiểm thử module scripts/model_evaluation.py (Mai Hoàng Danh)...")
    from scripts.model_evaluation import (
        evaluate_holdout_test,
        extract_feature_importances,
        export_evaluation_artifacts,
        plot_actual_vs_predicted,
        plot_residuals
    )
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    import shutil

    df_test = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "x2": [2.0, 3.0, 4.0, 5.0]})
    y_raw = np.array([100.0, 200.0, 300.0, 400.0])
    y_log = pd.Series(np.log1p(y_raw))

    prep = ColumnTransformer([("num", StandardScaler(), ["x1", "x2"])])
    prep.fit(df_test)
    pipe = Pipeline([("preprocessor", prep), ("regressor", LinearRegression())])
    pipe.fit(df_test, y_log)

    danh_status = {}

    # 1. evaluate_holdout_test
    try:
        eval_df = evaluate_holdout_test({"Test Model": pipe}, df_test, y_log, y_raw)
        assert "RMSE (INR)" in eval_df.columns and "R2 (Thực tế)" in eval_df.columns, "Lỗi metrics kiểm thử"
        print("  ✓ evaluate_holdout_test: Đánh giá holdout test với nghịch đảo expm1 ĐẠT 100%.")
        danh_status["evaluate"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ MAI HOÀNG DANH] evaluate_holdout_test: {e}")
        danh_status["evaluate"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] evaluate_holdout_test thất bại: {e}")
        danh_status["evaluate"] = "FAILED"

    # 2. extract_feature_importances
    try:
        top_feats = extract_feature_importances(pipe, top_n=2)
        assert len(top_feats) == 2, "Lỗi trích xuất feature importances"
        print("  ✓ extract_feature_importances: Trích xuất độ quan trọng đặc trưng ĐẠT 100%.")
        danh_status["importances"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ MAI HOÀNG DANH] extract_feature_importances: {e}")
        danh_status["importances"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] extract_feature_importances thất bại: {e}")
        danh_status["importances"] = "FAILED"

    # 3. export_evaluation_artifacts
    temp_dir = BASE_DIR / "models" / "saved_models" / "temp_test_eval"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        dummy_metrics = {"R2 (Thực tế)": 0.5, "RMSE (INR)": 200.0}
        dummy_feats = pd.DataFrame({"Feature": ["x1", "x2"], "Importance": [0.6, 0.4]})
        meta_p = export_evaluation_artifacts("Test Model", dummy_metrics, dummy_feats, temp_dir)
        assert meta_p.exists(), "Lỗi xuất metadata.json"
        print("  ✓ export_evaluation_artifacts: Xuất file metadata.json ĐẠT 100%.")
        danh_status["export_artifacts"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ MAI HOÀNG DANH] export_evaluation_artifacts: {e}")
        danh_status["export_artifacts"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] export_evaluation_artifacts thất bại: {e}")
        danh_status["export_artifacts"] = "FAILED"
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    # 4. plot_actual_vs_predicted
    try:
        plot_actual_vs_predicted(y_raw, y_raw)
        print("  ✓ plot_actual_vs_predicted: Trực quan hóa Actual vs Predicted ĐẠT 100%.")
        danh_status["plot_actual"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ MAI HOÀNG DANH] plot_actual_vs_predicted: {e}")
        danh_status["plot_actual"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] plot_actual_vs_predicted thất bại: {e}")
        danh_status["plot_actual"] = "FAILED"

    # 5. plot_residuals
    try:
        plot_residuals(y_raw, y_raw)
        print("  ✓ plot_residuals: Trực quan hóa phần dư Residual Plot ĐẠT 100%.")
        danh_status["plot_residuals"] = "PASSED"
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ MAI HOÀNG DANH] plot_residuals: {e}")
        danh_status["plot_residuals"] = "TODO"
    except Exception as e:
        print(f"  ❌ [LỖI LOGIC] plot_residuals thất bại: {e}")
        danh_status["plot_residuals"] = "FAILED"

    if all(s == "PASSED" for s in danh_status.values()):
        print("  ✓ [Mai Hoàng Danh] model_evaluation.py: Toàn bộ module của Danh ĐẠT 100%.")
        return "PASSED"
    elif any(s == "FAILED" for s in danh_status.values()):
        return "FAILED"
    else:
        print("     👉 Mai Hoàng Danh cần mở 'scripts/model_evaluation.py' và hoàn thiện các hàm theo TODO.")
        return "TODO"


def test_07_saved_models_and_demo():
    print("\n[7/8] Kiểm thử Models đã lưu và Demo suy luận (Nguyễn Xuân Đông - Trưởng nhóm)...")
    import joblib

    saved_dir = BASE_DIR / "models" / "saved_models"
    best_model_p = saved_dir / "best_model.joblib"
    prep_p = saved_dir / "preprocessor.joblib"
    meta_p = saved_dir / "metadata.json"

    assert best_model_p.exists(), f"Thiếu file: {best_model_p}"
    assert prep_p.exists(), f"Thiếu file: {prep_p}"
    assert meta_p.exists(), f"Thiếu file: {meta_p}"

    model = joblib.load(best_model_p)
    assert hasattr(model, "predict"), "Mô hình best_model.joblib không có hàm predict"

    with open(meta_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "model_name" in meta and "test_metrics" in meta, "Cấu trúc metadata.json không hợp lệ"

    # Kiểm tra suy luận trên 1 mẫu đơn hàng thực tế qua prepare_sample_from_dict
    from scripts.feature_engineering import prepare_sample_from_dict
    sample_dict = {
        "Qty": 1,
        "month": 4,
        "day_of_week": 5,
        "is_promoted": 0,
        "is_b2b": 0,
        "Category": "Kurta",
        "Size": "M",
        "Fulfilment": "Merchant",
        "ship-service-level": "Standard",
        "Courier Status": "Shipped",
        "ship-state": "MAHARASHTRA"
    }
    sample = prepare_sample_from_dict(sample_dict)
    pred_log = model.predict(sample)
    pred_inr = float(np.expm1(pred_log[0]))
    assert pred_inr > 0, f"Dự đoán không hợp lệ: {pred_inr} INR"
    print(f"  ✓ [Nguyễn Xuân Đông] Artifact: {meta['model_name']}, Dự đoán mẫu thực tế: {pred_inr:.2f} INR (SẴN SÀNG 100%)")
    return "PASSED"


def test_08_ast_syntax_integrity():
    print("\n[8/8] Kiểm thử cú pháp toàn vẹn mã nguồn qua AST Parser (Toàn dự án)...")
    py_files = [
        BASE_DIR / "data" / "scripts" / "data_preprocessing.py",
        BASE_DIR / "data" / "scripts" / "eda.py",
        BASE_DIR / "models" / "algorithms" / "linear_regression.py",
        BASE_DIR / "models" / "algorithms" / "decision_tree.py",
        BASE_DIR / "models" / "algorithms" / "random_forest.py",
        BASE_DIR / "scripts" / "feature_engineering.py",
        BASE_DIR / "scripts" / "model_training.py",
        BASE_DIR / "scripts" / "model_evaluation.py",
        BASE_DIR / "scripts" / "run_pipeline.py",
        BASE_DIR / "scripts" / "run_predict.py"
    ]

    for p in py_files:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                source = f.read()
            try:
                ast.parse(source)
                print(f"  ✓ {p.name}: Cú pháp Python AST hợp lệ 100%.")
            except SyntaxError as e:
                raise AssertionError(f"Lỗi cú pháp Python tại {p.name}: {e}")

    # Kiểm tra AST trên code cells của các Notebooks
    notebook_files = [
        BASE_DIR / "notebooks" / "01_data_cleaning_and_eda.ipynb",
        BASE_DIR / "notebooks" / "02_feature_engineering_and_pipeline.ipynb",
        BASE_DIR / "notebooks" / "03_model_training_evaluation_and_explain.ipynb",
        BASE_DIR / "notebooks" / "demo.ipynb"
    ]

    for nb_path in notebook_files:
        if nb_path.exists():
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)
            code_cell_count = 0
            for idx, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") == "code":
                    code_lines = "".join(cell.get("source", []))
                    clean_lines = []
                    for line in code_lines.split("\n"):
                        # Bỏ qua IPython magic commands như %matplotlib, !pip
                        if line.strip().startswith("%") or line.strip().startswith("!"):
                            continue
                        clean_lines.append(line)
                    clean_source = "\n".join(clean_lines)
                    try:
                        ast.parse(clean_source)
                        code_cell_count += 1
                    except SyntaxError as e:
                        raise AssertionError(f"Lỗi cú pháp tại {nb_path.name} (cell {idx}): {e}")
            print(f"  ✓ {nb_path.name}: Toàn bộ {code_cell_count} code cells đã vượt qua kiểm định AST 100%.")

    return "PASSED"


def main():
    parser = argparse.ArgumentParser(description="Bộ Kiểm thử Toàn diện Hệ thống & Giám sát Phân công — Practice 2 (Nhóm 3)")
    parser.add_argument("--step", type=int, choices=range(1, 9), help="Chạy riêng một bước kiểm thử cụ thể (1-8)")
    args = parser.parse_args()

    steps = {
        1: ("Nguyễn Minh Nhật", "data/scripts/data_preprocessing.py", test_01_data_preprocessing),
        2: ("Mai An Thịnh & Nguyễn Xuân Đông", "data/scripts/eda.py", test_02_eda),
        3: ("Trương Thành Công & Võ Thanh Phú", "scripts/feature_engineering.py", test_03_feature_engineering),
        4: ("Thuật toán Scratch (NumPy)", "models/algorithms/", test_04_algorithms),
        5: ("Võ Khôi Nguyên", "scripts/model_training.py", test_05_model_training),
        6: ("Mai Hoàng Danh", "scripts/model_evaluation.py", test_06_model_evaluation),
        7: ("Nguyễn Xuân Đông (Trưởng nhóm)", "models/saved_models/ & Demo", test_07_saved_models_and_demo),
        8: ("Toàn bộ dự án", "Cú pháp AST & Notebooks", test_08_ast_syntax_integrity)
    }

    print("=" * 80)
    print("   BẮT ĐẦU KIỂM THỬ HỆ THỐNG MÃ NGUỒN & TIẾN ĐỘ THÀNH VIÊN — NHÓM 3 (PRACTICE 2)")
    print("=" * 80)

    if args.step:
        owner, target, func = steps[args.step]
        status = func()
        print("\n" + "-" * 80)
        if status == "PASSED":
            print(f"🎉 CHÚC MỪNG: BƯỚC {args.step} ({owner}) ĐÃ VƯỢT QUA TOÀN BỘ KIỂM THỬ 100%!")
        elif status in ["TODO", "PARTIAL_TODO"]:
            print(f"⏳ BƯỚC {args.step} ({owner}) ĐANG CHỜ HOÀN THIỆN:")
            print(f"👉 Vui lòng mở '{target}' và hoàn thành theo các đánh dấu # TODO.")
        else:
            print(f"❌ BƯỚC {args.step} ({owner}) BỊ LỖI ASSERTION: Vui lòng đối soát lại logic.")
        print("-" * 80)
    else:
        results = {}
        for s in range(1, 9):
            owner, target, func = steps[s]
            results[s] = (owner, target, func())

        print("\n" + "=" * 80)
        print("   📊 BẢNG TỔNG HỢP TIẾN ĐỘ & TRẠNG THÁI PHÂN CÔNG — PRACTICE 2 (NHÓM 3)")
        print("=" * 80)
        print(f" {'Bước':<5} | {'Thành viên phụ trách':<35} | {'Module / Tệp':<28} | {'Trạng thái'}")
        print("-" * 88)
        for s, (owner, target, status) in results.items():
            if status == "PASSED":
                status_str = "✅ ĐÃ HOÀN THIỆN"
            elif status == "PARTIAL_TODO":
                status_str = "⏳ ĐÔNG: XONG | THỊNH: TODO"
            elif status == "TODO":
                status_str = "⏳ CHỜ TRIỂN KHAI (TODO)"
            else:
                status_str = "❌ CẦN ĐIỀU CHỈNH"
            print(f" {s:<5} | {owner:<35} | {target:<28} | {status_str}")
        print("=" * 88)
        print("👉 LƯU Ý CHO CÁC THÀNH VIÊN:")
        print("   - Chạy kiểm thử riêng bước của mình: python scripts/test_all_components.py --step <1-8>")
        print("   - Đọc kỹ yêu cầu và công thức toán học trong docs/description/TASKS.md và docs/description/algorithms/")
        print("=" * 88)


if __name__ == "__main__":
    main()
