"""
Kịch bản Tự động hóa Toàn bộ Quy trình Học máy (End-to-End Automated Pipeline)
Dự án: Practice 2: Predicting Product Sales — Group 3 (Khoa CNTT - UTH)
Trưởng nhóm điều phối: Nguyễn Xuân Đông (16%)
Thực thi toàn bộ chu trình 4 chặng: Tiền xử lý -> Kỹ thuật đặc trưng -> Huấn luyện 5-Fold CV -> Đánh giá Test & Lưu Artifacts
"""
from pathlib import Path
import sys
import time
import joblib
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

from data.scripts.data_preprocessing import clean_amazon_sales_data
from scripts.feature_engineering import (
    extract_domain_features,
    detect_collinear_features,
    split_train_test,
    build_column_transformer,
    export_train_test_csv
)
from scripts.model_training import (
    create_scratch_models,
    build_training_pipelines,
    evaluate_cv_models,
    train_and_save_all_models
)
from scripts.model_evaluation import (
    evaluate_holdout_test,
    extract_feature_importances,
    export_evaluation_artifacts
)


def run_full_pipeline():
    start_total = time.time()
    print("=" * 80)
    print("   🚀 BẮT ĐẦU CHẠY TOÀN BỘ QUY TRÌNH HỌC MÁY (END-TO-END PIPELINE) — NHÓM 3")
    print("      ĐỀ TÀI: DỰ ĐOÁN DOANH SỐ BÁN HÀNG THƯƠNG MẠI ĐIỆN TỬ (PRACTICE 2)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # CHẶNG 1: NẠP VÀ LÀM SẠCH DỮ LIỆU THÔ (Nguyễn Minh Nhật - 14%)
    # -------------------------------------------------------------------------
    print("\n[Chặng 1/4] Làm sạch dữ liệu và khử trùng lặp (Nguyễn Minh Nhật - 14%)...")
    raw_csv = BASE_DIR / "data" / "raw" / "Amazon Sale Report.csv"
    clean_csv = BASE_DIR / "data" / "processed" / "clean_amazon_sales.csv"

    try:
        t0 = time.time()
        df_clean = clean_amazon_sales_data(raw_csv, clean_csv)
        print(f"  ✓ Đã làm sạch và xuất {len(df_clean):,} đơn hàng sạch sang '{clean_csv.name}' ({time.time() - t0:.2f}s).")
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ TRIỂN KHAI]: {e}")
        print("  👉 Bước 1 chưa được hoàn thiện. Đường ống tạm dừng tại Chặng 1.")
        return

    # -------------------------------------------------------------------------
    # CHẶNG 2: KỸ THUẬT ĐẶC TRƯNG & CHIA TẬP (Trương Thành Công & Võ Thanh Phú - 14%)
    # -------------------------------------------------------------------------
    print("\n[Chặng 2/4] Trích xuất đặc trưng & ColumnTransformer (Thành Công & Thanh Phú - 14%)...")
    try:
        t0 = time.time()
        df_feat = extract_domain_features(df_clean)
        collinear = detect_collinear_features(df_feat, threshold=0.6)
        print(f"  • Quét đa cộng tuyến (|r| > 0.6): {len(collinear)} cặp: {collinear}")

        X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = split_train_test(
            df_feat, test_size=0.2, random_state=42
        )
        print(f"  ✓ Train set: {len(X_train):,} mẫu | Test set: {len(X_test):,} mẫu (Tỷ lệ 80/20).")

        num_cols = ['Qty', 'month', 'day_of_week', 'is_promoted', 'is_b2b']
        cat_cols = ['Category', 'Size', 'Fulfilment', 'ship-service-level', 'Courier Status', 'state_grouped']

        preprocessor = build_column_transformer(num_cols, cat_cols)
        preprocessor.fit(X_train)

        export_train_test_csv(X_train, X_test, y_train_raw, y_test_raw, BASE_DIR / "data" / "processed")
        joblib.dump({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_raw': y_train_raw,
            'y_test_raw': y_test_raw,
            'num_cols': num_cols,
            'cat_cols': cat_cols
        }, BASE_DIR / "data" / "processed" / "train_test_data.joblib")

        saved_models_dir = BASE_DIR / "models" / "saved_models"
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, saved_models_dir / "preprocessor.joblib")
        print(f"  ✓ Đã hoàn tất đóng gói và lưu trữ dữ liệu ({time.time() - t0:.2f}s).")
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ TRIỂN KHAI]: {e}")
        print("  👉 Bước 2 chưa được hoàn thiện. Đường ống tạm dừng tại Chặng 2.")
        return

    # -------------------------------------------------------------------------
    # CHẶNG 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ CHÉO 5-FOLD CV (Võ Khôi Nguyên - 14%)
    # -------------------------------------------------------------------------
    print("\n[Chặng 3/4] Huấn luyện mô hình Scratch & 5-Fold CV (Võ Khôi Nguyên - 14%)...")
    try:
        t0 = time.time()
        pipelines = build_training_pipelines(preprocessor)
        cv_df = evaluate_cv_models(pipelines, X_train, y_train, cv=5, sample_size=10000)
        print(cv_df.to_string(index=False))

        trained_pipelines = train_and_save_all_models(pipelines, X_train, y_train, saved_models_dir)
        print(f"  ✓ Đã hoàn tất huấn luyện và lưu mô hình tối ưu ({time.time() - t0:.2f}s).")
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ TRIỂN KHAI]: {e}")
        print("  👉 Bước 3 chưa được hoàn thiện. Đường ống tạm dừng tại Chặng 3.")
        return

    # -------------------------------------------------------------------------
    # CHẶNG 4: ĐÁNH GIÁ KIỂM THỬ ĐỘC LẬP & LƯU SIÊU DỮ LIỆU (Mai Hoàng Danh - 14%)
    # -------------------------------------------------------------------------
    print("\n[Chặng 4/4] Đánh giá kiểm thử Test & Nghịch đảo expm1 (Mai Hoàng Danh - 14%)...")
    try:
        t0 = time.time()
        eval_df = evaluate_holdout_test(trained_pipelines, X_test, y_test, y_test_raw)
        print(eval_df.to_string(index=False))

        best_pipe = trained_pipelines["Decision Tree Regressor"]
        top_feats = extract_feature_importances(best_pipe, top_n=10)
        meta_p = export_evaluation_artifacts(
            model_name="Decision Tree Regressor (Optimized Scratch)",
            metrics_dict=eval_df.iloc[0].to_dict(),
            top_features=top_feats,
            output_dir=saved_models_dir
        )
        print(f"  ✓ Đã hoàn tất xuất siêu dữ liệu nghiệm thu tại '{meta_p.name}' ({time.time() - t0:.2f}s).")
    except NotImplementedError as e:
        print(f"  ⏳ [CHỜ TRIỂN KHAI]: {e}")
        print("  👉 Bước 4 chưa được hoàn thiện. Đường ống tạm dừng tại Chặng 4.")
        return

    print("\n" + "=" * 80)
    print(f"   🎉 TOÀN BỘ QUY TRÌNH HỌC MÁY ĐÃ HOÀN TẤT THÀNH CÔNG TRONG {time.time() - start_total:.2f} GIÂY!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
