import sys
import time
from pathlib import Path
import pandas as pd

# Cấu hình UTF-8 cho console Windows tránh lỗi ký tự
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data.scripts.data_preprocessing import process_and_save_clean_data
from scripts.feature_engineering import (
    extract_domain_features,
    split_stratified_train_test,
    build_column_transformer,
    save_pipeline_and_splits
)
from scripts.model_training import (
    create_scratch_models,
    build_training_pipelines,
    run_cross_validation,
    tune_grid_search
)
from scripts.model_evaluation import (
    evaluate_holdout_test,
    get_top_spam_keywords,
    select_best_model,
    export_model_artifacts
)


def run_full_pipeline():
    start_total = time.time()
    print("=" * 75)
    print("   🚀 BẮT ĐẦU CHẠY TOÀN BỘ QUY TRÌNH HỌC MÁY (END-TO-END PIPELINE) — NHÓM 3")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # CHẶNG 1: NẠP VÀ LÀM SẠCH DỮ LIỆU THÔ (DATA CLEANING)
    # -------------------------------------------------------------------------
    print("\n[Chặng 1/4] Làm sạch dữ liệu và khử trùng lặp (Zero Duplicate Leakage)...")
    t0 = time.time()
    raw_csv = BASE_DIR / "data" / "raw" / "CEAS_08.csv"
    clean_csv = BASE_DIR / "data" / "processed" / "clean_emails.csv"

    df_raw = pd.read_csv(raw_csv)
    df_clean, n_removed, _ = process_and_save_clean_data(df_raw, str(clean_csv))
    print(f"  ✓ Đã làm sạch và xuất {len(df_clean):,} email sạch sang '{clean_csv.name}' (loại {n_removed:,} bản trùng, {time.time() - t0:.2f}s).")

    # -------------------------------------------------------------------------
    # CHẶNG 2: KỸ THUẬT ĐẶC TRƯNG & CHIA TẬP PHÂN TẦNG (FEATURE ENGINEERING)
    # -------------------------------------------------------------------------
    print("\n[Chặng 2/4] Trích xuất 7 đặc trưng miền & Thiết lập ColumnTransformer...")
    t0 = time.time()
    df_feat = extract_domain_features(df_clean)
    X_train, X_test, y_train, y_test = split_stratified_train_test(df_feat, test_size=0.2, random_state=42)
    preprocessor = build_column_transformer(max_features=5000, k_best=2000)

    train_test_joblib = BASE_DIR / "data" / "processed" / "train_test_data.joblib"
    prep_joblib = BASE_DIR / "models" / "saved_models" / "preprocessor.joblib"
    save_pipeline_and_splits(preprocessor, X_train, X_test, y_train, y_test, str(prep_joblib), str(train_test_joblib))
    print(f"  ✓ Train set: {len(X_train):,} mẫu | Test set: {len(X_test):,} mẫu (Phân tầng 80/20).")
    print(f"  ✓ Đã lưu 'preprocessor.joblib' và 'train_test_data.joblib' ({time.time() - t0:.2f}s).")

    # -------------------------------------------------------------------------
    # CHẶNG 3: HUẤN LUYỆN 3 MÔ HÌNH SCRATCH & 5-FOLD CV
    # -------------------------------------------------------------------------
    print("\n[Chặng 3/4] Huấn luyện 3 thuật toán Scratch & Kiểm định chéo 5-Fold Stratified CV...")
    t0 = time.time()
    models = create_scratch_models(random_state=42)
    pipelines = build_training_pipelines(preprocessor, models)

    print("  • Đang thực hiện 5-Fold Stratified Cross Validation...")
    cv_summary_df = run_cross_validation(pipelines, X_train, y_train, random_state=42)
    print("\n--- BẢNG KẾT QUẢ 5-FOLD STRATIFIED CV TRÊN TẬP TRAIN ---")
    print(cv_summary_df.to_string(index=False))

    print("\n  • Đang tinh chỉnh siêu tham số và đóng gói mô hình tối ưu...")
    saved_models_dir = BASE_DIR / "models" / "saved_models"
    tuned_models = tune_grid_search(pipelines, X_train=X_train, y_train=y_train, models_dir=str(saved_models_dir))
    print(f"  ✓ Đã huấn luyện và tối ưu xong 3 mô hình ({time.time() - t0:.2f}s).")

    # -------------------------------------------------------------------------
    # CHẶNG 4: ĐÁNH GIÁ ĐỘC LẬP TẬP TEST & XUẤT HIỆN VẬT TỐI ƯU
    # -------------------------------------------------------------------------
    print("\n[Chặng 4/4] Đánh giá độc lập trên 6,532 mẫu Test set & Xuất best_model.joblib...")
    t0 = time.time()
    test_results, cms = evaluate_holdout_test(tuned_models, X_test, y_test)
    best_name, best_model = select_best_model(tuned_models, test_results)

    # Trích xuất Top từ khóa nếu có mô hình tuyến tính
    top15_df = get_top_spam_keywords(best_model, top_n=15)

    best_path, meta_path = export_model_artifacts(best_model, test_results, top15_df, models_dir=str(saved_models_dir))
    print(f"  ✓ Đã chọn mô hình Quán quân: '{best_name}'")
    print(f"  ✓ Đã xuất 'best_model.joblib' và 'metadata.json' vào '{saved_models_dir.name}/' ({time.time() - t0:.2f}s).")

    # -------------------------------------------------------------------------
    # TỔNG KẾT BẢNG SO SÁNH HIỆU NĂNG TEST SET
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("   🏆 BẢNG TỔNG KẾT HIỆU NĂNG KIỂM THỬ ĐỘC LẬP (HOLD-OUT TEST SET)")
    print("=" * 75)
    header = f"{'Mô hình':<22} | {'Accuracy (%)':<12} | {'Precision (%)':<13} | {'Recall (%)':<10} | {'F1-Score (%)':<12}"
    print(header)
    print("-" * 75)
    for r in test_results:
        print(f"{r['Mô hình']:<22} | {r['Accuracy (%)']:<12.2f} | {r['Precision (%)']:<13.2f} | {r['Recall (%)']:<10.2f} | {r['F1-Score (%)']:<12.2f}")
    print("=" * 75)

    total_time = time.time() - start_total
    print(f"\n🎉 HOÀN TẤT TOÀN BỘ QUY TRÌNH HỌC MÁY THÀNH CÔNG TRONG {total_time:.2f} GIÂY!")
    print(f"👉 Sẵn sàng thử nghiệm email thời gian thực tại: notebooks/demo.ipynb\n")


if __name__ == "__main__":
    run_full_pipeline()

