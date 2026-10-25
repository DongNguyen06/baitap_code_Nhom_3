"""
Kiểm thử toàn diện tất cả các thành phần sau khi đồng bộ:
1. Nạp và kiểm tra các artifacts trong models/
2. Kiểm tra suy luận trên app.py
3. Kiểm tra CLI run_predict.py
4. Kiểm tra mã nguồn notebook 03 (outputs: [], execution_count: None, AST)
"""

import os
import sys
import json
import ast

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

def test_models():
    print("[1/4] Kiểm tra các mô hình trong models/...")
    models_dir = os.path.join(base_dir, "models")
    expected_files = [
        "best_model.joblib",
        "linear_svm.joblib",
        "logistic_regression.joblib",
        "naive_bayes.joblib",
        "metadata.json",
        "preprocessor.joblib"
    ]
    for fname in expected_files:
        fpath = os.path.join(models_dir, fname)
        assert os.path.exists(fpath), f"Thiếu file: {fname}"
        print(f"  ✓ {fname} ({os.path.getsize(fpath):,} bytes)")

    # Nạp và kiểm tra metadata
    with open(os.path.join(models_dir, "metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["best_model"] == "Support Vector Machine (Linear SVM)"
    assert meta["test_metrics"]["f1_score_percent"] > 99.0
    assert len(meta["top_15_keywords"]) == 15
    print(f"  ✓ Metadata: best={meta['best_model']}, F1={meta['test_metrics']['f1_score_percent']}%, top1={meta['top_15_keywords'][0]['keyword']}")


def test_app():
    print("\n[2/4] Kiểm tra app.py...")
    from app import analyze_email_risk_features, load_production_model, BEST_MODEL_PATH
    model = joblib.load(BEST_MODEL_PATH)
    
    # Test case Ham
    risk_ham = analyze_email_risk_features("Quarterly Project Review", "Hello team, please see the slide deck for our meeting.")
    df_ham = risk_ham["df_features"]
    assert len(df_ham.columns) == 7, f"Sai số lượng cột: {df_ham.columns.tolist()}"
    assert "has_re_prefix" in df_ham.columns and "subject_exclamation" in df_ham.columns
    prob_ham = model.predict_proba(df_ham)[0][1]
    pred_ham = model.predict(df_ham)[0]
    assert pred_ham == 0
    print(f"  ✓ Test Ham: pred={pred_ham}, proba={prob_ham*100:.2f}% (An toàn)")

    # Test case Spam
    risk_spam = analyze_email_risk_features("URGENT: YOU WON $50,000 CASH!", "Congratulations winner! Claim your $50,000 cash lottery now at http://win-now.biz!!!")
    df_spam = risk_spam["df_features"]
    prob_spam = model.predict_proba(df_spam)[0][1]
    pred_spam = model.predict(df_spam)[0]
    assert pred_spam == 1
    print(f"  ✓ Test Spam: pred={pred_spam}, proba={prob_spam*100:.2f}% (Bắt trúng)")

    # Test case Symbols & Risk signals (không bị short-circuit nhầm thành Ham)
    risk_sym = analyze_email_risk_features("$$$ WIN PRIZE $$$", "!!! $$$ !!! $$$")
    df_sym = risk_sym["df_features"]
    prob_sym = model.predict_proba(df_sym)[0][1]
    assert prob_sym > 0.90, f"Symbol email bị dự đoán sai: {prob_sym}"
    print(f"  ✓ Test Symbols: proba={prob_sym*100:.2f}% (Bắt trúng dấu hiệu ngoại quan)")


def test_cli():
    print("\n[3/4] Kiểm tra CLI run_predict.py...")
    from scripts.run_predict import extract_features, predict_email

    # Email có định dạng tiêu đề Re:
    email_ham = "Subject: Re: Meeting notes\nThanks for sending over the minutes."
    res_ham = predict_email(email_ham)
    assert res_ham["label"] == "HAM (0)"
    assert res_ham["features"]["has_re_prefix"] == 1
    print(f"  ✓ Test CLI Ham (Re:): {res_ham['label']} ({res_ham['spam_probability']*100:.2f}%)")

    # Email lừa đảo
    email_spam = "Subject: PRIZE ALERT!\nYou won $1,000 gift card! Click here now!"
    res_spam = predict_email(email_spam)
    assert res_spam["label"] == "SPAM (1)"
    assert res_spam["features"]["subject_exclamation"] == 1
    print(f"  ✓ Test CLI Spam (!): {res_spam['label']} ({res_spam['spam_probability']*100:.2f}%)")

    # Edge case 1: Email chỉ toàn ký hiệu rác ($ ! $) không có từ vựng NLP
    email_sym = "Subject: $$$ !!!\n$$$ !!! $$$ !!!"
    res_sym = predict_email(email_sym)
    assert res_sym["label"] == "SPAM (1)", f"Email ký hiệu bị gán sai nhãn: {res_sym}"
    print(f"  ✓ Test CLI Ký hiệu rác: {res_sym['label']} ({res_sym['spam_probability']*100:.2f}%)")

    # Edge case 2: Email 1 dòng không bị nhân đôi nội dung
    df_single = extract_features("Please review the draft proposal before afternoon.")
    words = df_single.iloc[0]["text_clean"].split()
    assert len(words) == len(set(words)), f"Nội dung email 1 dòng bị nhân đôi từ: {words}"
    print(f"  ✓ Test CLI 1 dòng: sạch từ, không nhân đôi chuỗi ('{df_single.iloc[0]['text_clean']}')")

    # Edge case 3: Guardrail bảo vệ chuỗi quá ngắn
    res_short = predict_email("hi")
    assert res_short["status"] == "warning"
    print(f"  ✓ Test CLI Chuỗi ngắn: Kích hoạt guardrail '{res_short['status']}'")


def test_notebook_03():
    print("\n[4/4] Kiểm tra Notebook 03 (outputs sạch và cấu trúc)...")
    nb_path = os.path.join(base_dir, "notebooks", "03_model_training_evaluation_and_explain.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_count = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code_count += 1
            assert cell["outputs"] == [], f"Cell {i} có outputs không rỗng!"
            assert cell["execution_count"] is None, f"Cell {i} có execution_count khác None!"
            ast.parse("".join(cell["source"]))

    # Rà soát tiêu đề Mục 6 và Mục 7
    src_all = " ".join(["".join(c["source"]) for c in nb["cells"]])
    assert "Mục 6: Huấn luyện mô hình & Tối ưu siêu tham số" in src_all
    assert "Mục 7: Đánh giá Mô hình trên Tập Kiểm tra" in src_all
    assert "### 6.1" in src_all
    assert "### 6.2" in src_all
    assert "### 7.1" in src_all
    assert "### 7.2" in src_all
    assert "### 7.3" in src_all
    assert "### 7.4" in src_all
    assert "preprocessor__text__select__k" in src_all
    assert "preprocessor__text__tfidf__max_features" in src_all
    assert "prep.get_feature_names_out()" in src_all
    assert "clf.calibrated_classifiers_" in src_all

    print(f"  ✓ Notebook 03: {code_count} code cells chuẩn sạch outputs, AST cú pháp hợp lệ, đầy đủ tiêu đề Mục 6 & Mục 7.")


if __name__ == "__main__":
    print("================== BẮT ĐẦU KIỂM THỬ TOÀN DIỆN ==================")
    test_models()
    test_app()
    test_cli()
    test_notebook_03()
    print("\n================== TOÀN BỘ 4 BƯỚC KIỂM THỬ ĐẠT 100% ==================")
