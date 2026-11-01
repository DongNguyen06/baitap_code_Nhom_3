"""
Runner kiểm thử 3 Thuật toán Machine Learning Scratch trên Dataset Thật
Dự án: SpamGuard-ML (Practice 1: Classifying Spam Emails)
Dữ liệu sử dụng: data/processed/clean_emails.csv
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Cấu hình UTF-8 cho Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Đảm bảo đường dẫn import
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.algorithms.naive_bayes import MultinomialNaiveBayesScratch
from scripts.algorithms.logistic_regression import LogisticRegressionScratch
from scripts.algorithms.support_vector_machine import LinearSVMScratch


def load_real_data(sample_size=3000):
    """Nạp dataset thật clean_emails.csv và trích xuất TF-IDF."""
    csv_path = ROOT / "data" / "processed" / "clean_emails.csv"
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['text_clean', 'label'])

    # Lấy mẫu phân tầng để demo terminal chạy siêu tốc (~3 giây)
    if sample_size and len(df) > sample_size:
        df_spam = df[df['label'] == 1].sample(n=sample_size // 2, random_state=42)
        df_ham = df[df['label'] == 0].sample(n=sample_size // 2, random_state=42)
        df = pd.concat([df_spam, df_ham]).sample(frac=1, random_state=42).reset_index(drop=True)

    texts = df['text_clean'].astype(str).values
    labels = df['label'].astype(int).values

    # Chia tập 80% Train - 20% Test phân tầng
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
    X_train = tfidf.fit_transform(X_train_raw)
    X_test = tfidf.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, tfidf


def evaluate_model(y_true, y_pred):
    """Tính các chỉ số đo lường hiệu năng."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1, [[tn, fp], [fn, tp]]


def main():
    print("=" * 75)
    print("   SPAMGUARD-ML: KIỂM THỬ 3 THUẬT TOÁN SCRATCH TRÊN DATASET THẬT")
    print("=" * 75)

    print("\n⏳ Đang nạp dataset thực tế và trích xuất TF-IDF...")
    X_train, X_test, y_train, y_test, tfidf = load_real_data(sample_size=3000)
    print(f"✅ Dữ liệu sẵn sàng: {X_train.shape[0]} mẫu Train | {X_test.shape[0]} mẫu Test | 2000 đặc trưng từ khóa.")

    # 1. Multinomial Naive Bayes
    print("\n[1/3] Đang huấn luyện Multinomial Naive Bayes Scratch...")
    nb = MultinomialNaiveBayesScratch(alpha=1.0).fit(X_train, y_train)
    acc_nb, prec_nb, rec_nb, f1_nb, cm_nb = evaluate_model(y_test, nb.predict(X_test))

    # 2. Logistic Regression
    print("[2/3] Đang huấn luyện Logistic Regression Scratch (L2 Penalty)...")
    lr = LogisticRegressionScratch(learning_rate=0.2, n_iters=300, l2=1e-4).fit(X_train, y_train)
    acc_lr, prec_lr, rec_lr, f1_lr, cm_lr = evaluate_model(y_test, lr.predict(X_test))

    # 3. Linear SVM
    print("[3/3] Đang huấn luyện Linear Support Vector Machine Scratch (Hinge Loss)...")
    svm = LinearSVMScratch(learning_rate=0.01, lambda_param=0.01, n_iters=300).fit(X_train, y_train)
    acc_svm, prec_svm, rec_svm, f1_svm, cm_svm = evaluate_model(y_test, svm.predict(X_test))

    # Bảng tổng hợp
    print("\n" + "=" * 75)
    print("               BẢNG SO SÁNH HIỆU NĂNG THỰC TẾ TRÊN TẬP TEST")
    print("=" * 75)
    print(f"{'Thuật toán (From Scratch)':<28} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)
    print(f"{'1. Multinomial Naive Bayes':<28} | {acc_nb*100:>8.2f}% | {prec_nb*100:>8.2f}% | {rec_nb*100:>8.2f}% | {f1_nb*100:>8.2f}%")
    print(f"{'2. Logistic Regression':<28} | {acc_lr*100:>8.2f}% | {prec_lr*100:>8.2f}% | {rec_lr*100:>8.2f}% | {f1_lr*100:>8.2f}%")
    print(f"{'3. Linear SVM':<28} | {acc_svm*100:>8.2f}% | {prec_svm*100:>8.2f}% | {rec_svm*100:>8.2f}% | {f1_svm*100:>8.2f}%")
    print("=" * 75)

    # Thử nghiệm trên 2 email thực tế
    sample_emails = [
        "URGENT: Your bank account is locked! Click here to verify and claim your $1,000 prize now!",
        "Hello team, please review the attached machine learning project report for Group 3."
    ]
    sample_vec = tfidf.transform(sample_emails)

    print("\n🧪 THỬ NGHIỆM DỰ ĐOÁN TRÊN 2 EMAIL MẪU THỰC TẾ:")
    for idx, email in enumerate(sample_emails, 1):
        nb_p = "SPAM 🚨" if nb.predict(sample_vec[idx-1])[0] == 1 else "HAM ✅"
        lr_p = "SPAM 🚨" if lr.predict(sample_vec[idx-1])[0] == 1 else "HAM ✅"
        svm_p = "SPAM 🚨" if svm.predict(sample_vec[idx-1])[0] == 1 else "HAM ✅"
        print(f"\n  Email {idx}: \"{email}\"")
        print(f"  -> Kết luận: Naive Bayes: {nb_p} | Logistic: {lr_p} | SVM: {svm_p}")
    print("\n🎉 Toàn bộ 3 thuật toán Scratch đã chạy thực nghiệm hoàn hảo trên dataset thật!")


if __name__ == '__main__':
    main()
