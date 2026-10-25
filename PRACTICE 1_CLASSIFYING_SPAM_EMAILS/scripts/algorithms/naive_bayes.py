"""
Module: Multinomial Naive Bayes Classifier (From Scratch)
Dự án: SpamGuard-ML (Practice 1: Classifying Spam Emails)
Mô tả: Tự cài đặt thuật toán Naive Bayes Đa thức bằng NumPy thuần túy với làm mịn Laplace và tính toán trong không gian Log.
Hỗ trợ cả ma trận dày (Dense) và ma trận thưa (Scipy CSR Matrix từ TF-IDF hoặc CountVectorizer).
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Cấu hình UTF-8 cho Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class MultinomialNaiveBayesScratch:
    """
    Lớp Phân loại Naive Bayes Đa thức (Multinomial Naive Bayes) tự cài đặt.
    
    Tham số:
    ----------
    alpha : float, mặc định=1.0
        Hệ số làm mịn Laplace (Laplace smoothing) để tránh xác suất bằng 0 với từ chưa gặp.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.classes = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        """
        Huấn luyện mô hình Naive Bayes bằng cách đếm tần suất từ khóa theo từng lớp.
        
        Tham số:
        ----------
        X : np.ndarray hoặc scipy.sparse matrix, kích thước (n_samples, n_features)
        y : np.ndarray, kích thước (n_samples,)
        """
        y = np.asarray(y, dtype=np.int64)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.class_log_prior_ = np.zeros(n_classes, dtype=np.float64)
        feature_counts = np.zeros((n_classes, n_features), dtype=np.float64)

        # 1. Tính toán xác suất tiên nghiệm P(c) và đếm tổng số lần xuất hiện của từ trong mỗi lớp
        for idx, c in enumerate(self.classes):
            mask = (y == c)
            X_c = X[mask]
            
            # log P(c)
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / n_samples)
            
            # Tổng số lần xuất hiện của từng từ trong lớp c
            if sparse.issparse(X_c):
                feature_counts[idx, :] = np.asarray(X_c.sum(axis=0)).ravel()
            else:
                feature_counts[idx, :] = np.sum(X_c, axis=0)

        # 2. Tính xác suất có điều kiện P(w | c) có áp dụng làm mịn Laplace
        # Công thức: (count + alpha) / (total_count_class + alpha * n_features)
        total_counts = np.sum(feature_counts, axis=1, keepdims=True)
        smoothed_prob = (feature_counts + self.alpha) / (total_counts + self.alpha * n_features)
        self.feature_log_prob_ = np.log(smoothed_prob)

        return self

    def _joint_log_likelihood(self, X) -> np.ndarray:
        """Tính toán tổng log-likelihood cho từng lớp: log P(c) + sum(x_i * log P(w_i | c))."""
        if sparse.issparse(X):
            return X.dot(self.feature_log_prob_.T) + self.class_log_prior_
        return np.dot(X, self.feature_log_prob_.T) + self.class_log_prior_

    def predict(self, X) -> np.ndarray:
        """Dự đoán nhãn có xác suất hậu nghiệm lớn nhất."""
        jll = self._joint_log_likelihood(X)
        best_indices = np.argmax(jll, axis=1)
        return self.classes[best_indices]

    def predict_proba(self, X) -> np.ndarray:
        """
        Chuẩn hóa Log-Likelihood qua hàm Softmax để xuất xác suất dự đoán cho từng lớp.
        Trả về mảng 2 chiều kích thước (n_samples, n_classes).
        """
        jll = self._joint_log_likelihood(X)
        # Ổn định số học: trừ đi giá trị cực đại để chống tràn số
        jll_shifted = jll - np.max(jll, axis=1, keepdims=True)
        exp_jll = np.exp(jll_shifted)
        return exp_jll / np.sum(exp_jll, axis=1, keepdims=True)


# Bí danh tương thích
MultinomialNaiveBayes = MultinomialNaiveBayesScratch


# ----------------------------------------------------------------------
# Hàm tiện ích: Nạp dữ liệu và Đánh giá
# ----------------------------------------------------------------------
def load_data(path: str, max_rows: int = None):
    """
    Đọc dữ liệu email:
    - Nếu là clean_emails.csv: Đọc trực tiếp cột 'text_clean' đã làm sạch và khử trùng lặp.
    - Nếu là file thô (CEAS_08.csv): Tự động ghép subject + body và loại bỏ giá trị khuyết.
    """
    df = pd.read_csv(path)
    if max_rows and max_rows < len(df):
        df = df.sample(n=max_rows, random_state=42)

    df = df.dropna(subset=['label'])

    if 'text_clean' in df.columns:
        texts = df['text_clean'].fillna('').astype(str).tolist()
    else:
        df['subject'] = df['subject'].fillna('')
        df['body'] = df['body'].fillna('')
        texts = (df['subject'] + ' ' + df['body']).astype(str).tolist()

    labels = df['label'].astype(int).values
    return texts, labels


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tính toán ma trận nhầm lẫn và 4 độ đo đánh giá cốt lõi."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = max(len(y_true), 1)
    acc = (tp + tn) / total
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': [[tn, fp], [fn, tp]]
    }


# ----------------------------------------------------------------------
# Khối thực thi CLI / Demo độc lập
# ----------------------------------------------------------------------
def main():
    default_candidates = [
        Path("data/processed/clean_emails.csv"),
        Path("../data/processed/clean_emails.csv"),
        Path(__file__).resolve().parent.parent.parent / "data/processed/clean_emails.csv"
    ]
    default_path = "data/processed/clean_emails.csv"
    for candidate in default_candidates:
        if candidate.exists():
            default_path = str(candidate)
            break

    parser = argparse.ArgumentParser(description="Chạy thử nghiệm Multinomial Naive Bayes Scratch trên tập dữ liệu email.")
    parser.add_argument("--data", default=default_path, help="Đường dẫn tới file CSV (mặc định: clean_emails.csv)")
    parser.add_argument("--max_rows", type=int, default=None, help="Số dòng lấy mẫu thử nghiệm nhanh (ví dụ: 10000)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Hệ số làm mịn Laplace (alpha)")
    args = parser.parse_args()

    print(f"[*] Đang nạp dữ liệu từ: {args.data}")
    texts, labels = load_data(args.data, args.max_rows)
    print(f"    Tổng số mẫu nạp: {len(texts):,} email | Tỷ lệ Spam: {np.mean(labels):.2%}")

    print("[*] Đang phân chia Train/Test (80/20 phân tầng)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("[*] Đang vector hóa văn bản bằng TF-IDF (max_features=10.000)...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print(f"[*] Đang huấn luyện MultinomialNaiveBayesScratch (alpha={args.alpha})...")
    model = MultinomialNaiveBayesScratch(alpha=args.alpha).fit(X_train, y_train)

    print("[*] Đang đánh giá trên tập kiểm thử Test Set...")
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds)

    print("\n" + "=" * 50)
    print("=== KẾT QUẢ MULTINOMIAL NAIVE BAYES (SCRATCH) ===")
    print("=" * 50)
    print(f"  Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"  Recall   : {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"  F1-Score : {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)")
    print(f"  Confusion Matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")

    # Demo suy luận 1 email mẫu
    sample_email = "URGENT: Your bank account has been suspended. Please click the link to verify your password immediately!"
    sample_vec = vectorizer.transform([sample_email])
    probs = model.predict_proba(sample_vec)[0]
    spam_idx = np.where(model.classes == 1)[0][0] if 1 in model.classes else 1
    prob_spam = probs[spam_idx]
    pred_label = "SPAM" if prob_spam >= 0.5 else "HAM"
    print("\n" + "-" * 50)
    print(f"Email mẫu : \"{sample_email}\"")
    print(f"Dự đoán   : {pred_label} (Xác suất rủi ro Spam: {prob_spam:.2%})")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()
