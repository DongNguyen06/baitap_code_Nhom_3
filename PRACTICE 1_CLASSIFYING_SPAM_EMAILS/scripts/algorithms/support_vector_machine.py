"""
Module: Linear Support Vector Machine (From Scratch)
Dự án: SpamGuard-ML (Practice 1: Classifying Spam Emails)
Mô tả: Tự cài đặt thuật toán Linear SVM bằng NumPy thuần túy với hàm mất mát Hinge Loss, điều chuẩn L2 và tối ưu hóa bằng Mini-batch Subgradient Descent.
Hỗ trợ cả ma trận dày (Dense) và ma trận thưa (Scipy CSR Matrix từ TF-IDF).
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


class LinearSVMScratch:
    """
    Lớp Máy Vector Hỗ trợ Tuyến tính (Linear SVM) tự cài đặt.
    
    Tham số:
    ----------
    learning_rate : float, mặc định=0.01
        Tốc độ học (alpha / eta) cho Subgradient Descent.
    lambda_param : float, mặc định=0.01
        Hệ số điều chuẩn L2 (tương đương 1/C) kiểm soát độ rộng của lề (margin).
    n_iters : int, mặc định=500
        Số lượng epoch / vòng lặp tối ưu hóa.
    batch_size : int hoặc None, mặc định=256
        Kích thước mini-batch (nếu None hoặc >= n_samples sẽ dùng Full Batch).
    random_state : int, mặc định=42
        Hạt giống ngẫu nhiên phục vụ xáo trộn mini-batch.
    verbose : bool, mặc định=False
        Hiển thị tiến trình suy giảm Hinge Loss qua từng epoch.
    """
    def __init__(self, learning_rate: float = 0.01, lambda_param: float = 0.01,
                 n_iters: int = 500, batch_size: int = 256,
                 random_state: int = 42, verbose: bool = False):
        self.learning_rate = float(learning_rate)
        self.lambda_param = float(lambda_param)
        self.n_iters = int(n_iters)
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)

        self.weights = None
        self.bias = 0.0
        self.losses = []

    def _linear_output(self, X):
        """Tính f(X) = Xw + b hỗ trợ cả ma trận thưa và ma trận dày."""
        if sparse.issparse(X):
            return X.dot(self.weights) + self.bias
        return np.dot(X, self.weights) + self.bias

    def decision_function(self, X) -> np.ndarray:
        """Trả về khoảng cách ký hiệu tới siêu phẳng quyết định: f(x) = w^T x + b."""
        return self._linear_output(X)

    def fit(self, X, y):
        """
        Huấn luyện Linear SVM bằng cách tối thiểu hóa Hinge Loss kết hợp phạt L2.
        
        Tham số:
        ----------
        X : np.ndarray hoặc scipy.sparse matrix, kích thước (n_samples, n_features)
        y : np.ndarray, kích thước (n_samples,) nhận nhãn {0, 1} hoặc {-1, +1}
        """
        y_raw = np.asarray(y, dtype=np.float64)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Ánh xạ nhãn {0, 1} sang {-1, +1} cho công thức Hinge Loss
        y_signed = np.where(y_raw <= 0, -1.0, 1.0)

        # Khởi tạo trọng số
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.losses = []

        batch_sz = self.batch_size if self.batch_size and self.batch_size < n_samples else n_samples

        for epoch in range(self.n_iters):
            if batch_sz < n_samples:
                idx = self.rng.permutation(n_samples)
                for start in range(0, n_samples, batch_sz):
                    batch_idx = idx[start:start + batch_sz]
                    X_b = X[batch_idx]
                    y_b = y_signed[batch_idx]
                    
                    scores = self._linear_output(X_b)
                    # Điều kiện vi phạm lề: y_i * (w*x_i + b) < 1
                    violating = (y_b * scores < 1.0)
                    
                    if np.any(violating):
                        X_v = X_b[violating]
                        y_v = y_b[violating]
                        
                        if sparse.issparse(X_v):
                            grad_w = self.lambda_param * self.weights - (X_v.T.dot(y_v) / len(batch_idx))
                        else:
                            grad_w = self.lambda_param * self.weights - (np.dot(X_v.T, y_v) / len(batch_idx))
                        grad_b = -float(np.sum(y_v)) / len(batch_idx)
                    else:
                        grad_w = self.lambda_param * self.weights
                        grad_b = 0.0

                    self.weights -= self.learning_rate * grad_w
                    self.bias -= self.learning_rate * grad_b
            else:
                scores = self._linear_output(X)
                violating = (y_signed * scores < 1.0)
                
                if np.any(violating):
                    X_v = X[violating]
                    y_v = y_signed[violating]
                    
                    if sparse.issparse(X_v):
                        grad_w = self.lambda_param * self.weights - (X_v.T.dot(y_v) / n_samples)
                    else:
                        grad_w = self.lambda_param * self.weights - (np.dot(X_v.T, y_v) / n_samples)
                    grad_b = -float(np.sum(y_v)) / n_samples
                else:
                    grad_w = self.lambda_param * self.weights
                    grad_b = 0.0

                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            # Tính Hinge Loss tổng thể sau mỗi epoch
            scores_all = self._linear_output(X)
            hinge_loss = float(np.mean(np.maximum(0.0, 1.0 - y_signed * scores_all)) + 0.5 * self.lambda_param * np.sum(self.weights ** 2))
            self.losses.append(hinge_loss)

            if self.verbose and (epoch % max(1, self.n_iters // 10) == 0 or epoch == self.n_iters - 1):
                print(f"Epoch {epoch + 1:>3}/{self.n_iters} - Hinge Loss: {hinge_loss:.4f}")

        return self

    def predict(self, X) -> np.ndarray:
        """Dự đoán nhãn nhị phân: 1 (Spam) nếu f(x) >= 0, ngược lại 0 (Ham)."""
        dec = self.decision_function(X)
        return np.where(dec >= 0.0, 1, 0)


# Bí danh tương thích
LinearSVM = LinearSVMScratch


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

    parser = argparse.ArgumentParser(description="Chạy thử nghiệm Linear SVM Scratch trên tập dữ liệu email.")
    parser.add_argument("--data", default=default_path, help="Đường dẫn tới file CSV (mặc định: clean_emails.csv)")
    parser.add_argument("--max_rows", type=int, default=None, help="Số dòng lấy mẫu thử nghiệm nhanh (ví dụ: 10000)")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch huấn luyện Subgradient Descent")
    parser.add_argument("--lr", type=float, default=0.05, help="Tốc độ học (learning rate)")
    parser.add_argument("--lam", type=float, default=1e-4, help="Hệ số điều chuẩn L2 (lambda)")
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

    print(f"[*] Đang huấn luyện LinearSVMScratch (Epochs={args.epochs}, lr={args.lr}, lam={args.lam})...")
    model = LinearSVMScratch(
        learning_rate=args.lr, lambda_param=args.lam, n_iters=args.epochs,
        batch_size=256, random_state=42, verbose=True
    ).fit(X_train, y_train)

    print("[*] Đang đánh giá trên tập kiểm thử Test Set...")
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds)

    print("\n" + "=" * 50)
    print("=== KẾT QUẢ LINEAR SVM (SCRATCH) ===")
    print("=" * 50)
    print(f"  Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"  Recall   : {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"  F1-Score : {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)")
    print(f"  Confusion Matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")

    # Demo suy luận 1 email mẫu
    sample_email = "URGENT: Your bank account has been suspended. Please click the link to verify your password immediately!"
    sample_vec = vectorizer.transform([sample_email])
    score = float(model.decision_function(sample_vec)[0])
    pred_label = "SPAM" if score >= 0.0 else "HAM"
    print("\n" + "-" * 50)
    print(f"Email mẫu : \"{sample_email}\"")
    print(f"Dự đoán   : {pred_label} (Điểm siêu phẳng quyết định: {score:+.4f})")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()
