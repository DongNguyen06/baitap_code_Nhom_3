"""
Module: Logistic Regression Classifier (From Scratch)
Dự án: SpamGuard-ML (Practice 1: Classifying Spam Emails)
Mô tả: Tự cài đặt thuật toán Hồi quy Logistic bằng NumPy thuần túy với Mini-batch Gradient Descent và điều chuẩn L2.
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


class LogisticRegressionScratch:
    """
    Lớp Hồi quy Logistic (Logistic Regression) tự cài đặt.
    
    Tham số:
    ----------
    learning_rate : float, mặc định=0.1
        Tốc độ học (alpha) cho Gradient Descent.
    n_iters : int, mặc định=500
        Số lượng epoch / vòng lặp huấn luyện.
    batch_size : int hoặc None, mặc định=256
        Kích thước mini-batch (nếu None hoặc >= n_samples sẽ dùng Full Batch GD).
    l2 : float, mặc định=1e-4
        Hệ số điều chuẩn L2 (Ridge penalty) chống quá khớp.
    threshold : float, mặc định=0.5
        Ngưỡng quyết định xác suất để phân loại nhị phân.
    random_state : int, mặc định=42
        Hạt giống ngẫu nhiên phục vụ xáo trộn mini-batch.
    verbose : bool, mặc định=False
        Hiển thị tiến trình suy giảm hàm mất mát qua từng epoch.
    """
    def __init__(self, learning_rate: float = 0.1, n_iters: int = 500,
                 batch_size: int = 256, l2: float = 1e-4, threshold: float = 0.5,
                 random_state: int = 42, verbose: bool = False):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.l2 = l2
        self.threshold = threshold
        self.random_state = random_state
        self.verbose = verbose
        self.rng = np.random.RandomState(random_state)
        
        self.weights = None
        self.bias = 0.0
        self.losses = []

    def _sigmoid(self, z):
        """Hàm kích hoạt Sigmoid đưa điểm tuyến tính z về khoảng (0, 1)."""
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def _linear_predict(self, X):
        """Tính z = Xw + b hỗ trợ cả ma trận thưa và ma trận dày."""
        if sparse.issparse(X):
            return X.dot(self.weights) + self.bias
        return np.dot(X, self.weights) + self.bias

    def cost_function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Hàm mất mát Binary Cross-Entropy (Log-Loss) kèm phạt L2."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        ce = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        l2_reg = 0.5 * self.l2 * np.sum(self.weights ** 2) if self.weights is not None else 0.0
        return float(ce + l2_reg)

    def fit(self, X, y):
        """
        Huấn luyện mô hình bằng Mini-batch Gradient Descent.
        
        Tham số:
        ----------
        X : np.ndarray hoặc scipy.sparse matrix, kích thước (n_samples, n_features)
        y : np.ndarray, kích thước (n_samples,)
        """
        y = np.asarray(y, dtype=np.float64)
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Khởi tạo trọng số bằng 0
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        self.losses = []

        batch_sz = self.batch_size if self.batch_size and self.batch_size < n_samples else n_samples

        for epoch in range(self.n_iters):
            if batch_sz < n_samples:
                # Xáo trộn dữ liệu theo mini-batch
                idx = self.rng.permutation(n_samples)
                for start in range(0, n_samples, batch_sz):
                    batch_idx = idx[start:start + batch_sz]
                    X_b = X[batch_idx]
                    y_b = y[batch_idx]
                    
                    z_b = self._linear_predict(X_b)
                    p_b = self._sigmoid(z_b)
                    err_b = p_b - y_b
                    
                    if sparse.issparse(X_b):
                        grad_w = (X_b.T.dot(err_b) / len(batch_idx)) + (self.l2 * self.weights)
                    else:
                        grad_w = (np.dot(X_b.T, err_b) / len(batch_idx)) + (self.l2 * self.weights)
                        
                    grad_b = float(np.mean(err_b))
                    
                    self.weights -= self.learning_rate * grad_w
                    self.bias -= self.learning_rate * grad_b
            else:
                # Full Batch Gradient Descent
                z = self._linear_predict(X)
                p = self._sigmoid(z)
                err = p - y
                
                if sparse.issparse(X):
                    grad_w = (X.T.dot(err) / n_samples) + (self.l2 * self.weights)
                else:
                    grad_w = (np.dot(X.T, err) / n_samples) + (self.l2 * self.weights)
                    
                grad_b = float(np.mean(err))
                
                self.weights -= self.learning_rate * grad_w
                self.bias -= self.learning_rate * grad_b

            # Tính loss tổng thể sau mỗi epoch
            p_full = self._sigmoid(self._linear_predict(X))
            current_loss = self.cost_function(y, p_full)
            self.losses.append(current_loss)

            if self.verbose and (epoch % max(1, self.n_iters // 10) == 0 or epoch == self.n_iters - 1):
                print(f"Epoch {epoch + 1:>3}/{self.n_iters} - Log-Loss: {current_loss:.4f}")

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Trả về xác suất dự đoán lớp 1 (Spam)."""
        z = self._linear_predict(X)
        return self._sigmoid(z)

    def predict(self, X, threshold: float = None) -> np.ndarray:
        """Dự đoán nhãn nhị phân (0 hoặc 1) dựa trên ngưỡng phân loại."""
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return np.where(probs >= thresh, 1, 0)


# Bí danh tương thích
LogisticRegression = LogisticRegressionScratch


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
    # Tìm đường dẫn mặc định an toàn cho clean_emails.csv
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

    parser = argparse.ArgumentParser(description="Chạy thử nghiệm Logistic Regression Scratch trên tập dữ liệu email.")
    parser.add_argument("--data", default=default_path, help="Đường dẫn tới file CSV (mặc định: clean_emails.csv)")
    parser.add_argument("--max_rows", type=int, default=None, help="Số dòng lấy mẫu thử nghiệm nhanh (ví dụ: 10000)")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch huấn luyện Mini-batch GD")
    parser.add_argument("--lr", type=float, default=0.5, help="Tốc độ học (learning rate)")
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

    print(f"[*] Đang huấn luyện LogisticRegressionScratch (Epochs={args.epochs}, lr={args.lr}, Mini-batch=256)...")
    model = LogisticRegressionScratch(
        learning_rate=args.lr, n_iters=args.epochs, batch_size=256,
        l2=1e-4, random_state=42, verbose=True
    ).fit(X_train, y_train)

    print("[*] Đang đánh giá trên tập kiểm thử Test Set...")
    preds = model.predict(X_test)
    metrics = evaluate(y_test, preds)

    print("\n" + "=" * 50)
    print("=== KẾT QUẢ LOGISTIC REGRESSION (SCRATCH) ===")
    print("=" * 50)
    print(f"  Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"  Recall   : {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"  F1-Score : {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)")
    print(f"  Confusion Matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")

    # Demo suy luận 1 email mẫu
    sample_email = "URGENT: Your bank account has been suspended. Please click the link to verify your password immediately!"
    sample_vec = vectorizer.transform([sample_email])
    prob = model.predict_proba(sample_vec)[0]
    pred_label = "SPAM" if prob >= 0.5 else "HAM"
    print("\n" + "-" * 50)
    print(f"Email mẫu : \"{sample_email}\"")
    print(f"Dự đoán   : {pred_label} (Xác suất rủi ro: {prob:.2%})")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()
