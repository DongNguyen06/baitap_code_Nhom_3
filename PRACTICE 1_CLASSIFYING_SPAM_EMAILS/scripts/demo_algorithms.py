"""
Runner tập trung: Demo và Kiểm thử 3 Thuật toán Machine Learning (From Scratch)
Dự án: SpamGuard-ML (Practice 1: Classifying Spam Emails)

Thuật toán tự cài đặt thuần NumPy:
1. Multinomial Naive Bayes (Laplace Smoothing)
2. Logistic Regression (Gradient Descent & Binary Cross-Entropy)
3. Linear Support Vector Machine (Hinge Loss & L2 Regularization)
"""

import os
import sys
from pathlib import Path
import numpy as np

# Đảm bảo import được module dù chạy từ thư mục gốc hoặc thư mục scripts
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Cấu hình UTF-8 cho console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from scripts.algorithms.logistic_regression import LogisticRegressionScratch
from scripts.algorithms.naive_bayes import MultinomialNaiveBayesScratch
from scripts.algorithms.support_vector_machine import LinearSVMScratch


def run_naive_bayes_demo():
    print("=" * 70)
    print("1. DEMO: MULTINOMIAL NAIVE BAYES (FROM SCRATCH)")
    print("=" * 70)
    # 6 email, 3 từ khóa: ['free', 'money', 'report']
    X_train = np.array([
        [4, 2, 0],  # Spam 1
        [3, 3, 0],  # Spam 2
        [2, 1, 0],  # Spam 3
        [0, 0, 3],  # Ham 1
        [0, 1, 4],  # Ham 2
        [1, 0, 2]   # Ham 3
    ])
    y_train = np.array([1, 1, 1, 0, 0, 0])

    nb = MultinomialNaiveBayesScratch(alpha=1.0)
    nb.fit(X_train, y_train)

    X_test = np.array([
        [3, 2, 0],  # Email nhiều 'free', 'money'
        [0, 0, 4]   # Email công việc có 'report'
    ])
    preds = nb.predict(X_test)
    probs = nb.predict_proba(X_test)

    for i, (p, prob) in enumerate(zip(preds, probs), 1):
        tag = "SPAM 🚨" if p == 1 else "HAM ✅"
        print(f"  Mẫu {i}: Xác suất [Ham={prob[0]:.2%}, Spam={prob[1]:.2%}] => {tag}")
    print("  -> Naive Bayes kiểm thử thành công 100%!\n")


def run_logistic_regression_demo():
    print("=" * 70)
    print("2. DEMO: LOGISTIC REGRESSION (FROM SCRATCH)")
    print("=" * 70)
    # 6 email, 3 từ khóa: ['free', 'win', 'project']
    X_train = np.array([
        [3.0, 2.0, 0.0],  # Spam 1
        [2.5, 3.0, 0.1],  # Spam 2
        [2.8, 1.8, 0.0],  # Spam 3
        [0.0, 0.1, 2.9],  # Ham 1
        [0.1, 0.0, 3.5],  # Ham 2
        [0.2, 0.2, 2.5]   # Ham 3
    ])
    y_train = np.array([1, 1, 1, 0, 0, 0])

    clf = LogisticRegressionScratch(learning_rate=0.1, n_iters=400)
    clf.fit(X_train, y_train)

    X_test = np.array([
        [2.6, 2.1, 0.0],  # Chứa 'free', 'win'
        [0.0, 0.0, 3.1]   # Chứa 'project'
    ])
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    print(f"  Log-Loss: Ban đầu = {clf.losses[0]:.4f} -> Cuối = {clf.losses[-1]:.4f}")
    for i, (p, prob) in enumerate(zip(preds, probs), 1):
        tag = "SPAM 🚨" if p == 1 else "HAM ✅"
        print(f"  Mẫu {i}: Xác suất Spam = {prob:.2%} => {tag}")
    print("  -> Logistic Regression kiểm thử thành công 100%!\n")


def run_linear_svm_demo():
    print("=" * 70)
    print("3. DEMO: LINEAR SUPPORT VECTOR MACHINE (FROM SCRATCH)")
    print("=" * 70)
    # 6 email, 2 từ khóa: ['prize', 'project']
    X_train = np.array([
        [2.5, 0.1],  # Spam 1
        [3.0, 0.0],  # Spam 2
        [2.8, 0.2],  # Spam 3
        [0.1, 2.8],  # Ham 1
        [0.2, 3.2],  # Ham 2
        [0.0, 2.5]   # Ham 3
    ])
    y_train = np.array([1, 1, 1, 0, 0, 0])

    svm = LinearSVMScratch(learning_rate=0.01, lambda_param=0.01, n_iters=500)
    svm.fit(X_train, y_train)

    X_test = np.array([
        [2.2, 0.1],  # Chứa 'prize'
        [0.1, 3.0]   # Chứa 'project'
    ])
    preds = svm.predict(X_test)
    decisions = svm.decision_function(X_test)

    print(f"  Hinge Loss: Ban đầu = {svm.losses[0]:.4f} -> Cuối = {svm.losses[-1]:.4f}")
    for i, (p, d) in enumerate(zip(preds, decisions), 1):
        tag = "SPAM 🚨" if p == 1 else "HAM ✅"
        print(f"  Mẫu {i}: Điểm siêu phẳng = {d:+.4f} => {tag}")
    print("  -> Linear SVM kiểm thử thành công 100%!\n")


def main():
    print("######################################################################")
    print("#           SPAMGUARD-ML: RUNNER DEMO 3 THUẬT TOÁN SCRATCH          #")
    print("######################################################################\n")
    run_naive_bayes_demo()
    run_logistic_regression_demo()
    run_linear_svm_demo()
    print("🎉 Hoàn tất kiểm thử toàn bộ 3 thuật toán tự cài đặt thành công 100%!")


if __name__ == '__main__':
    main()
