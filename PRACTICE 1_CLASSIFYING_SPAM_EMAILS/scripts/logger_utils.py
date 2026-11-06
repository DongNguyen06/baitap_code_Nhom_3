import sys
import time
import platform
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

# Cấu hình UTF-8 cho console Windows tránh lỗi UnicodeEncodeError
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent


class TimedBlock:
    """Đối tượng lưu trữ thông tin thời gian thực thi của một bước."""
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.elapsed: float = 0.0


def setup_logger(log_dir: Path = None, name: str = None, prefix: str = "pipeline", extra_handler: logging.Handler = None) -> tuple:
    """
    Cấu hình bộ ghi nhật ký (Logger) chuẩn:
    - Ghi đồng thời ra console (stdout) và tệp logs/{prefix}_{YYYYMMDD_HHMMSS}.log (UTF-8).
    - Định dạng: HH:MM:SS | LEVEL | message.
    - Xử lý handlers để gọi nhiều lần không bị nhân đôi thông điệp.
    - Hỗ trợ extra_handler để ghi đồng thời vào tệp log của pipeline tổng.
    """
    if log_dir is None:
        log_dir = BASE_DIR / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{prefix}_{timestamp}.log"

    if name is None:
        name = prefix

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Dọn dẹp handlers cũ nếu có để tránh in lặp lại khi gọi nhiều lần
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # 1. File Handler (UTF-8)
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 3. Gắn thêm extra_handler (nếu có từ pipeline)
    if extra_handler is not None:
        logger.addHandler(extra_handler)

    logger.propagate = False
    logger.log_file = log_file
    logger.file_handler = file_handler

    return logger, log_file


@contextmanager
def timed(logger: logging.Logger, step: str):
    """
    Context manager ghi nhật ký tiến độ:
    - Log '▶ BẮT ĐẦU: ...'
    - Đo thời gian và log '✔ XONG: ... (x.xs)'
    - Khi có lỗi: log exception chi tiết và re-raise
    - Đối tượng trả về có thuộc tính .elapsed (giây)
    """
    block = TimedBlock(step)
    logger.info(f"▶ BẮT ĐẦU: {step}")
    t0 = time.time()
    try:
        yield block
        block.elapsed = round(time.time() - t0, 2)
        logger.info(f"✔ XONG: {step} ({block.elapsed:.2f}s)")
    except Exception as e:
        block.elapsed = round(time.time() - t0, 2)
        logger.exception(f"✖ LỖI tại '{step}' sau {block.elapsed:.2f}s: {e}")
        raise


def log_environment(logger: logging.Logger, seed: int = 42) -> None:
    """Ghi nhận thông số hệ điều hành, phiên bản thư viện cốt lõi và seed ngẫu nhiên."""
    import numpy as np
    import pandas as pd
    import sklearn

    logger.info(
        f"Môi trường thực thi: Python {platform.python_version()} | "
        f"OS: {platform.system()} {platform.release()} | "
        f"NumPy {np.__version__} | Pandas {pd.__version__} | Scikit-learn {sklearn.__version__} | "
        f"Random Seed = {seed}"
    )
