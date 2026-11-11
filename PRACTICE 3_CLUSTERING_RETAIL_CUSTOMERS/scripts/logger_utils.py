"""
Module: logger_utils.py
Dự án: Practice 3: Clustering Retail Customers — Group 3
Mục tiêu: Quản lý ghi nhật ký hệ thống kép (Dual Logging: Console + File UTF-8)
hỗ trợ multi-logging, định dạng chuẩn HH:MM:SS | LEVEL | message đồng bộ với Practice 1 & 2.
"""

import sys
import time
import platform
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Generator, Any, Tuple, Union

# Cấu hình UTF-8 cho console Windows
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent


def to_rel_path(path: Union[str, Path, None]) -> str:
    """Chuyển đổi đường dẫn tuyệt đối sang dạng tương đối so với thư mục gốc dự án."""
    if path is None:
        return ""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(BASE_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


class TimedBlock:
    """Đối tượng lưu trữ thông tin thời gian thực thi của một bước."""
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.elapsed: float = 0.0


@contextmanager
def timed(arg1: Any, arg2: Any = None) -> Generator[TimedBlock, None, None]:
    """
    Context manager ghi nhật ký tiến độ:
    - Hỗ trợ cả 2 chữ ký: timed(logger, step) hoặc timed(step, logger).
    - Log '▶ BẮT ĐẦU: ...'
    - Đo thời gian và log '✔ XONG: ... (x.xs)'
    """
    if isinstance(arg1, logging.Logger):
        logger = arg1
        step = str(arg2) if arg2 is not None else "Thực thi tác vụ"
    elif isinstance(arg2, logging.Logger):
        step = str(arg1)
        logger = arg2
    else:
        step = str(arg1)
        logger = None

    block = TimedBlock(step)
    t0 = time.time()
    msg_start = f"▶ BẮT ĐẦU: {step}"
    if logger:
        logger.info(msg_start)
    else:
        print(msg_start)

    try:
        yield block
    except Exception as e:
        block.elapsed = round(time.time() - t0, 2)
        msg_err = f"✖ THẤT BẠI: {step} ({block.elapsed:.2f}s) - {e}"
        if logger:
            logger.error(msg_err)
        else:
            print(msg_err)
        raise
    else:
        block.elapsed = round(time.time() - t0, 2)
        msg_end = f"✔ XONG: {step} ({block.elapsed:.2f}s)"
        if logger:
            logger.info(msg_end)
        else:
            print(msg_end)


def setup_logger(
    log_dir: Optional[Path] = None,
    name: Optional[str] = None,
    prefix: str = "pipeline",
    extra_handler: Optional[logging.Handler] = None,
    timestamp: Optional[str] = None
) -> Tuple[logging.Logger, Path]:
    """
    Cấu hình bộ ghi nhật ký (Logger) chuẩn Practice 1 & 2:
    - Ghi đồng thời ra console (stdout) và tệp logs/{prefix}_{YYYYMMDD_HHMMSS}.log (UTF-8).
    - Định dạng: HH:MM:SS | LEVEL | message.
    - Hỗ trợ extra_handler để ghi đồng thời vào tệp log của pipeline tổng.
    - Trả về tuple: (logger, log_file).
    """
    if log_dir is None:
        log_dir = BASE_DIR / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{prefix}_{ts}.log"
    log_file = log_dir / log_filename

    if name is None:
        name = f"practice3_{prefix}_{ts}"

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
    file_handler = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 3. Gắn thêm extra_handler (nếu có từ pipeline tổng)
    if extra_handler is not None:
        logger.addHandler(extra_handler)

    logger.propagate = False
    logger.log_file = log_file
    logger.file_handler = file_handler

    return logger, log_file


def log_environment(logger: logging.Logger, seed: int = 42) -> None:
    """Ghi lại thông số môi trường phần mềm và hệ thống dạng 1 dòng gọn gàng."""
    try:
        import numpy as np
        import pandas as pd
        import sklearn

        py_ver = sys.version.split()[0]
        os_name = f"{platform.system()} {platform.release()}"
        np_ver = np.__version__
        pd_ver = pd.__version__
        sk_ver = sklearn.__version__

        logger.info(
            f"Môi trường thực thi: Python {py_ver} | OS: {os_name} | "
            f"NumPy {np_ver} | Pandas {pd_ver} | Scikit-learn {sk_ver} | Random Seed = {seed}"
        )
    except Exception as e:
        logger.info(f"Môi trường thực thi: Python {sys.version.split()[0]} | OS: {platform.system()} (Info: {e})")
