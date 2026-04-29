# 功能：配置控制台和文件日志，供训练流程统一记录运行信息。
"""日志配置：控制台 + 文件 handler。"""

import logging
import os
import sys
from datetime import datetime


def setup_logger(log_dir: str, level: int = logging.INFO) -> None:
    """
    为 root logger 配置控制台输出和带时间戳的日志文件。
    可以安全重复调用；handler 会被替换，不会层层叠加。
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"run_{timestamp}.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    fmt_console = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    fmt_file = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
    )

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt_console)
    root.addHandler(ch)

    # 文件
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_file)
    root.addHandler(fh)

    # 降低噪声较多的第三方 logger 级别
    for noisy in ("pybullet", "stable_baselines3.common.on_policy_algorithm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(f"Log file: {log_file}")
