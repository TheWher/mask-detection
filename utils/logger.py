"""
统一日志工具
提供带时间戳的格式化日志输出，同时写入文件和控制台
"""

import os
import time


class Logger:
    """简易日志记录器，支持控制台输出 + 文件写入"""

    def __init__(self, log_file=None):
        self.log_file = log_file
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            # 清空旧日志
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('')

    def _write(self, msg):
        """写入文件"""
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')

    def info(self, msg):
        line = f"[INFO {time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        self._write(line)

    def warn(self, msg):
        line = f"[WARN {time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        self._write(line)

    def error(self, msg):
        line = f"[ERROR {time.strftime('%H:%M:%S')}] {msg}"
        print(line)
        self._write(line)

    def section(self, msg):
        """打印分隔标题"""
        line = "=" * 55
        print(f"\n{line}\n  {msg}\n{line}")
        self._write(f"\n{line}\n  {msg}\n{line}")

    def result(self, msg):
        line = f"[RESULT] {msg}"
        print(line)
        self._write(line)


# 全局默认 logger
_default_logger = Logger()


def get_logger(log_file=None):
    """获取 logger 实例"""
    if log_file:
        return Logger(log_file)
    return _default_logger
