"""Logs tournants : un fichier par processus, 10 Mo x 10, plus la console."""
from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_SECRET = re.compile(r"(api-key=)[^&\s'\"]+", re.I)


class _Redact(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _SECRET.sub(r"\1***", str(record.msg))
        if record.args:
            record.args = tuple(_SECRET.sub(r"\1***", a) if isinstance(a, str) else a for a in record.args)
        return True


def setup(name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = RotatingFileHandler(log_dir / f"{name}.log", maxBytes=10_000_000, backupCount=10, encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
        h.addFilter(_Redact())
        root.addHandler(h)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    return logging.getLogger(name)
