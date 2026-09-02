from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

# 结构化日志字段白名单：与请求日志中间件通过 extra= 传入的键一致。
_EXTRA_KEYS = ("request_id", "method", "path", "status_code", "duration_ms")


class JsonFormatter(logging.Formatter):
    """将每条日志序列化为单行 JSON，便于脚本/集中式日志系统解析。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in _EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def configure_logging(level: int = logging.INFO) -> None:
    """将根 logger 及 uvicorn 相关 logger 统一为 JSON 输出。

    幂等：可被 create_app 或测试重复调用。运行 uvicorn 时建议显式
    ``--no-access-log``（或自定义 --log-config）以避免 uvicorn 自身访问日志
    与结构化日志重复打印。
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers[:] = []
        logger.propagate = True
        logger.setLevel(level)
    # The application middleware emits the canonical correlated access record.
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers[:] = []
    uvicorn_access.propagate = False
    uvicorn_access.disabled = True
