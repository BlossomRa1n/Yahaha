"""Optional Redis caching layer with graceful degradation.

Redis is an optional accelerator, never a correctness dependency. When
``REDIS_URL`` is unset, the ``redis`` package is not installed, or the server is
unreachable, the service falls back to a no-op cache that never changes
behaviour. The cache only memoizes read-heavy payloads (currently the public
item payload) and is invalidated on the writes that affect them, so a cache hit
always returns the same value a miss would have computed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("recsys.cache")


class Cache:
    """Minimal string/JSON cache interface shared by the backends."""

    def get_json(self, key: str) -> Any | None:
        raise NotImplementedError

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        raise NotImplementedError

    def delete(self, *keys: str) -> None:
        raise NotImplementedError

    @property
    def backend(self) -> str:
        raise NotImplementedError


class NoopCache(Cache):
    """Always-miss cache used when Redis is unavailable or disabled."""

    def get_json(self, key: str) -> Any | None:
        return None

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        return None

    def delete(self, *keys: str) -> None:
        return None

    @property
    def backend(self) -> str:
        return "noop"


class RedisCache(Cache):
    """Redis-backed cache; every operation degrades silently on failure."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def connect(cls, url: str, *, timeout_seconds: float = 0.5) -> "RedisCache | None":
        try:
            import redis  # type: ignore
        except ImportError:
            logger.warning("redis package is not installed; degrading to no-op cache")
            return None
        try:
            client = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_timeout=timeout_seconds,
                socket_connect_timeout=timeout_seconds,
            )
            client.ping()
        except Exception as exc:  # noqa: BLE001 - any Redis failure degrades to no-op
            logger.warning("Redis unreachable at %s (%s); degrading to no-op cache", url, exc)
            return None
        return cls(client)

    def get_json(self, key: str) -> Any | None:
        try:
            raw = self._client.get(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis get failed for %s: %s", key, exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        try:
            self._client.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis set failed for %s: %s", key, exc)
        return None

    def delete(self, *keys: str) -> None:
        if not keys:
            return None
        try:
            self._client.delete(*keys)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis delete failed: %s", exc)
        return None

    @property
    def backend(self) -> str:
        return "redis"


def build_cache(redis_url: str | None) -> Cache:
    """Return a Redis cache, or a no-op cache when Redis is absent/disabled."""
    if not redis_url:
        return NoopCache()
    cache = RedisCache.connect(redis_url)
    return cache if cache is not None else NoopCache()
