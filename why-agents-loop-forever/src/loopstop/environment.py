"""The system the executor operates on: a service that is down because of a bad config value."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from strands import tool

GOOD_DB_URL = "postgres://db-new.internal:5432/orders"
BAD_DB_URL = "postgres://db-old.internal:5432/orders"


@dataclass
class ServiceEnv:
    """In-memory stand-in for a deployed service.

    Only the fields that define the goal are part of the snapshot. A restart succeeds
    and returns "ok", but it changes none of them while the config is wrong.
    """

    config: dict[str, str] = field(default_factory=lambda: {"db_url": BAD_DB_URL})
    replicas: int = 1
    restarts: int = 0  # operational counter, deliberately outside the snapshot

    @property
    def healthy(self) -> bool:
        return self.config.get("db_url") == GOOD_DB_URL

    def snapshot(self) -> dict[str, Any]:
        return {"config": dict(self.config), "replicas": self.replicas, "healthy": self.healthy}

    def fingerprint(self) -> str:
        """Stable hash of the snapshot; equal fingerprints mean nothing the goal sees changed."""
        raw = json.dumps(self.snapshot(), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


def make_tools(env: ServiceEnv) -> list[Any]:
    """Executor tools bound to one environment."""

    @tool
    def get_status() -> dict[str, Any]:
        """Return the service health, replica count, config and last error."""
        status = env.snapshot()
        if not env.healthy:
            reason = f"db-old.internal is decommissioned, use {GOOD_DB_URL}"
            status["last_error"] = f"connection refused: {reason}"
        return status

    @tool
    def restart_service() -> str:
        """Restart every replica of the service."""
        env.restarts += 1
        return "restart ok"

    @tool
    def set_config(key: str, value: str) -> str:
        """Set one config key on the service."""
        if key not in env.config:
            raise ValueError(f"unknown config key: {key}")
        env.config[key] = value
        return f"{key} updated"

    @tool
    def scale(replicas: int) -> str:
        """Set the replica count (1-10)."""
        if not 1 <= replicas <= 10:
            raise ValueError("replicas must be between 1 and 10")
        env.replicas = replicas
        return f"scaled to {replicas}"

    return [get_status, restart_service, set_config, scale]
