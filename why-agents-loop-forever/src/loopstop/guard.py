"""NoProgressGuard: stops the executor when it repeats a call from an unchanged state."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry

logger = logging.getLogger(__name__)


class NoProgressGuard(HookProvider):
    """Cancels a tool call that was already made from the same state, then ends the run.

    A deterministic tool called with the same input from the same state cannot produce
    a new outcome, so the second call is pure cost. The guard does not explain this to
    the executor: it stops the executor and leaves the verdict to the reviewer.

    The fingerprint is read when the hook fires, so run tools sequentially. A tool that
    polls for change over time (wait, poll) needs an exemption or a time field in the
    fingerprint, or the guard will stop it on the second poll.
    """

    def __init__(self, fingerprint: Callable[[], str]) -> None:
        self._fingerprint = fingerprint
        self._seen: set[tuple[str, str, str]] = set()
        self.tripped: str | None = None

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        registry.add_callback(BeforeToolCallEvent, self._before_tool_call)

    def _before_tool_call(self, event: BeforeToolCallEvent) -> None:
        name = event.tool_use["name"]
        args = json.dumps(event.tool_use["input"], sort_keys=True)
        key = (name, args, self._fingerprint())
        if key not in self._seen:
            self._seen.add(key)
            return
        self.tripped = f"{name}({args}) repeated with no state change"
        logger.warning("no_progress tool=%s args=%s state=%s", name, args, key[2])
        event.cancel_tool = self.tripped
        event.agent.cancel(after_current_tools=True)
