"""Live run: TENSORUP_LIVE=1 uv run python -m supportevals {eval | chat "<message>"}."""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from strands_evals import Case

from .agent import build_agent
from .cases import CASES
from .evals import (
    CUSTOMER_ID,
    LIVE_TRIALS,
    deterministic_evaluators,
    gate,
    judge_evaluators,
    make_task,
    run_suite,
)
from .model import MissingKeyError, build_judge_model, build_model
from .support import SupportDesk

USAGE = 'usage: python -m supportevals {eval | chat "<message>"}'


class StreamCloseNoise(logging.Filter):
    """Drops one asyncio error that every live call logs, and that does not affect the result.

    strands-agents 1.64.0's OpenAIResponsesModel.stream() leaves the response stream with `break`
    on response.completed without closing it. Each sync agent call runs on its own event loop,
    and when that loop shuts down it fails to close the half-read HTTP stream and logs this.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        return not (
            record.getMessage().startswith("an error occurred during closing of asynchronous")
            and isinstance(exc, RuntimeError)
            and "already running" in str(exc)
        )


def with_progress(
    task: Callable[[Case], dict[str, Any]], total: int
) -> Callable[[Case], dict[str, Any]]:
    """Prints a line as each case-trial finishes; a live suite makes ~100 sequential calls."""
    done = 0

    def run(case: Case) -> dict[str, Any]:
        nonlocal done
        start = time.monotonic()
        output = task(case)
        done += 1
        print(f"[{done}/{total}] {case.name:<24} {time.monotonic() - start:.1f}s", flush=True)
        return output

    return run


def run_eval() -> int:
    judge = build_judge_model()  # a missing key fails here, before any paid call
    evaluators = deterministic_evaluators() + judge_evaluators(judge)
    total = len(CASES) * LIVE_TRIALS
    print(f"{len(CASES)} cases x {LIVE_TRIALS} trials; the judges score each run as it finishes")
    task = with_progress(make_task(lambda case: build_model()), total)
    report = run_suite(task, evaluators, trials=LIVE_TRIALS)
    for case, passed, reason in zip(report.cases, report.test_passes, report.reasons, strict=True):
        verdict = "PASS" if passed else "FAIL"
        print(f"{verdict}  {case['evaluator']:<27} {case['name']:<24} {reason}")
    tally: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for case, passed in zip(report.cases, report.test_passes, strict=True):
        tally[(case["evaluator"], case["name"])].append(passed)
    for (evaluator, name), results in tally.items():
        if not all(results):
            print(f"FLAKY {evaluator:<27} {name:<24} {sum(results)}/{len(results)} trials passed")
    failures = gate(report)
    for line in failures:
        print(f"GATE  {line}")
    print("gate: fail" if failures else "gate: pass")
    return 1 if failures else 0


def run_chat(message: str) -> int:
    desk = SupportDesk(customer_id=CUSTOMER_ID)
    print(build_agent(build_model(), desk)(message))
    return 0


def main() -> int:
    load_dotenv(Path(__file__).parents[2] / ".env")
    if os.environ.get("TENSORUP_LIVE") != "1":
        print("Refusing to call a model: set TENSORUP_LIVE=1 to make a paid live run.")
        return 2
    args = sys.argv[1:]
    if not args or args[0] not in ("eval", "chat") or (args[0] == "chat" and len(args) < 2):
        print(USAGE)
        return 2
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("asyncio").addFilter(StreamCloseNoise())
    try:
        return run_eval() if args[0] == "eval" else run_chat(args[1])
    except (MissingKeyError, ValueError) as e:
        print(e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
