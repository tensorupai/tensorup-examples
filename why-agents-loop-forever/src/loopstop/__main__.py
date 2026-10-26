"""Live run: TENSORUP_LIVE=1 uv run python -m loopstop "<goal>"."""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

from .environment import ServiceEnv
from .model import MissingKeyError, build_model
from .orchestrator import Orchestrator

DEFAULT_GOAL = "Get the orders service healthy and running 3 replicas."


def main() -> int:
    load_dotenv()
    if os.environ.get("TENSORUP_LIVE") != "1":
        print("Refusing to call a model: set TENSORUP_LIVE=1 to make a paid live run.")
        return 2
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    goal = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_GOAL
    env = ServiceEnv()
    try:
        orchestrator = Orchestrator(env, build_model(), build_model)
    except MissingKeyError as e:
        print(e)
        return 2
    result = orchestrator.run(goal)
    for i, r in enumerate(result.rounds, 1):
        print(f"round {i}: executor={r.executor_stop} guard={r.guard_tripped} {r.verdict}")
    print(f"outcome={result.outcome} restarts={env.restarts} final_state={env.snapshot()}")
    return 0 if result.outcome == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
