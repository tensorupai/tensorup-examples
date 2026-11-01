"""Live run: TENSORUP_LIVE=1 uv run python -m controlflow <pattern> "<task>"."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .graph_flow import build_checkout_graph
from .model import MissingKeyError, build_model
from .planner_executor import PlannerExecutor
from .router import Router
from .shop import ShopSite
from .single_loop import build_single_loop, run_single_loop
from .supervisor import build_typed_supervisor

PATTERNS = ("loop", "router", "graph", "planner", "supervisor")
DEFAULT_TASK = "Buy the studio headphones (sku hp-100) if they cost at most $50."


def run(pattern: str, task: str, site: ShopSite) -> str:
    if pattern == "loop":
        return str(run_single_loop(build_single_loop(site, build_model()), task))
    if pattern == "router":
        return str(Router(site, build_model).handle(task))
    if pattern == "graph":
        # A fixed checkout: the task text goes to find_product, the cart check uses this sku.
        graph = build_checkout_graph(site, build_model, sku="hp-100", max_price=50.0)
        return str(graph(task).status)
    if pattern == "planner":
        run = PlannerExecutor(site, build_model(), build_model).run(task)
        return f"{run.outcome}: {run.summary} ({len(run.steps)} steps)"
    return str(build_typed_supervisor(site, build_model)(task))


def main() -> int:
    load_dotenv(Path(__file__).parents[2] / ".env")
    if os.environ.get("TENSORUP_LIVE") != "1":
        print("Refusing to call a model: set TENSORUP_LIVE=1 to make a paid live run.")
        return 2
    if len(sys.argv) < 2 or sys.argv[1] not in PATTERNS:
        print(f"usage: python -m controlflow {{{'|'.join(PATTERNS)}}} [task]")
        return 2
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    site = ShopSite()
    try:
        print(run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TASK, site))
    except (MissingKeyError, ValueError) as e:
        print(e)
        return 2
    print(f"page={site.page} cart={site.cart} orders={site.orders}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
