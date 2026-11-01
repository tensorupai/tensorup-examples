"""Pattern 2, the router: one classification picks a specialist; code dispatches, once."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from strands import Agent, tool
from strands.models import Model
from strands.types import Limits

from .shop import ShopSite, make_tools

logger = logging.getLogger(__name__)


class Route(StrEnum):
    SHOPPING = "shopping"
    ORDERS = "orders"
    HUMAN = "human"  # fallback: never chosen by the model, only by validation


ROUTER_PROMPT = (
    "Classify the customer request. Reply with exactly one word: "
    "'shopping' (find or buy products) or 'orders' (questions about past orders)."
)
SPECIALIST_PROMPTS = {
    Route.SHOPPING: "You help customers find and buy products with open_url and click.",
    Route.ORDERS: "You answer questions about the customer's past orders with list_orders.",
}


@dataclass(frozen=True)
class Routed:
    route: Route
    answer: str | None  # None when the request is escalated to a human


def classify(request: str, model: Model) -> Route:
    reply = str(Agent(model=model, system_prompt=ROUTER_PROMPT, callback_handler=None)(request))
    label = reply.strip().strip(".'\"").lower()
    if label not in (Route.SHOPPING, Route.ORDERS):
        logger.warning("router label %r is not a route; escalating", reply[:80])
        return Route.HUMAN
    return Route(label)


def specialist_tools(route: Route, site: ShopSite) -> list:
    if route is Route.SHOPPING:
        return make_tools(site)

    @tool
    def list_orders() -> str:
        """List the customer's orders, newest last."""
        return "\n".join(f"order {i + 1}: {', '.join(o)}" for i, o in enumerate(site.orders))

    return [list_orders]


class Router:
    def __init__(self, site: ShopSite, model: Callable[[], Model]) -> None:
        self.site = site
        self.model = model

    def handle(self, request: str) -> Routed:
        route = classify(request, self.model())
        if route is Route.HUMAN:
            return Routed(route, None)
        specialist = Agent(
            model=self.model(),
            system_prompt=SPECIALIST_PROMPTS[route],
            tools=specialist_tools(route, self.site),
            callback_handler=None,
        )
        result = specialist(request, limits=Limits(turns=10, total_tokens=100_000))
        return Routed(route, str(result).strip())
