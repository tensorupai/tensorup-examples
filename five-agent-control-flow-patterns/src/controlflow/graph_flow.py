"""Pattern 3, the graph: code owns the edges; agents fill only the steps that need judgement."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from strands import Agent
from strands.agent import AgentResult
from strands.models import Model
from strands.multiagent import GraphBuilder
from strands.multiagent.base import MultiAgentBase, MultiAgentResult, NodeResult, Status
from strands.multiagent.graph import Graph, GraphState
from strands.telemetry.metrics import EventLoopMetrics

from .shop import ShopSite, make_tools


class FunctionNode(MultiAgentBase):
    """A graph node that runs plain Python: no model call, the same output for the same state."""

    def __init__(self, name: str, func: Callable[[], str]) -> None:
        super().__init__()
        self.id, self.func = name, func

    async def invoke_async(
        self, task: Any, invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> MultiAgentResult:
        message = {"role": "assistant", "content": [{"text": self.func()}]}
        result = AgentResult("end_turn", message, EventLoopMetrics(), {})  # type: ignore[arg-type]
        node = NodeResult(result=result, status=Status.COMPLETED)
        return MultiAgentResult(status=Status.COMPLETED, results={self.id: node})


def build_checkout_graph(
    site: ShopSite, model: Callable[[], Model], sku: str, max_price: float, max_nodes: int = 8
) -> Graph:
    def cart_ok(_: GraphState | None = None) -> bool:
        return site.cart == [sku] and site.cart_total() <= max_price

    def verify() -> str:
        return "ok" if cart_ok() else f"mismatch: cart={site.cart}, expected [{sku!r}]"

    placed: list[int] = []

    def place_order() -> str:
        # The irreversible action runs as code, once, and only on the edge that proved the cart.
        if placed:
            raise RuntimeError("place_order already ran in this graph")
        placed.append(1)
        site.open_url("/cart")
        site.click("checkout")
        site.click("place-order")
        return f"order {len(site.orders)} placed"

    def agent(prompt: str) -> Agent:
        tools = make_tools(site)
        return Agent(model=model(), system_prompt=prompt, tools=tools, callback_handler=None)

    b = GraphBuilder()
    b.add_node(agent(f"Find product {sku} with open_url and add it to the cart."), "find_product")
    b.add_node(FunctionNode("verify_cart", verify), "verify_cart")
    b.add_node(agent(f"Open /cart, clear it, then add only {sku}."), "fix_cart")
    b.add_node(FunctionNode("place_order", place_order), "place_order")
    b.add_edge("find_product", "verify_cart")
    b.add_edge("verify_cart", "place_order", condition=cart_ok)
    b.add_edge("verify_cart", "fix_cart", condition=lambda s: not cart_ok(s))
    b.add_edge("fix_cart", "verify_cart")
    b.set_entry_point("find_product")
    # Python graphs have no default bound; a cycle without these can run until the bill stops it.
    b.set_max_node_executions(max_nodes)
    b.set_execution_timeout(300)
    b.set_node_timeout(120)
    b.reset_on_revisit(True)
    return b.build()
