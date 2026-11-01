"""Pattern 5, the supervisor: a coordinating agent calls specialist agents as tools."""

from __future__ import annotations

from collections.abc import Callable

from strands import Agent, tool
from strands.models import Model

from .shop import CATALOG, ShopSite, make_tools

SUPERVISOR_PROMPT = (
    "You coordinate specialists to complete a shopping task. You cannot use the shop yourself: "
    "find products with the catalog specialist, then buy with the cart specialist."
)
CATALOG_PROMPT = "Find products in the shop with open_url. Report sku, name and price for each."
CART_PROMPT = "Add the given sku to the cart with open_url and click, then check out and order."


def specialist(site: ShopSite, model: Model, name: str, prompt: str, description: str) -> Agent:
    return Agent(
        name=name,
        description=description,
        model=model,
        system_prompt=prompt,
        tools=make_tools(site),
        callback_handler=None,
    )


def build_supervisor(site: ShopSite, model: Callable[[], Model]) -> Agent:
    """Workers via as_tool(): each one receives a single free-text `input` the supervisor writes."""
    catalog = specialist(site, model(), "catalog", CATALOG_PROMPT, "Finds products and prices.")
    cart = specialist(site, model(), "cart", CART_PROMPT, "Buys one product by sku.")
    return Agent(
        model=model(),
        system_prompt=SUPERVISOR_PROMPT,
        tools=[catalog.as_tool(name="catalog_specialist"), cart.as_tool(name="cart_specialist")],
        callback_handler=None,
    )


def build_typed_supervisor(site: ShopSite, model: Callable[[], Model]) -> Agent:
    """Workers behind typed tools: constraints are schema fields the supervisor must fill."""

    @tool
    def find_products(query: str, max_price: float) -> str:
        """Find products matching the query that cost at most max_price dollars.

        Args:
            query: What the customer is looking for.
            max_price: The customer's hard price limit in dollars.
        """
        worker = specialist(site, model(), "catalog", CATALOG_PROMPT, "Finds products.")
        return str(worker(f"Find: {query}\nHard limit: price <= ${max_price:.2f}"))

    @tool
    def buy_product(sku: str, max_price: float) -> str:
        """Add one product to the cart by sku and place the order, if it costs at most max_price.

        Args:
            sku: The product sku returned by find_products.
            max_price: The customer's hard price limit in dollars.
        """
        # The limit is enforced here, in code, before the worker can spend money.
        if sku not in CATALOG or CATALOG[sku][1] > max_price:
            raise ValueError(f"{sku!r} is unknown or costs more than ${max_price:.2f}")
        worker = specialist(site, model(), "cart", CART_PROMPT, "Buys one product.")
        return str(worker(f"Buy sku {sku}"))

    return Agent(
        model=model(),
        system_prompt=SUPERVISOR_PROMPT,
        tools=[find_products, buy_product],
        callback_handler=None,
    )
