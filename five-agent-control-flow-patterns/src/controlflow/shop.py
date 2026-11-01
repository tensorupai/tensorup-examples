"""ShopSite: a fake web shop whose every action returns a full page snapshot, like a browser."""

from __future__ import annotations

from dataclasses import dataclass, field

from strands import tool

CATALOG = {
    "hp-100": ("Studio headphones", 45.0),
    "hp-200": ("Noise-cancelling headphones", 129.0),
    "kb-300": ("Mechanical keyboard", 89.0),
}
# Navigation and footer markup that every real page carries and the task never needs.
_CHROME = "".join(f'<a id="nav-{i}" href="/c/{i}">Category {i}</a>' for i in range(60))


@dataclass
class ShopSite:
    page: str = "/"
    cart: list[str] = field(default_factory=list)
    orders: list[list[str]] = field(default_factory=list)

    def snapshot(self) -> str:
        """The whole page, as a DOM snapshot tool would return it."""
        if self.page == "/cart":
            body = f'<ul id="cart">{"".join(f"<li>{s}</li>" for s in self.cart)}</ul>'
            body += '<button id="clear-cart">Clear</button><button id="checkout">Checkout</button>'
        elif self.page == "/checkout":
            body = '<button id="place-order">Place order</button>'
        elif self.page.startswith("/p/"):
            body = f'<h1>{CATALOG[self.page[3:]][0]}</h1><button id="add-to-cart">Add</button>'
        elif self.page.startswith("/search"):
            items = CATALOG.items()
            body = "".join(f'<a id="{s}" href="/p/{s}">{n} ${p}</a>' for s, (n, p) in items)
        else:
            body = '<input id="q"/>'
        return f'<page url="{self.page}">{_CHROME}<main>{body}</main>{_CHROME}</page>'

    def open_url(self, url: str) -> str:
        if not url.startswith("/") or url.startswith("//"):
            raise ValueError(f"only site-relative URLs are allowed, got {url!r}")
        self.page = url
        return self.snapshot()

    def click(self, element_id: str) -> str:
        if element_id == "add-to-cart" and self.page.startswith("/p/"):
            self.cart.append(self.page[3:])
        elif element_id == "clear-cart" and self.page == "/cart":
            self.cart.clear()
        elif element_id == "checkout" and self.page == "/cart":
            self.page = "/checkout"
        elif element_id == "place-order" and self.page == "/checkout":
            # A real shop takes the order again on a second click: nothing here is idempotent.
            self.orders.append(list(self.cart))
            self.page = "/order/confirmed"
        elif element_id in CATALOG:
            self.page = f"/p/{element_id}"
        else:
            raise ValueError(f"no clickable element {element_id!r} on {self.page}")
        return self.snapshot()

    def cart_total(self) -> float:
        return sum(CATALOG[s][1] for s in self.cart)


def make_tools(site: ShopSite) -> list:
    """Browser-style tools bound to one ShopSite."""

    @tool
    def open_url(url: str) -> str:
        """Open a site-relative URL such as /search?q=headphones and return the page snapshot."""
        return site.open_url(url)

    @tool
    def click(element_id: str) -> str:
        """Click the element with this id on the current page and return the new page snapshot."""
        return site.click(element_id)

    return [open_url, click]
