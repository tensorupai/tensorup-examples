"""Offline tests: every model is a scripted FakeModel; nothing here calls a provider."""

from __future__ import annotations

import json
from collections.abc import Callable

from strands import Agent
from strands.multiagent import GraphBuilder
from strands.multiagent.base import Status
from strands.types.content import Messages

from controlflow.graph_flow import build_checkout_graph
from controlflow.model import FakeModel, call, text
from controlflow.planner_executor import PlannerExecutor
from controlflow.router import Route, Router
from controlflow.shop import ShopSite
from controlflow.single_loop import build_single_loop, run_single_loop
from controlflow.supervisor import build_supervisor, build_typed_supervisor

TASK = "Buy the studio headphones (sku hp-100)."


def snapshots(messages: Messages) -> int:
    return json.dumps(messages).count("<page url=")


def factory(*models: FakeModel) -> Callable[[], FakeModel]:
    queue = list(models)
    return lambda: queue.pop(0)


def purchase_calls() -> list:
    return [
        call("open_url", "t1", url="/search?q=headphones"),
        call("click", "t2", element_id="hp-100"),
        call("click", "t3", element_id="add-to-cart"),
        call("open_url", "t4", url="/cart"),
        call("click", "t5", element_id="checkout"),
        call("click", "t6", element_id="place-order"),
    ]


# --- 1. single loop -------------------------------------------------------------------------------


def test_single_loop_resends_every_snapshot_on_every_call() -> None:
    site = ShopSite()
    model = FakeModel([*purchase_calls(), text("Order placed.")])
    result = run_single_loop(build_single_loop(site, model), TASK)
    assert result.stop_reason == "end_turn"
    assert site.orders == [["hp-100"]]
    assert [snapshots(m) for m in model.received] == [0, 1, 2, 3, 4, 5, 6]


def test_single_loop_overflow_trims_the_task_first() -> None:
    site = ShopSite()
    model = FakeModel([*purchase_calls(), text("Order placed.")], context_chars=20_000)
    run_single_loop(build_single_loop(site, model), TASK)
    assert TASK in json.dumps(model.received[0])
    assert TASK not in json.dumps(model.received[-1])  # the model finishes without its task


def test_pinning_the_task_keeps_it_through_overflow() -> None:
    site = ShopSite()
    model = FakeModel([*purchase_calls(), text("Order placed.")], context_chars=20_000)
    run_single_loop(build_single_loop(site, model, pin_task=True), TASK)
    assert TASK in json.dumps(model.received[-1])


def test_single_loop_turn_limit_stops_the_run() -> None:
    model = FakeModel(purchase_calls())
    result = run_single_loop(build_single_loop(ShopSite(), model), TASK, turns=3)
    assert result.stop_reason == "limit_turns"
    assert len(model.received) == 3


# --- 2. router ------------------------------------------------------------------------------------


def test_router_dispatches_a_valid_label_to_one_specialist() -> None:
    site = ShopSite(orders=[["kb-300"]])
    classifier = FakeModel([text("Orders.")])
    orders = FakeModel([call("list_orders", "o1"), text("You ordered kb-300.")])
    routed = Router(site, factory(classifier, orders)).handle("What did I order last week?")
    assert routed.route is Route.ORDERS
    assert routed.answer == "You ordered kb-300."
    assert "kb-300" in json.dumps(orders.received[-1])


def test_router_escalates_a_label_it_does_not_know() -> None:
    specialist = FakeModel([])
    routed = Router(ShopSite(), factory(FakeModel([text("billing")]), specialist)).handle("Refund?")
    assert (routed.route, routed.answer) == (Route.HUMAN, None)
    assert specialist.received == []


def test_substring_edge_conditions_fire_both_branches() -> None:
    """A demonstration of the docs' pattern, not of src/: routing on words in free text
    as a Strands graph. One reply matches both conditions, and both branches run."""
    classifier = Agent(
        model=FakeModel([text("No refund here; the customer asks where an order is.")]),
        callback_handler=None,
    )
    refund = Agent(model=FakeModel([text("Refund started.")]), callback_handler=None)
    orders = Agent(model=FakeModel([text("Order found.")]), callback_handler=None)

    def said(word: str) -> Callable:
        return lambda state: word in str(state.results["classifier"].result).lower()

    b = GraphBuilder()
    b.add_node(classifier, "classifier")
    b.add_node(refund, "refund")
    b.add_node(orders, "orders")
    b.add_edge("classifier", "refund", condition=said("refund"))
    b.add_edge("classifier", "orders", condition=said("order"))
    result = b.build()("I want to know where my order is.")
    assert {n.node_id for n in result.execution_order} == {"classifier", "refund", "orders"}


# --- 3. graph -------------------------------------------------------------------------------------


def find(sku: str) -> FakeModel:
    return FakeModel(
        [
            call("open_url", "f1", url=f"/p/{sku}"),
            call("click", "f2", element_id="add-to-cart"),
            text(f"Added {sku}."),
        ]
    )


def test_graph_places_the_order_only_after_the_deterministic_check() -> None:
    site = ShopSite()
    graph = build_checkout_graph(site, factory(find("hp-100"), FakeModel([])), "hp-100", 50.0)
    result = graph(TASK)
    assert result.status == Status.COMPLETED
    assert [n.node_id for n in result.execution_order] == [
        "find_product",
        "verify_cart",
        "place_order",
    ]
    assert site.orders == [["hp-100"]]


def test_graph_routes_a_wrong_cart_through_the_repair_cycle() -> None:
    site = ShopSite()
    fix = FakeModel(
        [
            call("open_url", "x1", url="/cart"),
            call("click", "x2", element_id="clear-cart"),
            call("open_url", "x3", url="/p/hp-100"),
            call("click", "x4", element_id="add-to-cart"),
            text("Cart fixed."),
        ]
    )
    result = build_checkout_graph(site, factory(find("hp-200"), fix), "hp-100", 50.0)(TASK)
    order = [n.node_id for n in result.execution_order]
    assert order == ["find_product", "verify_cart", "fix_cart", "verify_cart", "place_order"]
    assert site.orders == [["hp-100"]]


def test_graph_cycle_stops_at_max_node_executions_without_ordering() -> None:
    site = ShopSite()
    idle_fix = FakeModel([text("Done.")] * 10)  # claims success, changes nothing
    graph = build_checkout_graph(site, factory(find("hp-200"), idle_fix), "hp-100", 50.0, 6)
    result = graph(TASK)
    assert result.status == Status.FAILED
    assert len(result.execution_order) == 6
    assert site.orders == []


# --- 4. planner/executor --------------------------------------------------------------------------


def planner_run(planner_replies: list, executors: list[FakeModel], max_steps: int = 8):
    site = ShopSite()
    planner = FakeModel(planner_replies)
    run = PlannerExecutor(site, planner, factory(*executors), max_steps=max_steps).run(TASK)
    return site, planner, run


def test_planner_never_sees_a_snapshot_and_each_executor_starts_empty() -> None:
    calls = purchase_calls()
    ex1 = FakeModel([*calls[:3], text("hp-100 is in the cart.")])
    ex2 = FakeModel([*calls[3:], text("Order placed.")])
    site, planner, run = planner_run(
        [
            text("NEXT: Find hp-100 and add it to the cart."),
            text("NEXT: Check out and place the order."),
            text("DONE: Ordered hp-100."),
        ],
        [ex1, ex2],
    )
    assert run.outcome == "done"
    assert site.orders == [["hp-100"]]
    assert [snapshots(m) for m in planner.received] == [0, 0, 0]
    assert [snapshots(m) for m in ex2.received] == [0, 1, 2, 3]  # nothing from step 1
    assert "hp-100 is in the cart." in json.dumps(planner.received[1])


def test_planner_executor_sends_fewer_snapshot_characters_than_one_loop() -> None:
    loop = FakeModel([*purchase_calls(), text("Order placed.")])
    run_single_loop(build_single_loop(ShopSite(), loop), TASK)
    calls = purchase_calls()
    ex1 = FakeModel([*calls[:3], text("In cart.")])
    ex2 = FakeModel([*calls[3:], text("Placed.")])
    _, planner, _ = planner_run(
        [text("NEXT: add hp-100"), text("NEXT: place the order"), text("DONE: ok")], [ex1, ex2]
    )

    def sent(*models: FakeModel) -> int:
        return sum(len(json.dumps(m)) for model in models for m in model.received)

    # Illustrative only: the ratio depends on the scripted steps and the page size.
    assert sent(planner, ex1, ex2) < 0.7 * sent(loop)


def test_planner_reply_outside_the_protocol_ends_the_run() -> None:
    _, _, run = planner_run([text("I will now open the shop.")], [])
    assert run.outcome == "bad_reply"


def test_planner_step_budget() -> None:
    executors = [FakeModel([text("Nothing to do.")]) for _ in range(2)]
    _, _, run = planner_run([text("NEXT: look around")] * 2, executors, max_steps=2)
    assert run.outcome == "step_budget"
    assert len(run.steps) == 2


# --- 5. supervisor --------------------------------------------------------------------------------

BUDGET_TASK = "Buy headphones, my budget is $50."


def test_as_tool_worker_sees_only_what_the_supervisor_wrote() -> None:
    site = ShopSite()
    catalog = FakeModel(
        [call("open_url", "c1", url="/search?q=headphones"), text("hp-100 $45, hp-200 $129")]
    )
    supervisor = FakeModel(
        [call("catalog_specialist", "s1", input="Find headphones."), text("Found two.")]
    )
    build_supervisor(site, factory(catalog, FakeModel([]), supervisor))(BUDGET_TASK)
    assert "$50" not in json.dumps(catalog.received)  # the budget never reached the worker
    assert "hp-200 $129" in json.dumps(supervisor.received[-1])


def test_typed_handoff_carries_the_constraint_as_a_field() -> None:
    worker = FakeModel([text("hp-100 $45")])
    supervisor = FakeModel(
        [call("find_products", "s1", query="headphones", max_price=50), text("hp-100.")]
    )
    build_typed_supervisor(ShopSite(), factory(supervisor, worker))(BUDGET_TASK)
    assert "price <= $50.00" in json.dumps(worker.received[0])


def test_typed_handoff_rejects_a_call_without_the_constraint() -> None:
    worker = FakeModel([text("unused")])
    supervisor = FakeModel([call("find_products", "s1", query="headphones"), text("Sorry.")])
    build_typed_supervisor(ShopSite(), factory(supervisor, worker))(BUDGET_TASK)
    assert worker.received == []
    tool_result = json.dumps(supervisor.received[-1][-1])
    assert '"status": "error"' in tool_result and "max_price" in tool_result


def test_typed_buy_rejects_a_product_over_the_limit() -> None:
    site = ShopSite()
    worker = FakeModel([text("unused")])
    supervisor = FakeModel(
        [call("buy_product", "s1", sku="hp-200", max_price=50), text("Cannot buy that.")]
    )
    build_typed_supervisor(site, factory(supervisor, worker))(BUDGET_TASK)
    assert worker.received == [] and site.orders == []
    assert '"status": "error"' in json.dumps(supervisor.received[-1][-1])


def test_open_url_rejects_another_host() -> None:
    import pytest

    with pytest.raises(ValueError):
        ShopSite().open_url("//evil.example/x")


def test_blank_env_values_fall_back_to_defaults(monkeypatch) -> None:
    # A .env copied from .env.example leaves TENSORUP_PROVIDER= and TENSORUP_MODEL= blank.
    from controlflow.model import DEFAULT_OPENAI_MODEL, build_model

    monkeypatch.setenv("TENSORUP_PROVIDER", "")
    monkeypatch.setenv("TENSORUP_MODEL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert build_model().config["model_id"] == DEFAULT_OPENAI_MODEL
