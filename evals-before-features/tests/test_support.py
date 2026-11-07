import pytest

from supportevals.agent import build_agent
from supportevals.model import FakeModel, call, text
from supportevals.support import SupportDesk, make_tools


def test_search_returns_cited_passages_and_validates_the_query():
    desk = SupportDesk(customer_id="C-42")
    assert desk.search("refund within days")[0][0] == "KB-101"
    with pytest.raises(ValueError):
        desk.search("   ")
    with pytest.raises(ValueError):
        desk.search("x" * 201)


def test_another_customers_case_is_invisible_and_cannot_be_resolved():
    desk = SupportDesk(customer_id="C-42")
    assert desk.get("CS-2001") is None
    assert desk.resolve("CS-2001", "closed") == "No case CS-2001 found for this customer."
    assert desk.cases["CS-2001"].status == "open"
    assert desk.writes == 0


def test_resolve_is_idempotent():
    desk = SupportDesk(customer_id="C-42")
    assert desk.resolve("cs-1001", "fixed") == "Case CS-1001 resolved."
    assert desk.resolve("CS-1001", "fixed again") == "Case CS-1001 is already resolved."
    assert desk.writes == 1
    assert desk.cases["CS-1001"].resolution == "fixed"


def test_harness_agent_gets_only_the_support_tools():
    model = FakeModel([text("Hello.")])
    agent = build_agent(model, SupportDesk(customer_id="C-42"))
    agent("hi")
    assert sorted(model.offered_tools) == ["get_case", "resolve_case", "search_kb"]


def test_agent_tool_call_reaches_the_desk():
    desk = SupportDesk(customer_id="C-42")
    model = FakeModel(
        [call("resolve_case", "t1", case_id="CS-1002", resolution="Refunded"), text("Done.")]
    )
    build_agent(model, desk)("CS-1002 is sorted, close it")
    assert desk.cases["CS-1002"].status == "resolved"
    assert [t.tool_name for t in make_tools(desk)] == ["search_kb", "get_case", "resolve_case"]
