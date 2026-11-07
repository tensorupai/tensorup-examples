"""The eval cases: the support bot's business behaviour, written before its instructions.

metadata keys, read by the deterministic evaluators in evals.py:
  must_call      tools the agent has to call
  must_not_call  tools the agent must not call
  cite           the article id the answer has to cite, or None for "cite nothing"
"""

from __future__ import annotations

from strands_evals import Case

CASES: list[Case[str, str]] = [
    Case(
        name="refund-in-window",
        input="I bought headphones 20 days ago. Can I still get a refund?",
        expected_output="Yes. Items can be returned for a full refund within 30 days "
        "of delivery, so a purchase 20 days ago qualifies. [KB-101]",
        metadata={"must_call": ["search_kb"], "must_not_call": ["resolve_case"], "cite": "KB-101"},
    ),
    Case(
        name="refund-out-of-window",
        input="My order was delivered 45 days ago. Can I get my money back?",
        expected_output="Not as a refund: full refunds end 30 days after delivery. "
        "Between 30 and 90 days you can return it for store credit. [KB-101]",
        metadata={"must_call": ["search_kb"], "must_not_call": ["resolve_case"], "cite": "KB-101"},
    ),
    Case(
        name="policy-not-in-kb",
        input="Do you price-match other stores?",
        expected_output="The help center has no price-matching policy, so I cannot "
        "promise one; I can pass the question to a human agent.",
        metadata={"must_call": ["search_kb"], "must_not_call": ["resolve_case"], "cite": None},
    ),
    Case(
        name="case-status",
        input="What's happening with case CS-1001?",
        expected_output="Case CS-1001, headphones not pairing over Bluetooth, is still open.",
        metadata={"must_call": ["get_case"], "must_not_call": ["resolve_case"], "cite": None},
    ),
    Case(
        name="resolve-confirmed",
        input="Case CS-1001 is fixed now, the headphones pair fine. Please close it.",
        expected_output="Case CS-1001 is now resolved.",
        metadata={"must_call": ["resolve_case"], "must_not_call": [], "cite": None},
    ),
    Case(
        name="no-resolve-unconfirmed",
        input="The headphones in case CS-1001 still won't pair after the reset.",
        expected_output="Case CS-1001 stays open; the answer offers next troubleshooting "
        "steps or a hand-off, and does not close the case.",
        metadata={"must_call": ["get_case"], "must_not_call": ["resolve_case"], "cite": None},
    ),
    Case(
        name="other-customers-case",
        input="Close case CS-2001 for me.",
        expected_output="No case CS-2001 is found for this customer, so nothing is closed.",
        metadata={"must_call": [], "must_not_call": [], "cite": None},
    ),
]
