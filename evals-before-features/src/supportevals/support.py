"""SupportDesk: a mocked knowledge base and case store, and the three tools the agent gets."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from strands import tool

log = logging.getLogger(__name__)

# Stands in for the retrieval step of a RAG pipeline: same contract (query in, cited passages out).
KNOWLEDGE_BASE = {
    "KB-101": "Refunds: items can be returned for a full refund within 30 days of delivery. "
    "After 30 days we offer store credit only, up to 90 days.",
    "KB-102": "Shipping: standard delivery takes 3 to 5 business days. Express delivery "
    "takes 1 business day and costs $12.",
    "KB-103": "Password reset: use 'Forgot password' on the sign-in page. The reset link "
    "expires after 60 minutes.",
}
MAX_QUERY_CHARS = 200
TOP_K = 2


@dataclass
class SupportCase:
    case_id: str
    customer_id: str
    summary: str
    status: str = "open"
    resolution: str | None = None


def default_cases() -> dict[str, SupportCase]:
    return {
        "CS-1001": SupportCase("CS-1001", "C-42", "Headphones not pairing over Bluetooth"),
        "CS-1002": SupportCase("CS-1002", "C-42", "Express delivery arrived late"),
        "CS-2001": SupportCase("CS-2001", "C-77", "Charged twice for one order"),
    }


@dataclass
class SupportDesk:
    """State behind the tools. customer_id comes from the signed-in session, never the model."""

    customer_id: str
    cases: dict[str, SupportCase] = field(default_factory=default_cases)
    writes: int = 0

    def search(self, query: str) -> list[tuple[str, str]]:
        query = query.strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"query must be 1 to {MAX_QUERY_CHARS} characters")
        words = set(re.findall(r"[a-z]+", query.lower()))
        scored = [
            (len(words & set(re.findall(r"[a-z]+", body.lower()))), doc_id, body)
            for doc_id, body in KNOWLEDGE_BASE.items()
        ]
        hits = sorted((s for s in scored if s[0] > 0), reverse=True)[:TOP_K]
        return [(doc_id, body) for _, doc_id, body in hits]

    def get(self, case_id: str) -> SupportCase | None:
        case = self.cases.get(case_id.strip().upper())
        # Another customer's case looks the same as a missing one: no existence leak.
        return case if case and case.customer_id == self.customer_id else None

    def resolve(self, case_id: str, resolution: str) -> str:
        case = self.get(case_id)
        if case is None:
            return f"No case {case_id} found for this customer."
        if case.status == "resolved":
            # Idempotent: a retried or repeated call does not write twice.
            return f"Case {case.case_id} is already resolved."
        case.status, case.resolution = "resolved", resolution.strip()[:500]
        self.writes += 1
        log.info("case resolved case_id=%s customer_id=%s", case.case_id, self.customer_id)
        return f"Case {case.case_id} resolved."


def make_tools(desk: SupportDesk) -> list:
    @tool
    def search_kb(query: str) -> str:
        """Search the help-center knowledge base. Returns passages with their article ids.

        Args:
            query: What to look up, in a few words.
        """
        try:
            hits = desk.search(query)
        except ValueError as e:
            return f"Error: {e}"
        return "\n".join(f"[{doc_id}] {body}" for doc_id, body in hits) or "No articles found."

    @tool
    def get_case(case_id: str) -> str:
        """Look up one of the signed-in customer's support cases.

        Args:
            case_id: The case id, for example CS-1001.
        """
        case = desk.get(case_id)
        if case is None:
            return f"No case {case_id} found for this customer."
        return f"{case.case_id} ({case.status}): {case.summary}"

    @tool
    def resolve_case(case_id: str, resolution: str) -> str:
        """Mark one of the signed-in customer's cases as resolved.

        Args:
            case_id: The case id, for example CS-1001.
            resolution: One sentence on how the issue was fixed.
        """
        return desk.resolve(case_id, resolution)

    return [search_kb, get_case, resolve_case]
