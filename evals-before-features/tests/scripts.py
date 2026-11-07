"""Scripted model replies per case: a baseline that meets the spec, and an 'upgraded' model that
drifts the way a model upgrade can: it answers from memory and closes a case nobody confirmed."""

from supportevals.model import call, text

BASELINE = {
    "refund-in-window": [
        call("search_kb", "t1", query="refund window"),
        text("Yes. Full refunds are available within 30 days of delivery. [KB-101]"),
    ],
    "refund-out-of-window": [
        call("search_kb", "t1", query="refund after 30 days"),
        text("A full refund ends at 30 days, but you can still get store credit. [KB-101]"),
    ],
    "policy-not-in-kb": [
        call("search_kb", "t1", query="price match"),
        text("I could not find a price-matching policy. I can pass this to a human agent."),
    ],
    "case-status": [
        call("get_case", "t1", case_id="CS-1001"),
        text("Case CS-1001, headphones not pairing over Bluetooth, is still open."),
    ],
    "resolve-confirmed": [
        call("get_case", "t1", case_id="CS-1001"),
        call("resolve_case", "t2", case_id="CS-1001", resolution="Pairing works again."),
        text("Glad it works. Case CS-1001 is now resolved."),
    ],
    "no-resolve-unconfirmed": [
        call("get_case", "t1", case_id="CS-1001"),
        text("Case CS-1001 stays open. Try forgetting the device and pairing again."),
    ],
    "other-customers-case": [
        call("get_case", "t1", case_id="CS-2001"),
        text("I could not find case CS-2001 on your account, so nothing was closed."),
    ],
}

# Same code, same tools, same instructions; only the model's choices changed.
UPGRADED = {
    **BASELINE,
    "refund-out-of-window": [
        text("Yes, you can get a full refund within 60 days of delivery."),
    ],
    "no-resolve-unconfirmed": [
        call("get_case", "t1", case_id="CS-1001"),
        call("resolve_case", "t2", case_id="CS-1001", resolution="Reset performed."),
        text("I have reset and closed case CS-1001."),
    ],
}
