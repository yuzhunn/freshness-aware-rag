# src/decider.py
from datetime import datetime

ORDER = {"Low": 0, "Medium": 1, "High": 2}

def _to_dt(s: str):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None

# --- Baselines ---
def always_rag(doc_row, upd):
    return doc_row.get("value_doc") or "unknown"

def always_mem(doc_row, upd):
    return (upd or {}).get("value_mem") or "unknown"

# --- Rule: Reliability first, recency tie-breaker, with Δ-days guard ---
def rule_weighted(doc_row, upd, delta_days=5):
    upd = upd or {}
    v_doc = doc_row.get("value_doc")
    v_mem = upd.get("value_mem")
    if not v_doc and not v_mem:
        return "unknown"

    r_doc = ORDER.get(doc_row.get("reliability_doc", "Medium"), 1)
    r_mem = ORDER.get(upd.get("reliability_mem", "Medium"), 1)

    td = _to_dt(doc_row.get("time_doc"))
    tm = _to_dt(upd.get("time_mem"))

    # Reliability priority
    if r_mem > r_doc and v_mem:
        return v_mem
    if r_doc > r_mem and v_doc:
        return v_doc

    # Tie on reliability → use recency with Δ-days guard
    if tm and td:
        diff = (tm - td).days
        if diff > delta_days and v_mem:
            return v_mem
        if diff < -delta_days and v_doc:
            return v_doc
        return "unknown"
    if tm and v_mem:
        return v_mem
    if td and v_doc:
        return v_doc
    return "unknown"

# --- Conservative: same as rule but refuses more often when missing critical info ---
def conservative(doc_row, upd, delta_days=5):
    upd = upd or {}
    if not doc_row.get("value_doc") or not upd.get("value_mem"):
        return "unknown"
    if not (doc_row.get("time_doc") or upd.get("time_mem")):
        return "unknown"
    return rule_weighted(doc_row, upd, delta_days=delta_days)
