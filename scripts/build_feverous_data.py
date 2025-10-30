import os, re, random, json, csv
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
import dateparser
from tqdm import tqdm

random.seed(42)

OUT_DOCS = Path("data/docs.csv")
OUT_DIALOGS = Path("data/dialogs.jsonl")


N_TOTAL = 100              
SPLIT = {
    "MemTrue_RAGStale": 40, # Memory true, Retrieval stale (should trust Memory)
    "RAGTrue_MemRumor": 40, # Retrieval true, Memory rumor/false (should trust Retrieval)
    "Unknown": 10,          # ambiguous/missing update (should abstain)
    "Edge": 10,             # borderline cases for sensitivity
}
MIN_EVID_LEN = 80           # evidence minimum characters


RELIAB = {"wikipedia_revision":"High","dept_site":"High","lms_post":"Medium","student_rumor":"Low","peer_chat":"Low"}
MONTHS = ("January","February","March","April","May","June","July","August","September","October","November","December")
DATE_PAT = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")  # ISO
MONTH_PAT = re.compile(
    r"\b(" + "|".join(MONTHS) + r")\s+([1-9]|[12]\d|3[01]),\s*(20\d{2})\b",
    re.IGNORECASE
)
MONTH2NUM = {m.lower(): f"{i:02d}" for i,m in enumerate(MONTHS, start=1)}

def to_iso_dates(text):
    out = []
    # ISO first
    for y,m,d in DATE_PAT.findall(text):
        out.append(f"{y}-{m}-{d}")
    # Month DD, YYYY
    for mon, day, year in MONTH_PAT.findall(text):
        out.append(f"{year}-{MONTH2NUM[mon.lower()]}-{str(day).zfill(2)}")
    # bare years → map to mid-year (July 01) to keep ISO form
    for y in re.findall(r"\b(19|20)\d{2}\b", text):
        pass  # we avoid bare years to reduce noise
    return out

def add_days(iso, days):
    try:
        dt = datetime.strptime(iso, "%Y-%m-%d")
        return (dt.fromordinal(dt.toordinal()+days)).strftime("%Y-%m-%d")
    except Exception:
        return ""

def choose_bucket(counts):
    for k in ("MemTrue_RAGStale","RAGTrue_MemRumor","Unknown","Edge"):
        if counts[k] < SPLIT[k]:
            return k
    return None

def main():
    os.makedirs("data", exist_ok=True)

    # Load FEVEROUS (train split has plenty)
    # If this fails due to network, you can download locally and pass local path.
    ds = load_dataset("fever/feverous", "default")["train"]

    # FEVEROUS fields we’ll use:
    # - "id", "claim", "evidence", "label", "wiki_page" (page title)
    # evidence is a list of lists; we’ll flatten to text via evidence sentences
    usable = []
    for ex in tqdm(ds, desc="Filter claims with dated evidence"):
        claim = ex.get("claim", "")
        label = ex.get("label", "")
        page = ex.get("wiki_page", "") or ex.get("page","")
        ev = ex.get("evidence", None)
        if not (claim and ev and isinstance(ev, list)):
            continue

        # FEVEROUS evidence items are indices; the dataset also includes "evidence_text" in some variants.
        # We'll try to recover plain text if available; if not, skip (to keep script simple).
        ev_texts = ex.get("evidence_text", None)
        if not ev_texts or not isinstance(ev_texts, list):
            continue

        # concatenate evidence into one block
        txt_blocks = [t for t in ev_texts if isinstance(t, str)]
        if not txt_blocks:
            continue
        ev_text = " ".join(txt_blocks)
        if len(ev_text) < MIN_EVID_LEN:
            continue

        dates = to_iso_dates(ev_text)
        if not dates:
            continue

        # pick a salient date from evidence to act as "document value"
        value_doc = dates[0]  # first seen date
        time_doc = value_doc  # use same as 'time' (Doc timestamp proxy)
        # Keep a trimmed text block
        text_doc = ev_text[:900]

        usable.append({
            "page": page or "Wikipedia",
            "claim": claim,
            "label": label,     # SUPPORTED / REFUTED / NOT_ENOUGH_INFO
            "value_doc": value_doc,
            "time_doc": time_doc,
            "text_doc": text_doc
        })

    if len(usable) < sum(SPLIT.values()):
        raise RuntimeError(f"Not enough dated-evidence items. Got {len(usable)}")

    random.shuffle(usable)
    counts = {k:0 for k in SPLIT}
    docs_rows, dialogs_rows = [], []
    used = 0
    i = 0
    while used < N_TOTAL and i < len(usable):
        ex = usable[i]; i += 1
        bucket = choose_bucket(counts)
        if not bucket: break

        course = f"FEV{used:03d}"
        dialog_id = f"{course.lower()}_{bucket.lower()}"

        # Retrieval (doc) — fixed from evidence
        value_doc = ex["value_doc"]
        time_doc  = ex["time_doc"]
        source_doc = "wikipedia_revision"
        reliab_doc = RELIAB[source_doc]
        title = ex["page"]
        text_doc = f"{title} — Evidence:\n{ex['text_doc']}\n\nRecorded date: {value_doc}."

        # Memory update we synthesize (with source + reliability labels)
        if bucket == "MemTrue_RAGStale":
            source_mem = "wikipedia_revision"; reliab_mem = RELIAB[source_mem]
            # newer date → +7~30 days
            value_mem = add_days(value_doc, random.choice([7,14,21,30]))
            time_mem  = value_mem
            update_text = f"Update (Wikipedia): the date is now {value_mem} ({time_mem})."
            gold = value_mem
        elif bucket == "RAGTrue_MemRumor":
            source_mem = "student_rumor"; reliab_mem = RELIAB[source_mem]
            # rumor says a different date (earlier or later), no timestamp
            shift = random.choice([-14,-7,7,14,30])
            value_mem = add_days(value_doc, shift)
            time_mem  = ""
            update_text = f"I heard someone said the date changed to {value_mem} (not sure)."
            gold = value_doc
        elif bucket == "Unknown":
            source_mem = "peer_chat"; reliab_mem = RELIAB[source_mem]
            value_mem = ""
            time_mem  = ""
            update_text = "A friend mentioned it might be moved next month."
            gold = "unknown"
        else:  # Edge
            # either LMS post (Medium) or Wikipedia (High), close in time (±3 days)
            if random.random()<0.5:
                source_mem = "lms_post"; reliab_mem = RELIAB[source_mem]
            else:
                source_mem = "wikipedia_revision"; reliab_mem = RELIAB[source_mem]
            shift = random.choice([-3,-2,-1,1,2,3])
            value_mem = add_days(value_doc, shift)
            time_mem  = value_mem if random.random()<0.7 else ""  # sometimes missing timestamp
            update_text = f"New post: the date is {value_mem} ({time_mem})."
            # rule-of-thumb gold: higher reliability or (if equal) newer date wins; missing time → unknown
            if not time_mem:
                gold = "unknown"
            else:
                gold = value_mem if reliab_mem=="High" and value_mem!=value_doc else value_doc

        turns = [
            {"role":"user","text": f"About {title}, when did this happen?"},
            {"role":"assistant","text":"The article includes a recorded date in its evidence."},
            {"role":"user","text": update_text},
            {"role":"user","text":"So what is the date now?"}
        ]

        docs_rows.append({
            "doc_id": f"d{used+1:03d}",
            "course": course,
            "slot": "date",
            "value_doc": value_doc,
            "time_doc": time_doc,
            "source_doc": source_doc,
            "reliability_doc": reliab_doc,
            "title": title,
            "text": text_doc
        })

        dialogs_rows.append({
            "dialog_id": dialog_id,
            "course": course,
            "turns": turns,
            "update": {
                "slot": "date",
                "value_mem": value_mem,
                "time_mem": time_mem,
                "source_mem": source_mem,
                "reliability_mem": reliab_mem
            },
            "question": "What is the date now?",
            "gold_truth": gold,
            "evaluation_tag": bucket
        })

        counts[bucket]+=1
        used+=1

    # write files
    OUT_DOCS.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIALOGS.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_DOCS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "doc_id","course","slot","value_doc","time_doc",
            "source_doc","reliability_doc","title","text"
        ])
        w.writeheader()
        for r in docs_rows:
            w.writerow(r)

    with open(OUT_DIALOGS, "w", encoding="utf-8") as f:
        for r in dialogs_rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

    print("Done.")
    print("Counts:", counts)
    print(f"Wrote {OUT_DOCS} and {OUT_DIALOGS}")

if __name__ == "__main__":
    main()
