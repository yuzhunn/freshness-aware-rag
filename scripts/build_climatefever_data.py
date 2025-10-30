import os, re, random, json, csv
from pathlib import Path
from datetime import datetime, timedelta
from datasets import load_dataset
import dateparser
from tqdm import tqdm

random.seed(42)

OUT_DOCS = Path("data/docs.csv")
OUT_DIALOGS = Path("data/dialogs.jsonl")

# ======= you can tweak this =======
N_TOTAL = 40             # 改为 30/50 也可以
SPLIT = {
    "MemTrue_RAGStale": 15, # 应信 Memory（RAG 旧）
    "RAGTrue_MemRumor": 15, # 应信 RAG（Memory 传言/假）
    "Unknown": 5,          # 应拒答
    "Edge": 5,             # 边界/难例
}
MIN_EVID_LEN = 60           # 证据文本最短长度
# ==================================

RELIAB = {"wikipedia_revision":"High","dept_site":"High","lms_post":"Medium","student_rumor":"Low","peer_chat":"Low"}
MONTHS = ("January","February","March","April","May","June","July","August","September","October","November","December")
ISO_RE   = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")  # ISO
MONTH_RE = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+([1-9]|[12]\d|3[01]),\s*(20\d{2})\b", re.IGNORECASE)
MONTH2NUM = {m.lower(): f"{i:02d}" for i,m in enumerate(MONTHS, start=1)}

def to_iso_dates(text: str):
    out = []
    for y,m,d in ISO_RE.findall(text):
        out.append(f"{y}-{m}-{d}")
    for mon, day, year in MONTH_RE.findall(text):
        out.append(f"{year}-{MONTH2NUM[mon.lower()]}-{str(day).zfill(2)}")
    return out

def shift_days(iso: str, days: int) -> str:
    try:
        dt = datetime.strptime(iso, "%Y-%m-%d") + timedelta(days=days)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return ""

def choose_bucket(counts):
    for k in ("MemTrue_RAGStale","RAGTrue_MemRumor","Unknown","Edge"):
        if counts[k] < SPLIT[k]:
            return k
    return None

def main():
    os.makedirs("data", exist_ok=True)

    # 重点：用 Climate-FEVER（带证据句文本，免脚本）
    # 推荐
    ds = load_dataset("tdiggelm/climate_fever", split="test")



    usable = []
    for ex in tqdm(ds, desc="Filter claims with dated evidence"):
        claim = ex.get("claim", "")
        # evidences 字段通常是 list[ { "evidence": <str>, ... }, ... ]
        evs = ex.get("evidences") or ex.get("evidence") or []
        if not (claim and isinstance(evs, list)):
            continue
        # 拼接证据文本（只取字符串字段）
        texts = []
        for it in evs:
            if isinstance(it, dict):
                # 常见键名：'evidence', 也可能叫 'evidence_text'
                s = it.get("evidence") or it.get("evidence_text") or ""
                if isinstance(s, str) and s.strip():
                    texts.append(s.strip())
            elif isinstance(it, str) and it.strip():
                texts.append(it.strip())
        if not texts:
            continue
        ev_text = " ".join(texts)
        if len(ev_text) < MIN_EVID_LEN:
            continue
        dates = to_iso_dates(ev_text)
        if not dates:
            continue

        value_doc = dates[0]             # 用证据中首个日期当作“文档记录值”
        time_doc  = value_doc            # 作为文档时间代理
        title     = ex.get("evidence_page", "") or ex.get("evidence_wiki_url", "") or "Wikipedia"
        usable.append({
            "title": title,
            "claim": claim,
            "value_doc": value_doc,
            "time_doc": time_doc,
            "text_doc": ev_text[:1200]
        })

    need = sum(SPLIT.values())
    if len(usable) < need:
        raise RuntimeError(f"Not enough dated items: got {len(usable)}, need {need}. Try lowering filters.")

    random.shuffle(usable)

    counts = {k:0 for k in SPLIT}
    docs_rows, dialogs_rows = [], []
    used = 0; i = 0
    while used < N_TOTAL and i < len(usable):
        ex = usable[i]; i += 1
        bucket = choose_bucket(counts)
        if not bucket:
            break

        course = f"CLM{used:03d}"
        dialog_id = f"{course.lower()}_{bucket.lower()}"

        value_doc = ex["value_doc"]
        time_doc  = ex["time_doc"]
        source_doc = "wikipedia_revision"
        reliab_doc = RELIAB[source_doc]
        title = ex["title"] or "Wikipedia"
        text_doc = f"{title} — Evidence:\n{ex['text_doc']}\n\nRecorded date: {value_doc}."

        # 生成 Memory（对话更新）
        if bucket == "MemTrue_RAGStale":
            source_mem = "wikipedia_revision"; reliab_mem = RELIAB[source_mem]
            value_mem = shift_days(value_doc, random.choice([7,14,21,30]))
            time_mem  = value_mem
            update_text = f"Update (Wikipedia): the date is now {value_mem} ({time_mem})."
            gold = value_mem
        elif bucket == "RAGTrue_MemRumor":
            source_mem = "student_rumor"; reliab_mem = RELIAB[source_mem]
            value_mem = shift_days(value_doc, random.choice([-21,-14,-7,7,14,21,30]))
            time_mem  = ""  # 传言通常没时间戳
            update_text = f"I heard someone said the date changed to {value_mem} (not sure)."
            gold = value_doc
        elif bucket == "Unknown":
            source_mem = "peer_chat"; reliab_mem = RELIAB[source_mem]
            value_mem = ""
            time_mem  = ""
            update_text = "A friend mentioned it might be moved next month."
            gold = "unknown"
        else:  # Edge
            source_mem = random.choice(["lms_post","wikipedia_revision"])
            reliab_mem = RELIAB[source_mem]
            value_mem = shift_days(value_doc, random.choice([-3,-2,-1,1,2,3]))
            time_mem  = value_mem if random.random()<0.7 else ""
            update_text = f"New post: the date is {value_mem} ({time_mem})."
            # 简单规则决定 gold：若没时间则 unknown；否则（High 且不同）偏向 mem，否则 doc
            if not time_mem:
                gold = "unknown"
            else:
                gold = value_mem if (source_mem=="wikipedia_revision" and value_mem!=value_doc) else value_doc

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

    # 写文件
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
