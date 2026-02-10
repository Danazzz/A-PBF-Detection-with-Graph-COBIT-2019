#!/usr/bin/env python3
"""
multianomaly_dataset.py  ·  2025-06-19
==========================================================
Generator *event log* AR-monitoring dengan **6** jenis anomali:

1. skipped              – satu aktivitas SOP dihapus
2. wrong_pattern        – urutan dua aktivitas non-kritis ditukar
3. wrong_decision       – cabang REMINDER-x salah
4. wrong_duty           – pelanggaran Separation-of-Duties (seq / dec / comb)
5. wrong_resource       – person ≠ role (atau sebaliknya)
6. wrong_time           – throughput < SLA_min atau > SLA_max

CLI (example)
-------------
# 100 % normal
python multianomaly_dataset.py --outfile baseline.csv --n 2000

# campuran 95 : 5  (tiap anomali 1 %)
python multianomaly_dataset.py --outfile anomaly.csv --n 2000             \
       --prop_skipped 0.01 --prop_wrong_pattern 0.01 --prop_wrong_decision 0.01 \
       --prop_wrong_duty 0.01 --prop_wrong_resource 0.01 --prop_wrong_time 0.01
"""
from __future__ import annotations

import argparse, random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

import pandas as pd

# ═════════════════════════════════════════════════════════════
# 1 ░  Konstanta proses, personel, SoD
# ═════════════════════════════════════════════════════════════
PROCESS_FLOW: List[Dict] = [
    {"name": "INVOICE_AUDIT",      "role": "Accounting", "min": 1, "max": 2},
    {"name": "DUE_REMINDER",       "role": "Accounting", "min": 1, "max": 2},
    {"name": "REMINDER_1",         "role": "Accounting", "min": 1, "max": 2, "delay": 30},
    {"name": "RECORD_RECEIPT",     "role": "Accounting", "min": 1, "max": 3},
    {"name": "INVISIBLE_TASK",     "role": "Accounting", "min": 1, "max": 1},
    {"name": "SUBMIT_REF",         "role": "Accounting", "min": 1, "max": 3},
    {"name": "APPROVE_REF",        "role": "FAM",        "min": 1, "max": 2},
    {"name": "REMINDER_2",         "role": "Accounting", "min": 1, "max": 2, "delay": 7},
    {"name": "REMINDER_3",         "role": "Accounting", "min": 1, "max": 2, "delay": 7},
    {"name": "ISSUE_CASH_NOTE",    "role": "Accounting", "min": 1, "max": 3, "delay": 7},
    {"name": "APPROVE_CASH_NOTE",  "role": "GM",         "min": 1, "max": 2},
    {"name": "SUSPEND_CREDIT_NOTE","role": "Accounting", "min": 1, "max": 3},
]
META = {m["name"]: m for m in PROCESS_FLOW}

# Node yang *tidak* boleh di-swap / di-skip
DECISION_CRIT = {
    "REMINDER_1", "REMINDER_2", "REMINDER_3",
    "RECORD_RECEIPT", "SUBMIT_REF", "ISSUE_CASH_NOTE"
}

NAMES = {
    "Accounting": ["Dewa", "Aris", "Lenny"],
    "FAM":        ["Yunia"],
    "GM":         ["Kadek"],
}
ALL_ROLES = list(NAMES.keys())

# SoD rule-set (sequence & decision)
SOD_SEQ = [("SUBMIT_REF", "APPROVE_REF"), ("ISSUE_CASH_NOTE", "APPROVE_CASH_NOTE")]
SOD_DEC = [
    ("DUE_REMINDER", "APPROVE_REF"),
    ("DUE_REMINDER", "APPROVE_CASH_NOTE"),
    ("SUBMIT_REF",   "APPROVE_CASH_NOTE"),
]

# ═════════════════════════════════════════════════════════════
# 2 ░  Util
# ═════════════════════════════════════════════════════════════
def pick_anomaly(props: Dict[str, float]) -> str:
    p, acc = random.random(), 0.0
    for k, v in props.items():
        acc += v
        if p <= acc:
            return k
    return "normal"

def safe_swap(path: list[str]) -> None:
    """Tukar dua node non-kritis di `path` (in-place)."""
    pool = [i for i, a in enumerate(path)
            if a not in DECISION_CRIT and a != "INVISIBLE_TASK"]
    if len(pool) >= 2:
        i1, i2 = random.sample(pool, 2)
        path[i1], path[i2] = path[i2], path[i1]

# ═════════════════════════════════════════════════════════════
# 3 ░  Generator utama
# ═════════════════════════════════════════════════════════════
def generate_events(
    num_cases: int = 1000,
    start_date: str = "2022-01-01",
    sla_min_days: int = 20,
    sla_max_days: int = 60,
    # proporsi anomali
    prop_skipped: float = 0.0,
    prop_wrong_pattern: float = 0.0,
    prop_wrong_decision: float = 0.0,
    prop_wrong_duty: float = 0.0,
    prop_wrong_resource: float = 0.0,
    prop_wrong_time: float = 0.0,
) -> pd.DataFrame:

    props = {
        "skipped":        prop_skipped,
        "wrong_pattern":  prop_wrong_pattern,
        "wrong_decision": prop_wrong_decision,
        "wrong_duty":     prop_wrong_duty,
        "wrong_resource": prop_wrong_resource,
        "wrong_time":     prop_wrong_time,
    }
    if sum(props.values()) > 1.0:
        raise ValueError("Total proporsinya > 1.0")
    props["normal"] = 1.0 - sum(props.values())

    events: list[dict] = []
    base_dt = datetime.strptime(start_date + " 08:00:00", "%Y-%m-%d %H:%M:%S")

    for num in range(1, num_cases + 1):
        cid       = f"{num:05d}"
        current   = base_dt + timedelta(days=random.randint(0, 365 * 2))
        while current.year > 2024:                   # keep within 2022-2024
            current = base_dt + timedelta(days=random.randint(0, 365 * 2))

        case_type = pick_anomaly(props)

        # ── branching keputusan pembayaran ───────────────────
        pay_r = {i: random.random() > 0.6 for i in (1, 2, 3)}

        # ── susun path SOP (selalu valid) ───────────────────
        acts = ["INVOICE_AUDIT", "DUE_REMINDER", "REMINDER_1"]
        if pay_r[1]:
            acts += ["RECORD_RECEIPT", "INVISIBLE_TASK"]
        else:
            acts += ["SUBMIT_REF", "APPROVE_REF", "REMINDER_2"]
            if pay_r[2]:
                acts += ["RECORD_RECEIPT", "INVISIBLE_TASK"]
            else:
                acts += ["REMINDER_3"]
                if pay_r[3]:
                    acts += ["RECORD_RECEIPT", "INVISIBLE_TASK"]
                else:
                    acts += ["ISSUE_CASH_NOTE", "APPROVE_CASH_NOTE",
                             "SUSPEND_CREDIT_NOTE", "INVISIBLE_TASK"]

        # ── injeksi anomaly spesifik ─────────────────────────
        skipped_set: set[str] = set()
        wrong_role_prob       = 0.0
        wrong_time_mult       = 1.0
        extra_delay, skip_delay = 0, False
        wd_seq: dict[str, tuple[str, str]] = {}
        wd_dec: dict[str, tuple[str, str]] = {}

        if case_type == "skipped":
            cand = [a for a in acts
                    if a not in DECISION_CRIT and
                       a not in ("INVOICE_AUDIT", "INVISIBLE_TASK")]
            if cand:
                skipped_set.add(random.choice(cand))

        elif case_type == "wrong_pattern":
            safe_swap(acts)

        elif case_type == "wrong_decision":
            flip = random.choice([1, 2, 3])
            pay_r[flip] = not pay_r[flip]      # paksa mismatch

        elif case_type == "wrong_resource":
            wrong_role_prob = 0.30

        elif case_type == "wrong_duty":
            subtype = random.choice(["seq", "dec", "comb"])
            if subtype in ("seq", "comb"):
                pairs = [p for p in SOD_SEQ if p[0] in acts and p[1] in acts]
                if pairs:
                    a, b = random.choice(pairs)
                    role0 = META[a]["role"]
                    name0 = random.choice(NAMES[role0])
                    wd_seq = {a: (role0, name0), b: (role0, name0)}
            if subtype in ("dec", "comb"):
                pairs = [p for p in SOD_DEC if p[0] in acts and p[1] in acts]
                if pairs:
                    a, b = random.choice(pairs)
                    role0 = META[a]["role"]
                    name0 = random.choice(NAMES[role0])
                    wd_dec = {a: (role0, name0), b: (role0, name0)}

        elif case_type == "wrong_time":
            variant = random.choice(["min", "max"])
            if variant == "max":
                wrong_time_mult = random.uniform(2, 4)
                extra_delay     = random.randint(5, 15)
            else:
                wrong_time_mult = random.uniform(0.2, 0.6)
                skip_delay      = True

        # ── traversal event by event ─────────────────────────
        case_events  : list[dict] = []
        mis_decision : list[int]  = []
        mis_resource : list[int]  = []
        violated_duty: bool      = bool(wd_seq or wd_dec)

        case_start = current
        for idx, act in enumerate(acts):
            if act in skipped_set:
                continue
            meta = META[act]

            # ── INVISIBLE_TASK
            if act == "INVISIBLE_TASK":
                current += timedelta(minutes=random.randint(30, 60))
                case_events.append({
                    "case_id": cid, "timestamp": current.strftime("%Y-%m-%d %H:%M:%S"),
                    "activity_name": act, "role": "-", "name": "-",
                    "assigned_role": "-", "case_status": case_type, "document": "-"
                })
                continue

            # ───────────────────── role & person
            if act in wd_seq:
                role, person = wd_seq[act]
            elif act in wd_dec:
                role, person = wd_dec[act]
            else:
                role = (meta["role"] if random.random() > wrong_role_prob
                        else random.choice([r for r in ALL_ROLES if r != meta["role"]]))
                person = random.choice(NAMES[role])

            mis_resource.append(int(role != meta["role"]))

            # ───────────────────── document & wrong-decision flag
            if act.startswith("REMINDER_"):
                idx_r = int(act.split("_")[1])
                doc   = "receipt" if pay_r[idx_r] else "none"

                # cari next node valid (lewati yg di-skip)
                nxt_i = idx + 1
                while nxt_i < len(acts) and acts[nxt_i] in skipped_set:
                    nxt_i += 1
                actual_next = acts[nxt_i] if nxt_i < len(acts) else "END"

                exp_next = (
                    "RECORD_RECEIPT" if doc == "receipt" else
                    ("SUBMIT_REF"     if act == "REMINDER_1" else
                     "REMINDER_3"     if act == "REMINDER_2" else
                     "ISSUE_CASH_NOTE")
                )
                mis_decision.append(int(actual_next != exp_next))

            elif act == "RECORD_RECEIPT":
                doc = "receipt"
                # RECORD_RECEIPT muncul padahal tidak ada satupun receipt di REMINDER
                mis_decision.append(int(not any(pay_r.values())))
            else:
                doc = "none"

            # ───────────────────── timing
            dur = int(random.randint(meta["min"], meta["max"]) * wrong_time_mult)
            if "delay" in meta:
                d = 0 if skip_delay else meta["delay"] + extra_delay
                current += timedelta(days=d)
            if current.time() > datetime.strptime("17:00:00", "%H:%M:%S").time():
                current = (current + timedelta(days=1)).replace(hour=8, minute=0, second=0)

            case_events.append({
                "case_id": cid, "timestamp": current.strftime("%Y-%m-%d %H:%M:%S"),
                "activity_name": act, "role": role, "name": person,
                "assigned_role": meta["role"], "case_status": case_type,
                "document": doc
            })
            current += timedelta(hours=dur)
        # ── END for act ───────────────────────────────────────

        # ── label final yg konsisten & mutually-exclusive ────
        final = case_type
        if case_type == "wrong_resource" and sum(mis_resource) == 0:
            final = "normal"

        if case_type == "wrong_decision":
            if sum(mis_decision) == 0:
                final = "normal"
        elif case_type == "normal":
            if sum(mis_decision) > 0:
                final = "wrong_decision"

        if case_type == "wrong_duty" and not violated_duty:
            final = "normal"

        if case_type == "wrong_time":
            tt_days = (current - case_start).days
            if sla_min_days <= tt_days <= sla_max_days:
                final = "normal"

        # ── commit ke global list ─────────────────────────────
        for ev in case_events:
            ev["case_status"] = final
        events.extend(case_events)

    df = pd.DataFrame(events).sort_values(["case_id", "timestamp"])
    return df

# ═════════════════════════════════════════════════════════════
# 4 ░  CLI
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--n", type=int, default=1000)

    ap.add_argument("--prop_skipped",        type=float, default=0.0)
    ap.add_argument("--prop_wrong_pattern",  type=float, default=0.0)
    ap.add_argument("--prop_wrong_decision", type=float, default=0.0)
    ap.add_argument("--prop_wrong_duty",     type=float, default=0.0)
    ap.add_argument("--prop_wrong_resource", type=float, default=0.0)
    ap.add_argument("--prop_wrong_time",     type=float, default=0.0)

    ap.add_argument("--sla_min", type=int, default=20)
    ap.add_argument("--sla_max", type=int, default=60)
    args = ap.parse_args()

    df_out = generate_events(
        num_cases=args.n,
        prop_skipped=args.prop_skipped,
        prop_wrong_pattern=args.prop_wrong_pattern,
        prop_wrong_decision=args.prop_wrong_decision,
        prop_wrong_duty=args.prop_wrong_duty,
        prop_wrong_resource=args.prop_wrong_resource,
        prop_wrong_time=args.prop_wrong_time,
        sla_min_days=args.sla_min,
        sla_max_days=args.sla_max,
    )

    out = Path(args.outfile)
    df_out.to_csv(out, index=False)
    print(f"Saved → {out.resolve()}")
    print("Distribusi label:")
    print(df_out["case_status"].value_counts(normalize=True).round(3))
