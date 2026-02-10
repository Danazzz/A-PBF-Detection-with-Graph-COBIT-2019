from __future__ import annotations
import os, atexit, tempfile, json, uuid
from pathlib import Path
from datetime import datetime

import numpy as np
import math
import pandas as pd
import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network
from typing import Dict, List, Tuple
from functools import wraps as _wraps
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# ────────── detektor & skor ──────────
from PBF.skipped   import run_skipped_activity, get_skipped_scores
from PBF.pattern   import run_wrong_pattern, get_wrong_pattern_scores
from PBF.decision  import run_wrong_decision, get_wrong_decision_scores
from PBF.duty      import run_wrong_duty,            get_wrong_duty_scores
from PBF.resource  import run_wrong_resource,        get_wrong_resource_scores
from PBF.time      import run_wrong_throughput_time, get_wrong_throughput_time_scores
# ──────────────────────────────────────
DETECTOR_FUNC = {
    "skipped_activity"     : run_skipped_activity,
    "wrong_pattern"        : run_wrong_pattern,
    "wrong_decision"       : run_wrong_decision,
    "wrong_duty"           : run_wrong_duty,
    "wrong_resource"       : run_wrong_resource,
    "wrong_throughput_time": run_wrong_throughput_time,
}
DETECTOR_SCORE_FUNC = {
    "skipped_activity"     : get_skipped_scores,
    "wrong_pattern"        : get_wrong_pattern_scores,
    "wrong_decision"       : get_wrong_decision_scores,
    "wrong_duty"           : get_wrong_duty_scores,
    "wrong_resource"       : get_wrong_resource_scores,
    "wrong_throughput_time": get_wrong_throughput_time_scores,
}

METRIC_LIST = list(DETECTOR_FUNC)

KIND_MAP: Dict[str, str] = {
    "skipped_activity"     : "skipped",
    "wrong_pattern"        : "pattern",
    "wrong_decision"       : "decision",
    "wrong_duty"           : "duty",
    "wrong_resource"       : "resource",
    "wrong_throughput_time": "time",
}

st.set_page_config(page_title="PBF Detector", layout="wide")
STATE_DEFAULTS = {
    "df_m":        pd.DataFrame(),  # hasil deteksi aktif
    "anom_type":   None,             # key detektor aktif
    "best_thr":    None,             # θ aktif (grid / corporate)
    "score_df":    pd.DataFrame()
}
for k, v in STATE_DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ────────── Neo4j driver ──────────
@st.cache_resource(show_spinner=False)
def get_driver():
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI",  "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"),
              os.getenv("NEO4J_PWD",  "Mahaputra97"))
    )
driver = get_driver(); atexit.register(driver.close)

# ──────────────────────────────────────────────
# APO12  ·  helper – compute & save & draw
# ──────────────────────────────────────────────
# ── 1. helper duplicate‑safe ────────────────────────────────
def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan hanya 1 row / CaseId."""
    return df.drop_duplicates("CaseId", keep="first")

# ganti fetch_scores
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_scores(det_key: str) -> pd.DataFrame:
    func = DETECTOR_SCORE_FUNC.get(det_key,
                                   lambda _: pd.DataFrame(columns=["CaseId","score"]))
    with driver.session() as s:
        return _dedup(func(s))
    

def run_all_detectors_get_tp() -> tuple[set[str], pd.DataFrame]:
    gt_map = fetch_gt_map()
    tp_union, rows = set(), []

    # ⬇️  tambahkan mapper sekali saja
    label_map = {
        "skipped_activity"     : "skipped",
        "wrong_pattern"        : "wrong_pattern",
        "wrong_decision"       : "wrong_decision",
        "wrong_duty"           : "wrong_duty",
        "wrong_resource"       : "wrong_resource",
        "wrong_throughput_time": "wrong_time",
    }

    for m in METRIC_LIST:
        df_score = fetch_scores(m)
        if df_score.empty:
            continue

        theta  = (fetch_risk_param(m) or {}).get("theta", 0.75)
        pred   = np.where(df_score["score"] < theta, "anomali", "normal")
        true_l = label_map.get(m, m)                 # ← gunakan mapper

        for cid, sc, p in zip(df_score["CaseId"], df_score["score"], pred):
            cid = str(cid)
            rows.append({"CaseId": cid, "metric": m, "score": sc, "pred": p})
            if p == "anomali" and gt_map.get(cid, "") == true_l:
                tp_union.add(cid)

    df_all = pd.DataFrame(rows).drop_duplicates(["CaseId", "metric"])
    return tp_union, df_all

# ═══════════════════════════════════════════════════════
#  APO12 – Risk-Prioritised Metrics   (tanpa ground truth)
# ═══════════════════════════════════════════════════════
# ──  helper: kategori Low/Med/High  ──────────────────────────
def band(val: float, tol: float) -> str:
    """Low ≤ 0.5·tol  ·  Medium ≤ tol  ·  High > tol"""
    if val <= 0.5 * tol:
        return "Low"
    if val <= tol:
        return "Medium"
    return "High"

@st.cache_data(ttl=900, show_spinner=False)
def toleransi_portofolio(n_case: int) -> tuple[float,float]:
    """
    Hitung ambang portofolio T01 (%), T02 (#case) dari wᵢ & θ_base.
    Rumus:
        τ_port = (1-θ_base) + θ_base·Σ wᵢ²
        T01 = τ_port·100   ;   T02 = τ_port·N_CS
    """
    # ambil wᵢ & θ_base
    with driver.session() as s:
        rows = s.run("MATCH (p:RiskParam) RETURN p.weight AS w, p.baseTheta AS b").data()
    if not rows:
        return 100.0, n_case        # fallback longgar

    weights   = [float(r["w"]) for r in rows if r["w"] is not None]
    theta_b   = next((float(r["b"]) for r in rows if r["b"] is not None), 0.75)
    tau_port  = (1 - theta_b) + theta_b * sum(w**2 for w in weights)
    return tau_port * 100, tau_port * n_case

REQUIRED_ATTRS_PER_LABEL: Dict[str, List[str]] = {
    "Anomaly":    ["CaseId", "Name", "Time", "employee_name", "employee_role", "case_Status", "document"],
    "Score":      ["caseId", "kind", "value", "updatedAt"],
    "RiskParam":  ["metric", "likelihood", "impact", "importance", "riskScore", "weight", "theta", "baseTheta"],
    "APO12Metric": ["metric", "value", "runId", "time", "anomaly"],
}

def total_caseids() -> int:
    """Hitung CaseId unik di graf actual (Anomaly) atau Score."""
    qry = "MATCH (a:Anomaly) RETURN count(DISTINCT a.CaseId) AS n"
    return driver.session().run(qry).single()["n"]

def _label_attr_completeness(sess, label: str, attrs: List[str]) -> Tuple[int, int]:
    """Return (filled_attr, total_attr) for a specific label."""
    q = f"MATCH (n:`{label}`) RETURN properties(n) AS props"
    rows = sess.run(q)
    filled = total = 0
    for r in rows:
        props = r["props"] or {}
        for a in attrs:
            total += 1
            if a in props and props[a] not in (None, ""):
                filled += 1
    return filled, total


def calc_attr_completeness(flagged_only=True) -> float:
    with driver.session() as s:
        if flagged_only:
            # Hanya hitung node :Anomaly yang flag_anomali = true
            f, t = _label_attr_completeness(
                s, "Anomaly",
                ["CaseId","Name","Time","employee_name",
                 "employee_role","case_Status","document"])
        else:
            f = t = 0
            for lab, attrs in REQUIRED_ATTRS_PER_LABEL.items():
                x, y = _label_attr_completeness(s, lab, attrs)
                f += x; t += y
        return 0 if t==0 else round(100*f/t,2)

def compute_apo12_all(theta_map: dict[str, float],
                      risk_pct: float = 0.10   # ← slider kini maknai “ambang skor”
                     ) -> tuple[dict, pd.DataFrame]:

    flagged = flagged_cases(theta_map)
    if not flagged:
        return {"APO12.01":0,"APO12.02":0,"APO12.03":0}, pd.DataFrame()

    # ── ranking & skor hanya utk yang ter-flag
    df_rank = (compute_global_risk_rank(theta_map)
                 .query("CaseId in @flagged"))

    # ── high-risk = skor ≥ ambang
    hi_set = set(df_rank.loc[df_rank.total_risk >= risk_pct, "CaseId"])
    n_flagged = len(flagged)
    n_hi      = len(hi_set)

    apo = {
        "APO12.01": round(n_hi / n_flagged * 100, 2),   # % high-risk di antara flagged
        "APO12.02": n_hi,                               # # high-risk caseId
        "APO12.03": calc_attr_completeness(flagged_only=True),           # kelengkapan atribut
    }

    # # kategori (pakai n_flagged sbg basis toleransi portofolio)
    # T01, T02           = toleransi_portofolio(n_flagged)
    # apo["APO12.01_cat"] = band(apo["APO12.01"], T01)
    # apo["APO12.02_cat"] = band(apo["APO12.02"], T02)

    # label utk visual
    lab_df = (pd.DataFrame({"CaseId": list(flagged)})
                .assign(label=lambda d:
                    d.CaseId.apply(lambda c:
                        ",".join(code for code, cond in
                                 (("01", c in hi_set),
                                  ("02", c in hi_set),
                                  ("03", True)) if cond))))
    return apo, lab_df

def build_theta_map(active_metric:str|None=None,
                    active_theta:float|None=None) -> dict[str,float]:
    """
    Ambil θ perusahaan utk tiap metrik, ganti dengan θ aktif
    (grid/corporate) jika sedang dihitung di antarmuka.
    """
    m2θ = {}
    for m in METRIC_LIST:
        rp = fetch_risk_param(m)
        if rp and rp["theta"] is not None:
            m2θ[m] = float(rp["theta"])
    if active_metric and active_theta is not None:
        m2θ[active_metric] = float(active_theta)
    return m2θ

# ── helper: himpunan kasus yg punya prediksi anomali
def flagged_cases(theta_map: dict[str, float]) -> set[str]:
    flagged = set()
    for m in METRIC_LIST:
        th = theta_map.get(m, 0.75)
        df = fetch_scores(m)
        if df.empty:
            continue
        flagged.update(df.loc[df["score"] < th, "CaseId"]
                         .astype(str).tolist())
    return flagged

def save_apo12_to_neo4j(metrics: dict):
    now, runid = datetime.utcnow().isoformat(), str(uuid.uuid4())[:8]
    rows = [{"metric": k,
         "val"   : v,
         "run"   : runid,
         "ts"    : now,
         "risk"  : metrics.get(f"{k}_cat", "")}   # ← NEW
        for k, v in metrics.items()
        if not k.endswith("_cat")]
    CYP="""UNWIND $rows AS r
    MERGE (m:APO12Metric {metric:r.metric, runId:r.run})
    SET m.value = r.val,
    m.riskLevel = r.risk,
    m.time = datetime(r.ts)"""
    driver.session().run(CYP, rows=rows)

def draw_apo12_case_graph(df_label: pd.DataFrame, height="480px"):
    net=Network(height=height, directed=False)
    net.add_node("APO12", label="APO12", shape="ellipse", color="#8e24aa")
    for _,row in df_label.iterrows():
        cid, lbl = row.CaseId, row.label
        color = "red" if "01" in lbl else "green"
        net.add_node(cid, label=f"{cid}\n({lbl})", color=color, shape="box")
        net.add_edge("APO12", cid)
    net.set_options('{"physics":{"enabled":false}}')
    with tempfile.NamedTemporaryFile(delete=False,suffix=".html") as f:
        net.save_graph(f.name)
        components.html(Path(f.name).read_text(),
                        height=int(height[:-2])+30, scrolling=True)
        
# ═══════════════════════════════════════════════════════
#  GLOBAL  RISK-RANK  (tanpa ground-truth, unsupervised)
# ═══════════════════════════════════════════════════════

def fetch_scores_from_db(metric: str) -> pd.DataFrame:
    """
    Ambil skor yang SUDAH disimpan pada node :Score.
    Jika detektor belum pernah dijalankan → kembalikan DataFrame kosong.
    """
    kind = KIND_MAP.get(metric)
    if kind is None:
        return pd.DataFrame()

    qry = """
    MATCH (s:Score {kind:$k})
    RETURN toString(s.caseId) AS CaseId, toFloat(s.value) AS score
    """
    with driver.session() as s:
        rows = s.run(qry, k=kind).data()
    return pd.DataFrame(rows)

def compute_global_risk_rank(theta_map: Dict[str, float]) -> pd.DataFrame:
    rows, flagged = [], set()

    for m, th in theta_map.items():     
        df = fetch_scores_from_db(m)          
        if df.empty:
            continue
        w = (fetch_risk_param(m) or {}).get("weight",
                                            1 / len(theta_map))
        flagged.update(df.loc[df.score < th, "CaseId"].astype(str))
        s_min, s_max = df.score.min(), df.score.max()
        rel = (df.score - s_min) / (s_max - s_min + 1e-9)
        rows.extend(
            {"CaseId": str(c), "risk": w * (1.0 - r)}
            for c, r in zip(df.CaseId, rel)
        )

    if not rows:
        return pd.DataFrame()
    agg = (pd.DataFrame(rows)
             .groupby("CaseId", as_index=False)
             .agg(total_risk=("risk", "sum"))
            #  .query("CaseId in @flagged")      # filter hanya yang < θ
             .sort_values("total_risk", ascending=False))
    agg.loc[agg.CaseId.isin(flagged), "total_risk"] *= 1.20

    agg["priority_rank"] = agg.total_risk.rank(method="dense",
                                               ascending=False).astype(int)
    return agg

# ────────── cache helper ──────────
@st.cache_data(ttl=900, show_spinner=False)
def fetch_gt_map() -> dict[str, str]:
    qry = """
    MATCH (a:Anomaly)
    WITH a.CaseId AS cid, collect(DISTINCT a.case_Status) AS labs
    RETURN cid,
           CASE
             WHEN any(l IN labs WHERE l <> 'normal')
                  THEN head([x IN labs WHERE x <> 'normal'])
             ELSE 'normal'
           END AS gt
    """
    with driver.session() as s:
        recs = s.run(qry)
        return {str(r["cid"]): r["gt"] for r in recs}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_baseline_tt() -> float|None:
    qry = """
    MATCH (e:Anomaly)
    WITH e.CaseId AS cid,
         head(collect(e.case_Status)) AS lab,
         min(e.Time).epochSeconds     AS t0,
         max(e.Time).epochSeconds     AS t1
    WITH lab, (t1-t0)/86400.0 AS tt
    WHERE lab <> 'wrong_time'
    RETURN percentileCont(tt,0.5) AS base
    """
    with driver.session() as s:
        rec = s.run(qry).single()
        return None if rec is None else rec["base"]

@st.cache_data(ttl=900, show_spinner=False)
def fetch_risk_param(metric:str) -> dict|None:
    with driver.session() as s:
        rec = s.run(
            "MATCH (p:RiskParam {metric:$m}) "
            "RETURN p.weight AS w, p.theta AS θ, p.riskScore AS r", m=metric
        ).single()
        return None if rec is None else {"weight": rec["w"], "theta": rec["θ"], "risk": rec["r"]}

# ────────── CSV → graf Anomaly (ingest) ──────────
CSV_TO_GRAPH = """
UNWIND $rows AS r
MERGE (a:Anomaly {
  CaseId:        r.case_id,
  Name:          r.activity_name,
  Time:          datetime(replace(r.timestamp,' ','T')),
  employee_name: r.name,
  employee_role: r.role,
  case_Status:   r.case_status,
  document:      coalesce(r.document,'')        //  ←  tambahkan baris ini
})
"""
SEQ_CYPHER = """
MATCH (a:Anomaly)
WITH a.CaseId AS cid, COLLECT(a) AS ev
UNWIND range(0,size(ev)-2) AS i
WITH ev[i] AS s1, ev[i+1] AS s2
MERGE (s1)-[:SEQUENCE]->(s2)"""
def ingest_csv_to_graph(df: pd.DataFrame) -> None:
    with driver.session() as s:
        s.run("MATCH (a:Anomaly) DETACH DELETE a")
        s.run(CSV_TO_GRAPH, rows=df.to_dict("records"))
        s.run(SEQ_CYPHER)

# ──────────────────────────────────────────────────────────────
#   util – graf referensi & graf per‑CaseId
# ──────────────────────────────────────────────────────────────
def draw_reference_graph(*, show_roles=False, show_baseline=False, height="550px"):
    EDGE_Q = """
    MATCH p = (:CaseActivity {Name:'INVOICE_AUDIT'})
             -[:SEQUENCE|XORSPLIT|XORJOIN*]->
             (:CaseActivity {Name:'INVISIBLE_TASK'})
    UNWIND relationships(p) AS r
    RETURN DISTINCT startNode(r) AS src, endNode(r) AS dst
    """
    ROLE_Q = """
    MATCH (ro:Role)-[:MAY_EXECUTE]->(ca:CaseActivity)
    RETURN ca, collect(ro.name) AS roles
    """
    with driver.session() as sess:
        edges = list(sess.run(EDGE_Q))
        roles_map = ({rec["ca"].element_id: rec["roles"] for rec in sess.run(ROLE_Q)}
                     if show_roles else {})

    net, added = Network(height=height, directed=True), set()
    for rec in edges:
        for k in ("src","dst"):
            n = rec[k]
            if n.element_id in added: continue
            added.add(n.element_id)
            lbl_base = n["Name"] if len(n["Name"])<=12 else n["Name"][:10]+"…"
            if show_roles:
                rl = roles_map.get(n.element_id, [])
                lbl = f"{lbl_base}\n({'/'.join(rl[:2])}{'…' if len(rl)>2 else ''})" if rl else lbl_base
                ttl = f"{n['Name']}<br>Roles: {', '.join(rl) if rl else '-'}"
            else:
                lbl, ttl = lbl_base, n["Name"]
            net.add_node(
                n.element_id, label=lbl, title=ttl,
                color="#29b6f6" if "Invisible_Task" in n.labels else "#ff7043",
                shape="box")
        net.add_edge(rec["src"].element_id, rec["dst"].element_id)

    net.set_options('{"physics":{"enabled":false}}')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        net.save_graph(f.name)
        components.html(Path(f.name).read_text(),
                        height=int(height.rstrip("px"))+30, scrolling=True)

    if show_baseline:
        base = fetch_baseline_tt()
        if base is not None:
            st.info(f"Baseline throughput‑time (median kasus normal): **{base:.2f} hari**")

def fmt_time(ts) -> str:
    if hasattr(ts,"to_native"): ts=ts.to_native()
    if isinstance(ts,str): ts=datetime.fromisoformat(ts.replace("Z","+00:00"))
    return ts.strftime("%d %b %Y %H:%M")

def draw_case_graph_single(cid:str, pred_flag:str, tp_fp_flag:str, height="550px"):
    q="MATCH (e:Anomaly) WHERE e.CaseId=$c OR toString(e.CaseId)=$c RETURN e ORDER BY e.Time"
    rows=list(driver.session().run(q,c=str(cid)))
    if not rows: return st.warning("CaseId tidak ditemukan.")
    start_t=fmt_time(rows[0]["e"]["Time"])
    inv=next((i for i,r in enumerate(rows) if r["e"]["Name"]=="INVISIBLE_TASK"),None)
    finish_t=fmt_time(rows[inv-1]["e"]["Time"]) if inv and inv>0 else fmt_time(rows[-1]["e"]["Time"])
    st.markdown(f"**{tp_fp_flag}:** Start Time: {start_t} &nbsp;|&nbsp; Finish time: {finish_t}")

    TP_COLOR      = {"border": "#d32f2f", "background": "#ffffff",
                 "highlight": {"border": "#d32f2f", "background": "#ffffff"}}
    NORMAL_COLOR  = {"border": "#388e3c", "background": "#ffffff",
                 "highlight": {"border": "#388e3c", "background": "#ffffff"}}

    net=Network(height=height,directed=True)
    for idx,rec in enumerate(rows):
        e=rec["e"]; nid=f"{cid}_{idx}"
        net.add_node(
            nid,
            label=e["Name"] if len(e["Name"]) <= 12 else e["Name"][:10] + "…",
            title=f"{cid}|{fmt_time(e['Time'])}|{e['employee_name']} ({e['employee_role']})|{e['document']}",
            color=TP_COLOR if tp_fp_flag == "TP" else
                TP_COLOR if pred_flag == "anomali" else
                NORMAL_COLOR,
            shape="box",
            borderWidth=2)
        if idx: net.add_edge(f"{cid}_{idx-1}", nid)
    net.set_options('{"physics":{"enabled":false}}')
    with tempfile.NamedTemporaryFile(delete=False,suffix=".html") as f:
        net.save_graph(f.name)
        components.html(Path(f.name).read_text(),
                        height=int(height[:-2])+30,scrolling=True)

# ════════════════════════════════════════════════════════════
#   Streamlit UI  –  Sidebar bagian atas
# ════════════════════════════════════════════════════════════
# ▸ Upload CSV
st.sidebar.header("Upload Event Log")
if (upl := st.sidebar.file_uploader(
        "CSV (case id,timestamp,activity,actor)", type="csv")):
    dfcsv = pd.read_csv(upl)
    st.sidebar.success(f"CSV dimuat ▸ {len(dfcsv):,} baris")
    if st.sidebar.button("Impor → Neo4j"):
        with st.spinner("Mengimpor …"): ingest_csv_to_graph(dfcsv)
        st.sidebar.success("Graf actual siap ✔")

# Risk‑Parameter editor (L‑I‑M  +  Base θ)
with st.sidebar.expander("Custom Threshold (Likelihood‑Impact‑Importance)"):
    # ── 1 · input θ_base yang dapat diubah user ──────────────────────────
    base_theta = st.number_input(
        "Base cut‑off θ (0–1)",
        min_value=0.0, max_value=1.0,
        value=0.75, step=0.01,  help="Nilai dasar sebelum dikalikan (1‑weight)"
    )
    # ── 2 · tabel L‑I‑M (muat nilai lama jika ada) ──────────────────────
    df_risk = pd.DataFrame({
        "metric": METRIC_LIST,
        "Likelihood": [3.0] * len(METRIC_LIST),   # ← float
        "Impact":     [3.0] * len(METRIC_LIST),
        "Importance": [3.0] * len(METRIC_LIST),
    })
    with driver.session() as s:
        for r in s.run(
            "MATCH (p:RiskParam) "
            "RETURN p.metric AS m, p.likelihood AS L, p.impact AS I, "
            "p.importance AS M, p.theta AS θ"):
            df_risk.loc[df_risk.metric == r["m"],
                        ["Likelihood","Impact","Importance"]] = [r["L"], r["I"], r["M"]]
    edit_df = st.data_editor(df_risk, num_rows="fixed")
    # ── 3 · tombol simpan ───────────────────────────────────────────────
    if st.button("Simpan Risk Param -> Neo4j"):
        # validasi numerik
        try:
            edit_df[["Likelihood","Impact","Importance"]] = (
                edit_df[["Likelihood","Impact","Importance"]].astype(float)
            )
        except Exception as e:
            st.error(f"Semua kolom harus numerik: {e}")
        else:
            # hitung skor, bobot, dan θ baru berdasarkan base_theta
            edit_df["riskScore"] = (
                edit_df.Likelihood * edit_df.Impact * edit_df.Importance
            )
            tot = edit_df.riskScore.sum() or 1.0
            edit_df["weight"] = edit_df.riskScore / tot
            edit_df["theta"]  = base_theta * (1 - edit_df.weight)

            # simpan ke Neo4j
            with driver.session() as s:
                s.run("MATCH (p:RiskParam) DETACH DELETE p")      
                s.run("""
                    UNWIND $rows AS r
                    MERGE (p:RiskParam {metric:r.metric})
                    SET  p.likelihood  = r.Likelihood,
                         p.impact      = r.Impact,
                         p.importance  = r.Importance,
                         p.riskScore   = r.riskScore,
                         p.weight      = r.weight,
                         p.theta       = r.theta,
                         p.baseTheta   = $θ_base
                    """, rows=edit_df.to_dict("records"), θ_base=base_theta)
            st.cache_data.clear()
            st.success("Risk parameters & base θ tersimpan")

# ▸ Pilih detektor
anom_choice = st.sidebar.selectbox(
    "Jenis anomali", METRIC_LIST, format_func=lambda s:s.replace("_"," ").title())

# ▸ SoD editor
if anom_choice == "wrong_duty":
    st.sidebar.markdown("### Separation‑of‑Duties (SoD) Rules")
    default_json = """[
  ["Accounting", ["INVOICE_AUDIT","DUE_REMINDER","REMINDER_1","REMINDER_2",
                  "REMINDER_3","SUBMIT_REF","ISSUE_CASH_NOTE",
                  "SUSPEND_CREDIT_NOTE","RECORD_RECEIPT"]],
  ["FAM", ["INVOICE_AUDIT","DUE_REMINDER","REMINDER_1","REMINDER_2",
            "REMINDER_3","APPROVE_CASH_NOTE","APPROVE_REF",
            "SUSPEND_CREDIT_NOTE","RECORD_RECEIPT"]],
  ["GM", ["INVOICE_AUDIT","APPROVE_CASH_NOTE","APPROVE_REF"]]
]"""
    txt=st.sidebar.text_area("Edit / tempel SoDRule (JSON)",value=default_json,height=210)
    if st.sidebar.button("Simpan SoDRule → Neo4j"):
        try:
            rule_list=json.loads(txt)
            rows=[{"role":r[0],"acts":r[1]} for r in rule_list if len(r)==2]
        except Exception as e:
            st.sidebar.error(f"Format JSON salah – {e}")
        else:
            keep=[r["role"] for r in rows]
            with driver.session() as s:
                s.run("""
                UNWIND $rows AS row
                MERGE (ro:Role {name:row.role})
                WITH ro,row
                OPTIONAL MATCH (ro)-[o:MAY_EXECUTE]->()
                DELETE o
                WITH ro,row
                UNWIND row.acts AS a
                MERGE (ca:CaseActivity {Name:a})
                MERGE (ro)-[:MAY_EXECUTE]->(ca)""", rows=rows)
                s.run("WITH $k AS k MATCH (r:Role) WHERE NOT r.name IN k DETACH DELETE r",k=keep)
            st.cache_data.clear(); st.sidebar.success("SoDRule diperbarui ✔")

# ▸ Jalankan deteksi 
if st.sidebar.button("Jalankan Deteksi"):
    st.cache_data.clear() 
    with st.spinner("Menjalankan detektor …"):
        with driver.session() as s: df_m = DETECTOR_FUNC[anom_choice](s)
    st.cache_data.clear()
    if "threshold" in map(str.lower,df_m.columns):
        df_m=df_m.rename(columns={c:"Threshold" for c in df_m.columns if c.lower()=="threshold"})
    cast={"Threshold":float,"TP":int,"TN":int,"FP":int,"FN":int,
          "Accuracy":float,"Precision":float,"Sensitivity":float,"F1_Score":float}
    df_m=df_m.astype({k:t for k,t in cast.items() if k in df_m.columns})

# ──────────────── setelah df_m didapat ────────────────
    st.session_state.update(df_m=df_m)  # simpan hasil
    # juga simpan skor per-CaseId
    st.session_state["score_df"] = fetch_scores(anom_choice)

# ──────────────── SECTION · HASIL DETEKSI ────────────────
df_m: pd.DataFrame = st.session_state.get("df_m", pd.DataFrame())

# placeholder agar NameError tidak pernah muncul
df_grid = pd.DataFrame()
df_corp = pd.DataFrame()
best_thr = None
best_idx_grid = best_idx_corp = None

def _bold_best(row, idx):          # helper penebal baris
    return ["font-weight:bold" if row.name == idx else "" for _ in row]

def _prep_display(df, best_idx):
    def _bold_best(row):
        return ['font-weight: bold' if row.name == best_idx else '' ] * len(row)

    return (
        df.style
          .apply(_bold_best, axis=1)          # ⬅️ axis hanya sekali
          .format(precision=3)
    )


# ── Jika SUDAH ada hasil deteksi ─────────────────────────
if not df_m.empty:

    # 1 ░ pisahkan grid & corporate
    if "Source" in df_m.columns:
        df_grid = (
            df_m[df_m["Source"] == "grid"]       # filter
                .drop(columns="Source")         # buang kolom
                .copy()
        )
        df_corp = (
            df_m[df_m["Source"] == "corporate"]
                .drop(columns="Source")
                .copy()
        )
    else:
        df_grid = df_m.copy()                     # detektor lama
                                                  
    # 2 ░ tentukan θ terbaik
    if not df_grid.empty:
        best_idx_grid = df_grid["F1_Score"].idxmax()
        best_thr = float(df_grid.at[best_idx_grid, "Threshold"])
    if not df_corp.empty:
        best_idx_corp = df_corp.index[0]
        best_thr = float(df_corp.at[best_idx_corp, "Threshold"])

    # 3 ░ TAMPILKAN tabel hanya jika ada datanya
    if not df_grid.empty:
        st.subheader("Grid-Search θ")
        st.dataframe(_prep_display(df_grid, best_idx_grid),
                     use_container_width=True)

    if not df_corp.empty:
        st.subheader("Custom Threshold (L-I-M)")
        st.dataframe(_prep_display(df_corp, best_idx_corp),
                     use_container_width=True)

    # 4 ░ info singkat threshold aktif
    if best_thr is not None:
        st.session_state["best_thr"] = best_thr
        st.markdown(f"**Threshold aktif:** {best_thr:.2f}")

# ╔══════════════════════════════════════════════════════════╗
#       Sidebar –  Integrasi ke COBIT 2019  (APO12)             
# ╚══════════════════════════════════════════════════════════╝
with st.sidebar.expander("Integrasi ke COBIT 2019 – APO12"):

    risk_pct = st.slider("Risk-appetite (porsi case ter-flag yang dikategorikan High-Risk)",
                        0.01, 0.50, value=st.session_state.get("risk_pct",0.10),
                        step=0.01, format="%.2f")
    st.session_state["risk_pct"] = risk_pct
    st.caption(f"Artinya: hanya kasus proses dalam {risk_pct*100:.0f} % skor risiko tertinggi "
               "yang dikategorikan High-Risk (APO12.01 & APO12.02).")

    k_done = driver.session().run("MATCH (s:Score) RETURN count(DISTINCT s.kind) AS k")\
                         .single()["k"]
    
    # Tombol aktif HANYA bila keenam detektor sudah ada di :Score
    btn = st.button("Hitung & Simpan APO12 (all metrics)",
                    disabled = k_done < len(METRIC_LIST))

    if btn:                                      # ← cukup syarat tombol diklik
        apo_res, lab_df = compute_apo12_all(
            build_theta_map(),
            risk_pct   # slider langsung bekerja
        )
        save_apo12_to_neo4j(apo_res)
        st.session_state["apo12_res"]   = apo_res
        st.session_state["apo12_label"] = lab_df
        st.success("APO12 disimpan & siap ditampilkan di main panel.")
    elif k_done < len(METRIC_LIST):
        st.info("Jalankan keenam detektor terlebih dahulu agar tombol aktif.")

score_df = st.session_state["score_df"]
best_thr  = st.session_state.get("best_thr") 

if best_thr is None:                                 # ← guard
    # ▸ coba fallback ke θ perusahaan
    rp = fetch_risk_param(anom_choice)
    if rp and rp["theta"] is not None:
        best_thr = float(rp["theta"])
    # ▸ atau ke grid-search terbaik (jika ada)
    elif not df_m.empty and "F1_Score" in df_m.columns:
        best_thr = float(df_m.loc[df_m["F1_Score"].idxmax(), "Threshold"])

if best_thr is None and score_df.empty:
    st.warning("Belum ada threshold aktif. Jalankan deteksi atau simpan Risk-Parameter.")
else:
    pred = {str(c): ("anomali" if s < best_thr else "normal")
            for c, s in zip(score_df.CaseId, score_df.score)}
    
    gt_map = fetch_gt_map()  
    # ambil ground truth
    label_map = {
        "skipped_activity": "skipped",
        "wrong_pattern":    "wrong_pattern",
        "wrong_decision":   "wrong_decision",
        "wrong_duty":       "wrong_duty",
        "wrong_resource":   "wrong_resource",
        "wrong_throughput_time": "wrong_time",
    }
    true_lbl = label_map.get(anom_choice, anom_choice)
    tp_ids = [cid for cid, p in pred.items()
              if p == "anomali" and gt_map.get(cid, "") == true_lbl]

    if tp_ids:
        sel = st.selectbox("Pilih CaseId True-Positive (atau Graf SOP)",
                           ["Graf SOP"] + sorted(tp_ids), format_func=str)
        if sel == "Graf SOP":
            draw_reference_graph(show_roles=anom_choice in
                                 ("wrong_duty", "wrong_resource"),
                                 show_baseline=anom_choice ==
                                 "wrong_throughput_time")
        else:
            draw_case_graph_single(sel, "anomali", "TP")
    else:
        st.info("Tidak ada True-Positive pada threshold aktif.")
        draw_reference_graph(show_roles=anom_choice in
                             ("wrong_duty", "wrong_resource"),
                             show_baseline=anom_choice ==
                             "wrong_throughput_time")
        
def draw_apo12_case_graph(
        df_label: pd.DataFrame,
        metric_filter: str = "All",
        page_idx: int = 0,
        page_size: int = 5,
        height: str = "480px"):
    """Visual APO12 dengan filter metric & paginasi."""
    # ── filter berdasarkan metric ───────────────
    if metric_filter != "All":
        code = {"APO12.01": "01", "APO12.02": "02", "APO12.03": "03"}[metric_filter]
        df_label = df_label[df_label["label"].str.contains(code)]

    if df_label.empty:
        st.warning("Tidak ada kasus proses untuk filter ini.")
        return

    # ── ambil subset 5-per-halaman ──────────────
    start = page_idx * page_size
    subset = df_label.iloc[start:start + page_size]

    net = Network(height=height, directed=False)
    net.add_node("APO12", label="APO12", shape="ellipse", color="#8e24aa")

    for _, row in subset.iterrows():
        cid, lbl = row.CaseId, row.label
        color = ("#d32f2f" if "01" in lbl                   # merah → APO12.01
                 else "#1976d2" if "02" in lbl              # biru  → APO12.02
                 else "#388e3c")                            # hijau → APO12.03
        net.add_node(str(cid), label=f"{cid}", color=color, shape="box",
                     title=f"Label: {lbl}")
        net.add_edge("APO12", str(cid))

    net.set_options('{"physics":{"enabled":false}}')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        net.save_graph(f.name)
        components.html(Path(f.name).read_text(),
                        height=int(height[:-2]) + 30, scrolling=True)

# ── tampilkan hasil APO‑12 bila tersedia ──
if "apo12_res" in st.session_state:
    st.header("COBIT 2019 – APO12")
        # tampilkan + pewarnaan sederhana
    df_apo = pd.DataFrame([st.session_state["apo12_res"]])
    cat_cols = [c for c in df_apo.columns if c.endswith("_cat")]

    def _color(r):
        style = []
        for c in r.index:
            if c.endswith("_cat"):
                style.append(
                    "color:#d32f2f;font-weight:bold" if r[c]=="High" else
                    "color:#fbc02d;font-weight:bold" if r[c]=="Medium" else
                    "color:#388e3c;font-weight:bold")
            else:
                style.append("")
        return style

    st.dataframe(df_apo.style.apply(_color, axis=1),
                 use_container_width=True)

    df_label = st.session_state["apo12_label"]

    # ── UI filter metric ───────────────────────
    met_opt = st.selectbox("Filter metric",
                           ["All", "APO12.01", "APO12.02", "APO12.03"],
                           index=0)

    # ── hitung halaman berdasarkan 5 case/hal ─
    page_size = 5
    if met_opt == "All":
        total_rows = len(df_label)
    else:
        code = {"APO12.01":"01","APO12.02":"02","APO12.03":"03"}[met_opt]
        total_rows = df_label["label"].str.contains(code).sum()

    max_page = max(1, math.ceil(total_rows / page_size))
    page = st.number_input("Halaman", 1, max_page, 1, step=1) - 1  # 0-based idx

    draw_apo12_case_graph(df_label,
                          metric_filter=met_opt,
                          page_idx=page,
                          page_size=page_size)


# ──────────  MAINTENANCE · batch-delete aman  ──────────
def _batch_delete(sess, label: str, lim: int = 20_000) -> None:
    """
    Hapus semua node ber-label `label` secara bertahap (≤ lim per transaksi)
    agar tidak melampaui dbms.memory.transaction.total.max.
    """
    cypher = f"""
    MATCH (n:{label})
    WITH n LIMIT $lim
    DETACH DELETE n
    RETURN count(*) AS deleted
    """
    while True:
        if sess.run(cypher, lim=lim).single()["deleted"] == 0:
            break

def clear_anomaly_and_score() -> None:
    """
    Bersihkan *hanya* :Score dan :Anomaly.
    :Case dibiarkan supaya bisa dipakai ulang.
    """
    with driver.session() as s:
        for lab in ("Score", "Anomaly"):     # urutan penting: Score dulu
            _batch_delete(s, lab)

# ── Tombol "hapus Anomaly + Score" ────────────────────────
if st.sidebar.button("🗑️  Hapus Anomaly & Score"):
    with st.spinner("Menghapus node …"):
        clear_anomaly_and_score()
        st.cache_data.clear()                # bersihkan cache Streamlit
    st.sidebar.success("Anomaly & Score dihapus. Struktur referensi tetap aman.")
