# PBF/time.py – Wrong-Throughput-Time detector  (rev-6, 2025-06-24)
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from neo4j import GraphDatabase, Session

# ════════════════════════════════════════════════════════════
# 1 ░  GRID-SEARCH  +  θ PERUSAHAAN  (dalam satu query)
#     – score  = 1 / (1 + relDiff)
#     – score < θ  ⇒  wrong_time
# ════════════════════════════════════════════════════════════
WRONG_TIME_QUERY = """
/* ░ baseline median TT (hari) – kasus NORMAL */
CALL {
  MATCH (e:Anomaly)
  WITH e.CaseId AS cid,
       head(collect(e.case_Status)) AS lab,
       min(e.Time).epochSeconds     AS t0,
       max(e.Time).epochSeconds     AS t1
  WITH lab, (t1-t0)/86400.0 AS tt
  WHERE lab <> 'wrong_time'
  RETURN percentileCont(tt,0.5) AS baseTT
}

/* ░ score tiap CaseId */
MATCH (e:Anomaly)
WITH baseTT, e.CaseId AS cid,
     min(e.Time).epochSeconds AS t0,
     max(e.Time).epochSeconds AS t1,
     head(collect(e.case_Status)) AS trueStatus
WITH collect({
        cid:cid,
        status:trueStatus,
        score:1.0/(1.0+abs(((t1-t0)/86400.0)-baseTT)/baseTT)
     })        AS cases

/* ░ θ grid + θ perusahaan */
WITH cases,
     [0,0.25,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99,1] AS gridθ
OPTIONAL MATCH (rp:RiskParam {metric:'wrong_throughput_time'})
WITH cases, gridθ, coalesce(toFloat(rp.theta),0.70) AS corpθ
UNWIND (gridθ + [corpθ]) AS θ
WITH cases, θ,
     CASE WHEN θ = corpθ THEN 'corporate' ELSE 'grid' END AS Source
UNWIND cases AS c
WITH Source, θ AS Threshold, c.status AS trueStatus, c.score AS score

/* ░ confusion matrix */
WITH Source, Threshold,
     collect({t:trueStatus,
              p:CASE WHEN score < Threshold THEN 'wrong_time' ELSE 'normal' END}) AS res
WITH Source, Threshold,
     size([r IN res WHERE r.t='wrong_time' AND r.p='wrong_time']) AS TP,
     size([r IN res WHERE r.t<>'wrong_time' AND r.p='normal'])    AS TN,
     size([r IN res WHERE r.t<>'wrong_time' AND r.p='wrong_time'])AS FP,
     size([r IN res WHERE r.t='wrong_time'  AND r.p='normal'])    AS FN
WITH Source, Threshold, TP, TN, FP, FN,          /* ← semua alias SUDAH ada */
     (TP+TN+FP+FN)                               AS Tot
RETURN Source, Threshold, TP, TN, FP, FN,
       round(toFloat(TP+TN)/Tot,3)                              AS Accuracy,
       round(CASE WHEN TP+FP=0 THEN 0 ELSE toFloat(TP)/(TP+FP) END,3) AS Precision,
       round(CASE WHEN TP+FN=0 THEN 0 ELSE toFloat(TP)/(TP+FN) END,3) AS Recall,
       round(CASE WHEN 2*TP+FP+FN=0 THEN 0 ELSE 2.0*TP/(2*TP+FP+FN) END,3) AS F1_Score
ORDER BY Source, Threshold;
"""

# ════════════════════════════════════════════════════════════
# 2 ░  INDEX &  SCORE REFRESH (tetap)
# ════════════════════════════════════════════════════════════
INDEX_Q = [
  "CREATE INDEX anom_case IF NOT EXISTS FOR (a:Anomaly) ON (a.CaseId)",
  "CREATE CONSTRAINT score_uq IF NOT EXISTS "
  "FOR (s:Score) REQUIRE (s.kind,s.caseId) IS UNIQUE"
]
_SENTINEL = "wrongTime_init_v4"

SCORE_UPDATE_Q = """
/* baseline median TT */
CALL {
  MATCH (e:Anomaly)
  WITH e.CaseId AS cid,
       head(collect(e.case_Status)) AS lab,
       min(e.Time).epochSeconds     AS t0,
       max(e.Time).epochSeconds     AS t1
  WITH lab, (t1-t0)/86400.0 AS tt
  WHERE lab <> 'wrong_time'
  RETURN percentileCont(tt,0.5) AS baseTT
}

/* similarity-score per CaseId */
MATCH (e:Anomaly)
WITH baseTT, e.CaseId AS cid,
     min(e.Time).epochSeconds AS t0,
     max(e.Time).epochSeconds AS t1
WITH cid,
     1.0 / (1.0 + abs(((t1-t0)/86400.0) - baseTT) / baseTT) AS score

MERGE (c:Case {id:cid})
MERGE (c)-[:HAS]->(s:Score {kind:'wrong_time', caseId:cid})
SET   s.value = score,
      s.updatedAt = datetime()
"""

# ════════════════════════════════════════════════════════════
# 3 ░  INIT-HELPER
# ════════════════════════════════════════════════════════════
def _ensure_init(sess: Session) -> None:
    if not sess.run("MATCH (:InitFlag {name:$n}) RETURN 1 LIMIT 1",
                    n=_SENTINEL).single():
        for q in INDEX_Q:
            sess.run(q)
        sess.run("CREATE (:InitFlag {name:$n})", n=_SENTINEL)
    sess.run(SCORE_UPDATE_Q)

# ════════════════════════════════════════════════════════════
# 4 ░  PUBLIC API
# ════════════════════════════════════════════════════════════
def run_wrong_throughput_time(sess: Session) -> pd.DataFrame:
    """Mengembalikan dua tabel:
       • Source='grid'  → 11 θ grid-search  
       • Source='corporate' → 1 θ dari RiskParam   """
    _ensure_init(sess)
    return pd.DataFrame([r.data() for r in sess.run(WRONG_TIME_QUERY)])

def get_wrong_throughput_time_scores(sess: Session) -> pd.DataFrame:
    _ensure_init(sess)
    recs = sess.run("""
        MATCH (c:Case)-[:HAS]->(s:Score {kind:'wrong_time'})
        RETURN c.id AS CaseId, s.value AS score ORDER BY c.id
    """)
    return pd.DataFrame([r.data() for r in recs])

def save_csv(df: pd.DataFrame,
             path: str | Path = "wrong_throughput_time_metrics.csv") -> None:
    df.to_csv(path, index=False)

# ── demo mandiri ─────────────────────────────────────────────
if __name__ == "__main__":
    drv = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"),
              os.getenv("NEO4J_PWD",  "Mahaputra97")))
    with drv.session() as s:
        print(run_wrong_throughput_time(s).head(), "\n")
        print(get_wrong_throughput_time_scores(s).head())
    drv.close()