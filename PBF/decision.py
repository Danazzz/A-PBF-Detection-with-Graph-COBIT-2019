# PBF/decision.py  – detektor “Wrong Decision”  (2025-06-24 r5)
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from neo4j import GraphDatabase, Session

# ═══════════════════════════════════════════════════════════════
# 1 ░  GRID-SEARCH + CORPORATE θ  (satu query)
# ═══════════════════════════════════════════════════════════════
WRONG_DECISION_QUERY = r"""
/* 0 ░ urut & skip INVISIBLE_TASK */
MATCH (e:Anomaly)
WITH e.CaseId AS cid, e ORDER BY cid, e.Time
WITH cid,
     [x IN collect(e) WHERE x.Name <> 'INVISIBLE_TASK']            AS seq,
     head(collect(DISTINCT e.case_Status))                         AS trueStatus
UNWIND range(0,size(seq)-2) AS idx
WITH cid,trueStatus,
     seq[idx]   AS cur,
     seq[idx+1] AS nxt
WHERE cur.Name STARTS WITH 'REMINDER'

/* receipt flag */
OPTIONAL MATCH (rr:Anomaly {CaseId:cid, Name:'RECORD_RECEIPT'})
              WHERE rr.Time <= cur.Time
WITH cid,trueStatus,cur,nxt,
     (coalesce(cur.document,'')='receipt' OR rr IS NOT NULL)       AS hasRcptDoc

/* cabang se-harus-nya */
WITH cid, trueStatus, cur, nxt, hasRcptDoc,
     CASE
       WHEN cur.Name='REMINDER_1' AND   hasRcptDoc THEN 'RECORD_RECEIPT'
       WHEN cur.Name='REMINDER_1' AND  NOT hasRcptDoc THEN 'SUBMIT_REF'
       WHEN cur.Name='REMINDER_2' AND   hasRcptDoc THEN 'RECORD_RECEIPT'
       WHEN cur.Name='REMINDER_2' AND  NOT hasRcptDoc THEN 'REMINDER_3'
       WHEN cur.Name='REMINDER_3' AND   hasRcptDoc THEN 'RECORD_RECEIPT'
       WHEN cur.Name='REMINDER_3' AND  NOT hasRcptDoc THEN 'ISSUE_CASH_NOTE'
       ELSE nxt.Name
     END AS exp,
     nxt.Name AS act

/* hitung pelanggaran */
WITH cid, trueStatus, exp, act,
     CASE WHEN act <> exp THEN 1 ELSE 0 END AS viol,
     1 AS cnt
WITH cid, trueStatus,
     sum(viol) AS violCnt,
     sum(cnt)  AS totRem

/* score 0–1 (1 = patuh) */
WITH cid, trueStatus,
     CASE WHEN totRem = 0
          THEN 1.0
          ELSE 1.0 - toFloat(violCnt)/totRem
     END AS score

/* ░ 6. LIST θ: grid + corporate */
WITH cid,trueStatus,score,
     [0,0.25,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99,1]       AS gridθ
OPTIONAL MATCH (rp:RiskParam {metric:'wrong_decision'})
WITH cid,trueStatus,score,gridθ,
     coalesce(toFloat(rp.theta),0.75)                              AS corpθ
UNWIND (gridθ + [corpθ]) AS Threshold
WITH cid,trueStatus,score,Threshold,
     CASE WHEN abs(Threshold - corpθ) < 1e-9
          THEN 'corporate' ELSE 'grid' END                         AS Source,
     CASE WHEN score < Threshold THEN 'wrong_decision' ELSE 'normal' END AS pred

/* confusion-matrix & metrik */
WITH Source,Threshold,
     collect({t:trueStatus,p:pred})                                AS res
WITH Source,Threshold,
     size([r IN res WHERE r.t='wrong_decision' AND r.p='wrong_decision']) AS TP,
     size([r IN res WHERE r.t<>'wrong_decision' AND r.p='normal'])        AS TN,
     size([r IN res WHERE r.t<>'wrong_decision' AND r.p='wrong_decision'])AS FP,
     size([r IN res WHERE r.t='wrong_decision'  AND r.p='normal'])        AS FN
WITH Source,Threshold,TP,TN,FP,FN,(TP+TN+FP+FN) AS Tot
RETURN Source,Threshold,
       TP,TN,FP,FN,
       round(toFloat(TP+TN)/Tot,3)                       AS Accuracy,
       round(CASE WHEN TP+FP=0 THEN 0 ELSE toFloat(TP)/(TP+FP) END,3)     AS Precision,
       round(CASE WHEN TP+FN=0 THEN 0 ELSE toFloat(TP)/(TP+FN) END,3)     AS Sensitivity,
       round(CASE WHEN 2*TP+FP+FN=0 THEN 0 ELSE 2.0*TP/(2*TP+FP+FN) END,3)AS F1_Score
ORDER BY Source, Threshold;
"""

# ═══════════════════════════════════════════════════════════════
# 2 ░  INDEX & CONSTRAINT   +   SCORE REFRESH
# ═══════════════════════════════════════════════════════════════
INDEX_QUERIES = [
    "CREATE INDEX anom_case IF NOT EXISTS FOR (a:Anomaly) ON (a.CaseId)",
    "CREATE INDEX anom_name IF NOT EXISTS FOR (a:Anomaly) ON (a.Name)",
    "CREATE INDEX ca_name   IF NOT EXISTS FOR (c:CaseActivity) ON (c.Name)",
    "CREATE CONSTRAINT score_unique IF NOT EXISTS "
    "FOR (s:Score) REQUIRE (s.kind, s.caseId) IS UNIQUE",
]
_SENTINEL = "wrongDecision_init_v5"

SCORE_UPDATE_QUERY = r"""
/* hitung skor Wrong-Decision per CaseId – identik dgn sebelumnya */
MATCH (e:Anomaly)
WITH e.CaseId AS cid, e ORDER BY cid, e.Time
WITH cid, collect(e) AS seq
UNWIND range(0,size(seq)-2) AS i
WITH cid, seq[i] AS cur, seq[i+1] AS nxt
WHERE cur.Name STARTS WITH 'REMINDER'
WITH cid, cur, nxt,
     (coalesce(cur.document,'')='receipt') AS hasRcptDoc
WITH cid, cur, nxt, hasRcptDoc,
     CASE
       WHEN cur.Name='REMINDER_1' AND   hasRcptDoc THEN 'RECORD_RECEIPT'
       WHEN cur.Name='REMINDER_1' AND  NOT hasRcptDoc THEN 'SUBMIT_REF'
       WHEN cur.Name='REMINDER_2' AND   hasRcptDoc THEN 'RECORD_RECEIPT'
       WHEN cur.Name='REMINDER_2' AND  NOT hasRcptDoc THEN 'REMINDER_3'
       WHEN cur.Name='REMINDER_3' AND   hasRcptDoc THEN 'RECORD_RECEIPT'
       WHEN cur.Name='REMINDER_3' AND  NOT hasRcptDoc THEN 'ISSUE_CASH_NOTE'
       ELSE nxt.Name
     END AS exp,
     nxt.Name AS act
WITH cid,
     CASE WHEN act <> exp THEN 1 ELSE 0 END AS viol,
     1 AS cnt
WITH cid,
     sum(viol) AS violCnt,
     sum(cnt)  AS totRem
WITH cid,
     CASE WHEN totRem = 0
          THEN 1.0
          ELSE 1.0 - toFloat(violCnt)/totRem
     END AS score
MERGE (c:Case {id:cid})
MERGE (c)-[:HAS]->(s:Score {kind:'wrong_decision', caseId:cid})
SET   s.value     = score,
      s.updatedAt = datetime()
"""

# ═══════════════════════════════════════════════════════════════
# 3 ░  INIT HELPER
# ═══════════════════════════════════════════════════════════════
def _ensure_init(sess: Session) -> None:
    if not sess.run("MATCH (:InitFlag {name:$n}) RETURN 1 LIMIT 1",
                    n=_SENTINEL).single():
        for q in INDEX_QUERIES:
            sess.run(q)
        sess.run("CREATE (:InitFlag {name:$n})", n=_SENTINEL)
    sess.run(SCORE_UPDATE_QUERY)

# ═══════════════════════════════════════════════════════════════
# 4 ░  PUBLIC API
# ═══════════════════════════════════════════════════════════════
def run_wrong_decision(sess: Session) -> pd.DataFrame:
    _ensure_init(sess)
    return pd.DataFrame([r.data() for r in sess.run(WRONG_DECISION_QUERY)])

def get_wrong_decision_scores(sess: Session) -> pd.DataFrame:
    _ensure_init(sess)
    recs = sess.run("""
        MATCH (c:Case)-[:HAS]->(s:Score {kind:'wrong_decision'})
        RETURN c.id AS CaseId, s.value AS score ORDER BY c.id
    """)
    return pd.DataFrame([r.data() for r in recs])

def save_csv(df: pd.DataFrame,
             path: str | Path = "wrong_decision_metrics.csv") -> None:
    df.to_csv(path, index=False)

# ── demo ----------------------------------------------------------------------------
if __name__ == "__main__":
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    drv = GraphDatabase.driver(uri,
        auth=(os.getenv("NEO4J_USER","neo4j"), os.getenv("NEO4J_PWD","Mahaputra97")))
    with drv.session() as s:
        print(run_wrong_decision(s).head(), "\n")
        print(get_wrong_decision_scores(s).head())
    drv.close()