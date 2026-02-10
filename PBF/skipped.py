# PBF/skipped.py – detektor “Skipped Activity”  (v 2025-06-24 r5)
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from neo4j import GraphDatabase, Session

# ═══════════════════════════════════════════════════════════════
# 1 ░  QUERY GRID + CORPORATE θ (single query)
# ═══════════════════════════════════════════════════════════════
SKIPPED_ACTIVITY_QUERY = """
/* ░ 1. VARIAN SOP (tanpa Invisible_Task) */
CALL {
  MATCH pSOP = (:CaseActivity {Name:'INVOICE_AUDIT'})
               -[:SEQUENCE|XORSPLIT|XORJOIN*]->
               (:CaseActivity {Name:'INVISIBLE_TASK'})
  WITH [n IN nodes(pSOP) WHERE NOT n:Invisible_Task | n.Name] AS vp
  RETURN collect(DISTINCT vp) AS validPaths
}

/* ░ 2. BEST-SIMILARITY PER CASE */
MATCH (a:Anomaly)
WITH validPaths,
     collect(DISTINCT a.CaseId)                                AS allCaseIds,
     collect(DISTINCT {cid:a.CaseId, status:a.case_Status})    AS statusMap
UNWIND allCaseIds AS cid
WITH validPaths, cid,
     head([m IN statusMap WHERE m.cid = cid | m.status])       AS trueStatus
OPTIONAL MATCH pAct = (:Anomaly {CaseId:cid, Name:'INVOICE_AUDIT'})-[:SEQUENCE*]->(:Anomaly)
WITH validPaths, cid, trueStatus,
     CASE WHEN pAct IS NULL THEN [] ELSE [n IN nodes(pAct)|n.Name] END AS actPath
UNWIND validPaths AS vp
WITH cid, trueStatus,
     size(apoc.coll.intersection(actPath,vp)) AS inter,
     size(apoc.coll.union       (actPath,vp)) AS uni
WITH cid, trueStatus,
     max(CASE WHEN uni = 0 THEN 0.0 ELSE toFloat(inter)/uni END) AS bestSim   /* 0-1 */
WITH collect({cid:cid, status:trueStatus, sim:bestSim})          AS cases

/* ░ 3. SIAPKAN θ LIST + θ CORPORATE */
WITH cases,
     [0,0.25,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99,1]                   AS gridθ
OPTIONAL MATCH (rp:RiskParam {metric:'skipped_activity'})
WITH cases, gridθ, coalesce( toFloat(rp.theta), 0.75 )          AS corpθ

/* ░ 4. UNWIND θ & HITUNG CONFUSION-MATRIX */
UNWIND (gridθ + [corpθ]) AS θ
WITH cases, θ,
     CASE WHEN abs(θ - corpθ) < 1e-9 THEN 'corporate' ELSE 'grid' END AS Source
UNWIND cases AS c
WITH θ AS Threshold, Source,
     c.status                               AS trueStatus,
     CASE WHEN c.sim < θ THEN 'skipped' ELSE 'normal' END AS prediction
WITH Threshold, Source,
     collect({t:trueStatus,p:prediction})                  AS res
WITH Threshold, Source,
     size([r IN res WHERE r.t='skipped'  AND r.p='skipped']) AS TP,
     size([r IN res WHERE r.t<>'skipped' AND r.p='normal'])  AS TN,
     size([r IN res WHERE r.t<>'skipped' AND r.p='skipped']) AS FP,
     size([r IN res WHERE r.t='skipped'  AND r.p='normal'])  AS FN
WITH Source, Threshold, TP, TN, FP, FN,
     (TP+TN+FP+FN)                                           AS Total
RETURN Source, Threshold,
       TP, TN, FP, FN,
       round(toFloat(TP+TN)/Total,3)                         AS Accuracy,
       round(CASE WHEN TP+FP=0 THEN 0 ELSE toFloat(TP)/(TP+FP) END,3) AS Precision,
       round(CASE WHEN TP+FN=0 THEN 0 ELSE toFloat(TP)/(TP+FN) END,3) AS Sensitivity,
       round(CASE WHEN 2*TP+FP+FN=0 THEN 0 ELSE 2.0*TP/(2*TP+FP+FN) END,3) AS F1_Score
ORDER BY Source, Threshold;
"""

# ═══════════════════════════════════════════════════════════════
# 2 ░  SCORE PER CASE  – best Jaccard (untuk Streamlit)
# ═══════════════════════════════════════════════════════════════
SCORE_UPDATE_QUERY = """
/* 1. Varian SOP */
MATCH pSOP = (:CaseActivity {Name:'INVOICE_AUDIT'})
             -[:SEQUENCE|XORSPLIT|XORJOIN*]->
             (:CaseActivity {Name:'INVISIBLE_TASK'})
WITH collect([n IN nodes(pSOP) WHERE NOT n:Invisible_Task | n.Name]) AS vPaths

/* 2. Best Jaccard per CaseId */
MATCH (a:Anomaly)
WITH DISTINCT a.CaseId AS cid, vPaths
OPTIONAL MATCH pAct = (:Anomaly {CaseId:cid, Name:'INVOICE_AUDIT'})-[:SEQUENCE*]->(:Anomaly)
WITH cid, vPaths,
     CASE WHEN pAct IS NULL THEN [] ELSE [n IN nodes(pAct)|n.Name] END AS actPath
UNWIND vPaths AS vp
WITH cid,
     size(apoc.coll.intersection(actPath,vp)) AS inter,
     size(apoc.coll.union       (actPath,vp)) AS uni
WITH cid, max(CASE WHEN uni=0 THEN 0.0 ELSE toFloat(inter)/uni END) AS bestSim

/* 3. Simpan */
MERGE (c:Case {id:cid})
MERGE (c)-[:HAS]->(s:Score {kind:'skipped', caseId:cid})
SET   s.value     = bestSim,
      s.updatedAt = datetime()
"""

# ═══════════════════════════════════════════════════════════════
# 3 ░  HELPER INIT
# ═══════════════════════════════════════════════════════════════
INDEX_QUERIES = [
  "CREATE CONSTRAINT case_id_unique IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE",
  "CREATE INDEX anom_case IF NOT EXISTS FOR (a:Anomaly) ON (a.CaseId)",
  "CREATE INDEX anom_name IF NOT EXISTS FOR (a:Anomaly) ON (a.Name)",
  "CREATE INDEX ca_name   IF NOT EXISTS FOR (c:CaseActivity) ON (c.Name)",
  "CREATE CONSTRAINT score_unique IF NOT EXISTS FOR (s:Score) REQUIRE (s.caseId,s.kind) IS UNIQUE"
]
_SENTINEL = "skipped_init_v3"

def ensure_init(sess: Session):
    if not sess.run("MATCH (:InitFlag {name:$n}) RETURN 1 LIMIT 1", n=_SENTINEL).single():
        for q in INDEX_QUERIES: sess.run(q)
        sess.run("CREATE (:InitFlag {name:$n})", n=_SENTINEL)
    sess.run(SCORE_UPDATE_QUERY)

# ═══════════════════════════════════════════════════════════════
# 4 ░  API
# ═══════════════════════════════════════════════════════════════
def run_skipped_activity(sess: Session) -> pd.DataFrame:
    ensure_init(sess)
    return pd.DataFrame([r.data() for r in sess.run(SKIPPED_ACTIVITY_QUERY)])

def get_skipped_scores(sess: Session) -> pd.DataFrame:
    ensure_init(sess)
    q = ("MATCH (c:Case)-[:HAS]->(s:Score {kind:'skipped'}) "
         "RETURN c.id AS CaseId, s.value AS score ORDER BY c.id")
    return pd.DataFrame([r.data() for r in sess.run(q)])

def save_csv(df: pd.DataFrame,
             path: str | Path = "skipped_activity_metrics.csv") -> None:
    df.to_csv(path, index=False)

# ═══════════════════════════════════════════════════════════════
# 5 ░  Demo mandiri
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    drv = GraphDatabase.driver(uri, auth=("neo4j", os.getenv("NEO4J_PWD","neo4j")))
    with drv.session() as s:
        print(run_skipped_activity(s).head(), "\n")
        print(get_skipped_scores(s).head())
    drv.close()