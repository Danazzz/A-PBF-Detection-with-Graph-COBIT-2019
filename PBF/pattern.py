# PBF/wrong_pattern.py  –  detektor “Wrong Pattern” (2025-06-24 r3)
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from neo4j import GraphDatabase, Session

# ═════════════════════════════════════════════════════════════
# 1 ░  GRID-SEARCH + θ-PERUSAHAAN  (single query)
# ═════════════════════════════════════════════════════════════
WRONG_PATTERN_QUERY = """
/* 1 ── VARIAN SOP (tanpa Invisible_Task) */
CALL {
  MATCH pSOP = (:CaseActivity {Name:'INVOICE_AUDIT'})
               -[:SEQUENCE|XORSPLIT|XORJOIN*]->
               (:CaseActivity {Name:'INVISIBLE_TASK'})
  WITH [n IN nodes(pSOP) WHERE NOT n:Invisible_Task | n.Name] AS vp
  RETURN collect(DISTINCT vp) AS validPaths
}

/* 2 ── SEMUA CaseId + GT */
MATCH (a:Anomaly)
WITH validPaths,
     collect(DISTINCT a.CaseId)                             AS allCaseIds,
     collect(DISTINCT {cid:a.CaseId,status:a.case_Status})  AS labs
UNWIND allCaseIds AS cid
WITH validPaths, labs, cid,
     head([x IN labs WHERE x.cid = cid | x.status])         AS trueStatus

/* 3 ── JALUR AKTUAL */
OPTIONAL MATCH pAct = (:Anomaly {CaseId:cid,Name:'INVOICE_AUDIT'})-[:SEQUENCE*]->(:Anomaly)
WITH validPaths, cid, trueStatus,
     CASE WHEN pAct IS NULL THEN [] ELSE [n IN nodes(pAct)|n.Name] END AS actPath

/* 4 ── BEST Levenshtein-SIM */
UNWIND validPaths AS vp
WITH cid, trueStatus, actPath, vp,
     apoc.text.join(actPath,'>') AS pathA,
     apoc.text.join(vp,'>')      AS pathB
WITH cid, trueStatus,
     CASE WHEN size(actPath)=0
          THEN 0.0
          ELSE coalesce(
                 apoc.text.levenshteinSimilarity(pathA,pathB),
                 1.0 - toFloat(apoc.text.levenshteinDistance(pathA,pathB))
                       / ((size(actPath)+size(vp))/2.0) )
     END                                       AS sim
WITH cid, trueStatus, max(sim) AS bestSim      /* 0–1 */

/* 5 ── LIST θ  (grid + corporate) */
OPTIONAL MATCH (rp:RiskParam {metric:'wrong_pattern'})
WITH cid, trueStatus, bestSim,
     coalesce(rp.theta,0.75)                                   AS corpθ,
     [0,0.25,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99,1]   AS gridθ
UNWIND (gridθ + [corpθ]) AS θ

/* 6 ── PREDIKSI & LABEL SOURCE */
WITH θ                      AS Threshold,
     CASE WHEN abs(θ - corpθ) < 1e-9 THEN 'corporate' ELSE 'grid' END AS Source,
     cid, trueStatus, bestSim
WITH Source, Threshold,
     CASE WHEN bestSim < Threshold THEN 'wrong_pattern' ELSE 'normal' END AS prediction,
     trueStatus
WITH Source, Threshold,
     collect({t:trueStatus,p:prediction})                        AS res

/* 7 ── CONFUSION MATRIX */
WITH Source, Threshold,
     size([r IN res WHERE r.t='wrong_pattern' AND r.p='wrong_pattern']) AS TP,
     size([r IN res WHERE r.t<>'wrong_pattern' AND r.p='normal'])       AS TN,
     size([r IN res WHERE r.t<>'wrong_pattern' AND r.p='wrong_pattern'])AS FP,
     size([r IN res WHERE r.t='wrong_pattern' AND r.p='normal'])        AS FN
WITH Source, Threshold, TP,TN,FP,FN,(TP+TN+FP+FN) AS Tot
RETURN Source, Threshold,
       TP, TN, FP, FN,
       round(toFloat(TP+TN)/Tot,3)                                   AS Accuracy,
       round(CASE WHEN TP+FP=0 THEN 0 ELSE toFloat(TP)/(TP+FP) END,3)AS Precision,
       round(CASE WHEN TP+FN=0 THEN 0 ELSE toFloat(TP)/(TP+FN) END,3)AS Sensitivity,
       round(CASE WHEN 2*TP+FP+FN=0 THEN 0 ELSE 2.0*TP/(2*TP+FP+FN) END,3) AS F1_Score
ORDER BY Source, Threshold;
"""

# ═════════════════════════════════════════════════════════════
# 2 ░  INDEX / CONSTRAINT  +  SCORE REFRESH  (tetap)
# ═════════════════════════════════════════════════════════════
INDEX_QUERIES = [
    "CREATE INDEX anom_case IF NOT EXISTS FOR (a:Anomaly)      ON (a.CaseId)",
    "CREATE INDEX anom_name IF NOT EXISTS FOR (a:Anomaly)      ON (a.Name)",
    "CREATE INDEX ca_name   IF NOT EXISTS FOR (c:CaseActivity) ON (c.Name)",
    "CREATE CONSTRAINT score_unique IF NOT EXISTS FOR (s:Score) REQUIRE (s.kind, s.caseId) IS UNIQUE",
]
_SENTINEL = "wrongPattern_init_v2"

SCORE_UPDATE_QUERY = """
/* Varian SOP */
MATCH pSOP = (:CaseActivity {Name:'INVOICE_AUDIT'})
             -[:SEQUENCE|XORSPLIT|XORJOIN*]->
             (:CaseActivity {Name:'INVISIBLE_TASK'})
WITH collect([n IN nodes(pSOP) WHERE NOT n:Invisible_Task | n.Name]) AS vPaths

MATCH (a:Anomaly)
WITH DISTINCT a.CaseId AS cid, vPaths
OPTIONAL MATCH pAct = (:Anomaly {CaseId:cid, Name:'INVOICE_AUDIT'})-[:SEQUENCE*]->(:Anomaly)
WITH cid, vPaths,
     CASE WHEN pAct IS NULL THEN [] ELSE [n IN nodes(pAct)|n.Name] END AS actPath
UNWIND vPaths AS vp
WITH cid,
     CASE WHEN size(actPath)=0
          THEN 0.0
          ELSE coalesce(
                 apoc.text.levenshteinSimilarity(apoc.text.join(actPath,'>'),apoc.text.join(vp,'>')),
                 1.0 - toFloat(apoc.text.levenshteinDistance(apoc.text.join(actPath,'>'),apoc.text.join(vp,'>')))
                       / ((size(actPath)+size(vp))/2.0) )
     END AS sim
WITH cid, max(sim) AS bestSim
MERGE (c:Case {id:cid})
MERGE (c)-[:HAS]->(s:Score {kind:'wrong_pattern', caseId:cid})
SET   s.value = bestSim,
      s.updatedAt = datetime()
"""

# ═════════════════════════════════════════════════════════════
# 3 ░  INIT HELPER
# ═════════════════════════════════════════════════════════════
def _ensure_init(sess: Session) -> None:
    if not sess.run("MATCH (:InitFlag {name:$n}) RETURN 1 LIMIT 1",
                    n=_SENTINEL).single():
        for q in INDEX_QUERIES:
            sess.run(q)
        sess.run("CREATE (:InitFlag {name:$n})", n=_SENTINEL)
    sess.run(SCORE_UPDATE_QUERY)

# ═════════════════════════════════════════════════════════════
# 4 ░  API
# ═════════════════════════════════════════════════════════════
def run_wrong_pattern(sess: Session) -> pd.DataFrame:
    _ensure_init(sess)
    return pd.DataFrame([r.data() for r in sess.run(WRONG_PATTERN_QUERY)])

def get_wrong_pattern_scores(sess: Session) -> pd.DataFrame:
    _ensure_init(sess)
    recs = sess.run("""
        MATCH (c:Case)-[:HAS]->(s:Score {kind:'wrong_pattern'})
        RETURN c.id AS CaseId, s.value AS score ORDER BY c.id
    """)
    return pd.DataFrame([r.data() for r in recs])

# ═════════════════════════════════════════════════════════════
# 5 ░  Demo mandiri
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    drv = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER","neo4j"),
                                          os.getenv("NEO4J_PWD","Mahaputra97")))
    with drv.session() as s:
        print(run_wrong_pattern(s).head(), "\n")
        print(get_wrong_pattern_scores(s).head())
    drv.close()