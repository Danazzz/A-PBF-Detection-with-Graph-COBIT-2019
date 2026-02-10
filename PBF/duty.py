# PBF/wrong_duty.py – detektor “Wrong Duty”  (v 2025-06-24 r4)
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from neo4j import GraphDatabase, Session

# ════════════════════════════════════════════════════════════
# 1 ░ GRID-SEARCH  +  θ Perusahaan   (SATU query)
# ════════════════════════════════════════════════════════════
WRONG_DUTY_QUERY = r"""
/* ░ EVENT BER-ROLE EXPLISIT */
MATCH (e:Anomaly)
WHERE e.employee_role <> '-'

/* ░ CEK IZIN Role→Activity (case-insensitive) */
OPTIONAL MATCH (ro:Role)-[:MAY_EXECUTE]->(ca:CaseActivity {Name:e.Name})
WHERE toUpper(ro.name) = toUpper(e.employee_role)
WITH e,
     CASE WHEN ca IS NULL THEN 1 ELSE 0 END AS viol          /* 1 = ilegal */

/* ░ SKOR PER CASE */
WITH e.CaseId AS cid,
     head(collect(e.case_Status)) AS trueStatus,
     sum(viol)  AS violCnt,
     count(*)   AS totEvt
WITH cid, trueStatus,
     CASE WHEN totEvt = 0
          THEN 1.0
          ELSE 1.0 - toFloat(violCnt)/totEvt
     END AS score                                           /* 0–1 */
WITH collect({cid:cid,status:trueStatus,score:score}) AS cases

/* ░ DAFTAR θ GRID + θ PERUSAHAAN */
WITH cases,
     [0,0.25,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99,1]     AS gridθ
OPTIONAL MATCH (rp:RiskParam {metric:'wrong_duty'})
WITH cases, gridθ,
     coalesce(toFloat(rp.theta),0.75)                            AS corpθ

/* ░ UNION θ → Label Source */
UNWIND (gridθ + [corpθ]) AS θ
WITH cases, θ,
     CASE WHEN abs(θ - corpθ) < 1e-9 THEN 'corporate' ELSE 'grid' END AS Source
UNWIND cases AS c

/* ░ KLASIFIKASI & CONFUSION MATRIX */
WITH Source, θ AS Threshold,
     c.status AS trueStatus,
     CASE WHEN c.score < θ THEN 'wrong_duty' ELSE 'normal' END AS pred
WITH Source, Threshold,
     collect({t:trueStatus,p:pred}) AS res
WITH Source, Threshold,
     size([r IN res WHERE r.t='wrong_duty' AND r.p='wrong_duty']) AS TP,
     size([r IN res WHERE r.t<>'wrong_duty' AND r.p='normal'])    AS TN,
     size([r IN res WHERE r.t<>'wrong_duty' AND r.p='wrong_duty'])AS FP,
     size([r IN res WHERE r.t='wrong_duty'  AND r.p='normal'])    AS FN
WITH Source, Threshold, TP,TN,FP,FN,(TP+TN+FP+FN) AS Tot
RETURN Source, Threshold, TP,TN,FP,FN,
       round(toFloat(TP+TN)/Tot,3)                        AS Accuracy,
       round(CASE WHEN TP+FP=0 THEN 0 ELSE toFloat(TP)/(TP+FP) END,3) AS Precision,
       round(CASE WHEN TP+FN=0 THEN 0 ELSE toFloat(TP)/(TP+FN) END,3) AS Recall,
       round(CASE WHEN 2*TP+FP+FN=0 THEN 0 ELSE 2.0*TP/(2*TP+FP+FN) END,3) AS F1_Score
ORDER BY Source, Threshold;
"""

# ════════════════════════════════════════════════════════════
# 2 ░ SCORE per-CaseId (tak berubah)
# ════════════════════════════════════════════════════════════
DUTY_SCORE_QUERY = """
MATCH (e:Anomaly)
WHERE e.employee_role <> '-'
OPTIONAL MATCH (ro:Role)-[:MAY_EXECUTE]->(ca:CaseActivity {Name:e.Name})
WHERE toUpper(ro.name) = toUpper(e.employee_role)
WITH e,
     CASE WHEN ca IS NULL THEN 1 ELSE 0 END AS viol
WITH e.CaseId AS cid,
     sum(viol)  AS violCnt,
     count(*)   AS totEvt
WITH cid,
     CASE WHEN totEvt = 0
          THEN 1.0
          ELSE 1.0 - toFloat(violCnt)/totEvt
     END AS score
MERGE (c:Case {id:cid})
MERGE (c)-[:HAS]->(s:Score {kind:'wrong_duty', caseId:cid})
SET   s.value = score,
      s.updatedAt = datetime()
RETURN cid AS CaseId, score ORDER BY cid;
"""

# ════════════════════════════════════════════════════════════
# 3 ░ PYTHON WRAPPER
# ════════════════════════════════════════════════════════════
def run_wrong_duty(session: Session) -> pd.DataFrame:
    return pd.DataFrame([r.data() for r in session.run(WRONG_DUTY_QUERY)])

def get_wrong_duty_scores(session: Session) -> pd.DataFrame:
    return pd.DataFrame([r.data() for r in session.run(DUTY_SCORE_QUERY)])

def save_csv(df: pd.DataFrame,
             path: str | Path = "wrong_duty_metrics.csv") -> None:
    df.to_csv(path, index=False)

# ── demo singkat ────────────────────────────────────────────
if __name__ == "__main__":
    drv = GraphDatabase.driver("bolt://localhost:7687",
                               auth=("neo4j", "Mahaputra97"))
    with drv.session() as s:
        print(run_wrong_duty(s).head(), "\n")
        print(get_wrong_duty_scores(s).head())
    drv.close()