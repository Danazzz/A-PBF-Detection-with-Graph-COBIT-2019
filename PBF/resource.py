# PBF/wrong_resource.py  ·  detektor “Wrong Resource”   (v 2025-06-24 r2)
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from neo4j import GraphDatabase, Session


# ════════════════════════════════════════════════════════════
# 1 ░ GRID-SEARCH  +  θ-PERUSAHAAN  (satu query)
# ════════════════════════════════════════════════════════════
WRONG_RESOURCE_QUERY = """
/* 1 ── <aktivitas, expected_role> dari graf SOP */
CALL {
  MATCH (s:Activity)
  WHERE s.employee_role IS NOT NULL
  RETURN collect(DISTINCT {act:s.Name, role:s.employee_role}) AS expRoles
}

/* 2 ── daftar kasus + label GT */
MATCH (a:Anomaly)
WITH expRoles,
     collect(DISTINCT a.CaseId)                             AS allCaseIds,
     collect(DISTINCT {cid:a.CaseId, status:a.case_Status}) AS statusList

/* 3 ── iterasi CaseId */
UNWIND allCaseIds AS case_id
WITH expRoles, case_id,
     head([x IN statusList WHERE x.cid = case_id | x.status]) AS trueStatus

/* 4 ── kumpulkan event {act,role} */
MATCH (ev:Anomaly {CaseId:case_id})
WITH expRoles, case_id, trueStatus,
     collect({act:ev.Name, role:ev.employee_role}) AS evs

/* 5 ── bestSim per case */
WITH expRoles, case_id, trueStatus,
     size(evs) AS totalActs,
     size([e IN evs WHERE
          coalesce(head([er IN expRoles WHERE er.act=e.act | er.role]),
                   '__NF__') <> e.role])         AS wrongCnt
WITH case_id, trueStatus,
     CASE WHEN totalActs=0
          THEN 0.0
          ELSE 1.0 - toFloat(wrongCnt)/totalActs
     END AS bestSim                          /* 0-1 */

/* 6 ── UNWIND daftar θ (grid + corporate) */
OPTIONAL MATCH (rp:RiskParam {metric:'wrong_resource'})
WITH case_id, trueStatus, bestSim,
     [0,0.25,0.50,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99,1] AS gridθ,
     coalesce(toFloat(rp.theta),0.75)                        AS corpθ
UNWIND (gridθ + [corpθ]) AS threshold
WITH threshold, case_id, trueStatus, bestSim,
     CASE WHEN abs(threshold - corpθ) < 1e-9
          THEN 'corporate' ELSE 'grid' END                  AS Source,
     CASE WHEN bestSim < threshold
          THEN 'wrong_resource' ELSE 'normal' END           AS prediction

/* 7 ── confusion matrix & metrik */
WITH Source, threshold, collect({t:trueStatus,p:prediction}) AS res
WITH Source, threshold,
     size([r IN res WHERE r.t='wrong_resource' AND r.p='wrong_resource']) AS TP,
     size([r IN res WHERE r.t<>'wrong_resource' AND r.p='normal'])        AS TN,
     size([r IN res WHERE r.t<>'wrong_resource' AND r.p='wrong_resource'])AS FP,
     size([r IN res WHERE r.t='wrong_resource'  AND r.p='normal'])        AS FN
WITH Source, threshold AS Threshold, TP,TN,FP,FN,
     (TP+TN+FP+FN)                                         AS Tot
RETURN Source, Threshold,
       TP, TN, FP, FN,
       round(toFloat(TP+TN)/Tot,3)                         AS Accuracy,
       round(CASE WHEN TP+FP=0 THEN 0 ELSE toFloat(TP)/(TP+FP) END,3) AS Precision,
       round(CASE WHEN TP+FN=0 THEN 0 ELSE toFloat(TP)/(TP+FN) END,3) AS Sensitivity,
       round(CASE WHEN 2*TP+FP+FN=0 THEN 0 ELSE 2.0*TP/(2*TP+FP+FN) END,3) AS F1_Score
ORDER BY Source, Threshold;
"""

# ════════════════════════════════════════════════════════════
# 2 ░ SCORE per-CaseId (UI & PyVis)
# ════════════════════════════════════════════════════════════
RESOURCE_SCORE_QUERY = """
MATCH (e:Anomaly) WHERE e.employee_role <> '-'
OPTIONAL MATCH (ro:Role)-[:MAY_EXECUTE]->(ca:CaseActivity {Name:e.Name})
WHERE toUpper(ro.name)=toUpper(e.employee_role)
WITH e, CASE WHEN ca IS NULL THEN 1 ELSE 0 END AS viol
WITH e.CaseId AS cid, sum(viol) AS violCnt, count(*) AS totEvt
WITH cid,
     CASE WHEN totEvt=0 THEN 1.0
          ELSE 1.0 - toFloat(violCnt)/totEvt END AS score
MERGE (c:Case {id:cid})
MERGE (c)-[:HAS]->(s:Score {kind:'wrong_resource', caseId:cid})
SET   s.value = score,
      s.updatedAt = datetime()
RETURN cid AS CaseId, score ORDER BY cid;
"""

# ════════════════════════════════════════════════════════════
# 3 ░ WRAPPER PYTHON
# ════════════════════════════════════════════════════════════
def run_wrong_resource(session: Session) -> pd.DataFrame:
    return pd.DataFrame([r.data() for r in session.run(WRONG_RESOURCE_QUERY)])

def get_wrong_resource_scores(session: Session) -> pd.DataFrame:
    return pd.DataFrame([r.data() for r in session.run(RESOURCE_SCORE_QUERY)])

def save_csv(df: pd.DataFrame,
             path: str | Path = "wrong_resource_metrics.csv") -> None:
    df.to_csv(path, index=False)

# ── demo singkat ────────────────────────────────────────────
if __name__ == "__main__":
    drv = GraphDatabase.driver("bolt://localhost:7687",
                               auth=("neo4j", "Mahaputra97"))
    with drv.session() as s:
        print(run_wrong_resource(s).head(), "\n")
        print(get_wrong_resource_scores(s).head())
    drv.close()