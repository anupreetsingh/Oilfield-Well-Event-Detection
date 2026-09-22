"""BM25 search over report remarks, filterable by wellbore, date range and NPT.

Lexical rather than embedding search: remarks are short and dense with rig jargon and
abbreviations (POOH, IBOP, WOW, LCM) that general-purpose embeddings blur together, and BM25
needs no model download on the cluster.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sqlalchemy import Engine, text

TOKEN = re.compile(r"[a-z0-9]+(?:/[0-9]+)?")


def tokenize(s: str) -> list[str]:
    return TOKEN.findall((s or "").lower())


class RemarkIndex:
    def __init__(self, remarks: pd.DataFrame):
        self.docs = remarks.reset_index(drop=True)
        self.bm25 = BM25Okapi([tokenize(t) for t in self.docs["comments"]]) if len(self.docs) else None

    @classmethod
    def from_engine(cls, engine: Engine) -> RemarkIndex:
        sql = text("SELECT activity_id, wellbore, report_date, code, is_npt, duration_h, comments "
                   "FROM activities WHERE comments IS NOT NULL")
        with engine.connect() as con:
            return cls(pd.read_sql(sql, con))

    def search(self, query: str, wellbore: str | None = None, start_date: str | None = None,
               end_date: str | None = None, npt_only: bool = False, k: int = 6) -> list[dict]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        mask = np.ones(len(self.docs), dtype=bool)
        if wellbore:
            mask &= (self.docs["wellbore"] == wellbore).to_numpy()
        if start_date:
            mask &= (self.docs["report_date"] >= start_date).to_numpy()
        if end_date:
            mask &= (self.docs["report_date"] <= end_date).to_numpy()
        if npt_only:
            mask &= self.docs["is_npt"].astype(bool).to_numpy()
        scores = np.where(mask & (scores > 0), scores, -np.inf)
        top = [i for i in np.argsort(-scores)[:k] if np.isfinite(scores[i])]
        hits = self.docs.iloc[top].assign(score=scores[top].round(3))
        return hits.to_dict("records")
