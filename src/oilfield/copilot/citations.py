"""Check that a drafted summary only cites activity IDs the tools actually returned."""

from __future__ import annotations

import re
from dataclasses import dataclass

ACTIVITY_ID = re.compile(r"[A-Z0-9][A-Z0-9-]*:\d{8}:\d{2}")
BRACKETS = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class CitationCheck:
    cited: list[str]
    unknown: list[str]

    @property
    def ok(self) -> bool:
        return not self.unknown


def cited_ids(markdown: str) -> list[str]:
    ids = []
    for group in BRACKETS.findall(markdown):
        ids += ACTIVITY_ID.findall(group)
    return list(dict.fromkeys(ids))


def check_citations(markdown: str, seen: set[str]) -> CitationCheck:
    cited = cited_ids(markdown)
    return CitationCheck(cited=cited, unknown=[i for i in cited if i not in seen])
