from __future__ import annotations

from typing import Any

from . import _strategies as st
from ._query import Query


def identity() -> Query:
    return Query(st.identity())


def field(name: str) -> Query:
    return Query(st.identity()).field(name)


def lit(value: Any) -> Query:
    return Query(st.literal(value))


def select_list(*exprs: Query) -> Query:
    return Query(st.multi_list(tuple(e.node for e in exprs)))


def select_dict(**items: Query) -> Query:
    return Query(st.multi_dict(tuple((k, v.node) for k, v in items.items())))
