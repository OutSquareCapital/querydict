from __future__ import annotations

from typing import Any

from ._nodes import Identity, LiteralExpr, MultiDict, MultiList
from ._query import Query


def identity() -> Query:
    return Query(Identity())


def field(name: str) -> Query:
    return Query(Identity()).field(name)


def lit(value: Any) -> Query:
    return Query(LiteralExpr(value))


def select_list(*exprs: Query) -> Query:
    return Query(MultiList(tuple(e.node for e in exprs)))


def select_dict(**items: Query) -> Query:
    return Query(MultiDict(tuple((k, v.node) for k, v in items.items())))
