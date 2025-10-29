from __future__ import annotations

from typing import Any

from . import _strategies as st
from ._core import Node
from ._query import Query


def identity() -> Query:
    return Query(Node(st.identity()))


def field(name: str) -> Query:
    return Query(Node(st.identity())).field(name)


def lit(value: Any) -> Query:
    return Query(Node(st.literal(value)))


def select_list(*exprs: Query) -> Query:
    return Query(Node(st.multi_list(tuple(e.node.eval for e in exprs))))


def select_dict(**items: Query) -> Query:
    return Query(Node(st.multi_dict(tuple((k, v.node.eval) for k, v in items.items()))))
