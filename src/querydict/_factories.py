from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from . import _strategies as st
from ._core import IntoExpr, Node

if TYPE_CHECKING:
    from ._query import Query


def into_expr(obj: IntoExpr) -> Node:
    from ._main import field
    from ._query import Query

    match obj:
        case Query():
            return obj.node
        case str():
            return field(obj).node
        case _:
            return st.literal(obj)


def field(node: Node, name: str) -> Node:
    return st.sub_expr((node, st.field(name)))


def index(node: Node, i: int) -> Node:
    return st.sub_expr((node, st.index(i)))


def slice(
    node: Node,
    start: int | None = None,
    end: int | None = None,
    step: int | None = None,
) -> Node:
    return st.sub_expr((node, st.slice_on(start, end, step)))


def flatten(node: Node) -> Node:
    return st.flatten(node)


def not_(node: Node) -> Node:
    return st.not_(node)


def equality(node: Node, right: IntoExpr, op: Callable[[Any, Any], bool]) -> Node:
    return st.eq(node, into_expr(right), op)


def comparator(node: Node, right: IntoExpr, op: Callable[[Any, Any], bool]) -> Node:
    return st.comparator(node, into_expr(right), op)


def callable(node: Node, func: Node) -> Node:
    return st.callable_node(node, func)


def projection(
    node: Node, rhs: IntoExpr, func: Callable[[Any], list[Any] | None]
) -> Node:
    return st.project_base(node, into_expr(rhs), func)


def filter_project(base: Node, then: IntoExpr, cond: Query) -> Node:
    return st.filter_projection(base, into_expr(then), cond.node)


def key(
    node: Node,
    func: Callable[[list[Any], Any], Any],
    key: Callable[[Query], Query],
) -> Node:
    from ._main import identity

    return st.key_node(node, func, key(identity()).node)


def and_op(left: Node, right: IntoExpr) -> Node:
    return st.and_(left, into_expr(right))


def or_op(left: Node, right: IntoExpr) -> Node:
    return st.associate(left, into_expr(right))


def pipe_op(left: Node, right: Query) -> Node:
    return st.pipe(left, right.node)


def map_apply(base: Node, build: Callable[[Query], Query]) -> Node:
    from ._main import identity

    return st.map_apply(base, build(identity()).node)
