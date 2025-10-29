from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from . import _strategies as st
from ._core import EvalFunc, IntoExpr, Node

if TYPE_CHECKING:
    from ._query import Query


def into_expr(obj: IntoExpr) -> EvalFunc:
    from ._main import field
    from ._query import Query

    match obj:
        case Query():
            return obj.node.eval
        case str():
            return field(obj).node.eval
        case _:
            return st.literal(obj)


def field(node: Node, name: str) -> Node:
    return Node(st.sub_expr((node.eval, st.field(name))))


def index(node: Node, i: int) -> Node:
    return Node(st.sub_expr((node.eval, st.index(i))))


def slice(
    node: Node,
    start: int | None = None,
    end: int | None = None,
    step: int | None = None,
) -> Node:
    return Node(st.sub_expr((node.eval, st.slice_on(start, end, step))))


def flatten(node: Node) -> Node:
    return Node(st.flatten(node.eval))


def not_(node: Node) -> Node:
    return Node(st.not_(node.eval))


def equality(node: Node, right: IntoExpr, op: Callable[[Any, Any], bool]) -> Node:
    return Node(st.eq(node.eval, into_expr(right), op))


def comparator(node: Node, right: IntoExpr, op: Callable[[Any, Any], bool]) -> Node:
    return Node(st.comparator(node.eval, into_expr(right), op))


def callable(node: Node, func: EvalFunc) -> Node:
    return Node(st.callable_node(node.eval, func))


def projection(
    node: Node, rhs: IntoExpr, func: Callable[[Any], list[Any] | None]
) -> Node:
    return Node(st.project_base(node.eval, into_expr(rhs), func))


def filter_project(base: Node, then: IntoExpr, cond: Query) -> Node:
    return Node(st.filter_projection(base.eval, into_expr(then), cond.node.eval))


def key(
    node: Node,
    func: Callable[[list[Any], Any], Any],
    key: Callable[[Query], Query],
) -> Node:
    from ._main import identity

    return Node(st.key_node(node.eval, func, key(identity()).node.eval))


def and_op(left: Node, right: IntoExpr) -> Node:
    return Node(st.and_(left.eval, into_expr(right)))


def or_op(left: Node, right: IntoExpr) -> Node:
    return Node(st.associate(left.eval, into_expr(right)))


def pipe_op(left: Node, right: Query) -> Node:
    return Node(st.pipe(left.eval, right.node.eval))


def map_apply(base: Node, build: Callable[[Query], Query]) -> Node:
    from ._main import identity

    return Node(st.map_apply(base.eval, build(identity()).node.eval))
