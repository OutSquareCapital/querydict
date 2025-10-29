from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from . import _nodes as nd
from ._core import IntoExpr, Kword

if TYPE_CHECKING:
    from ._main import Query


def into_expr(obj: IntoExpr) -> nd.Node:
    from ._main import Query, field

    match obj:
        case nd.Node():
            return obj
        case Query():
            return obj.node
        case str():
            return field(obj).node
        case _:
            return nd.LiteralExpr(obj)


def binary_op(
    node: nd.Node,
    sort: type[nd.BinaryOp],
    right: IntoExpr,
    op: Callable[[Any, Any], bool],
) -> nd.BinaryOp:
    return sort(node, into_expr(right), op)


def binary(
    node: nd.Node, op: Callable[[nd.Node, nd.Node], nd.BinaryNode], right: IntoExpr
) -> nd.BinaryNode:
    return op(node, into_expr(right))


def unary(node: nd.Node, op: Callable[[nd.Node], nd.Node]) -> nd.Node:
    return op(node)


def callable(node: nd.Node, func: Callable[[Any], Any]) -> nd.CallableNode:
    return nd.CallableNode(node, func)


def projection(
    node: nd.Node, rhs: IntoExpr, func: Callable[[Any], list[Any] | None]
) -> nd.ProjectionBase:
    return nd.ProjectionBase(node, into_expr(rhs), Kword[func.__name__.upper()], func)


def filter_project(base: nd.Node, then: IntoExpr, cond: nd.Node) -> nd.FilterProjection:
    return nd.FilterProjection(base, into_expr(then), cond)


def key(
    node: nd.Node,
    name: str,
    func: Callable[[list[Any], Any], Any],
    key: Callable[[Query], Query],
) -> nd.KeyNode:
    from ._main import identity

    def _b(_: nd.Identity) -> nd.Node:
        return key(identity()).node

    return nd.KeyNode(node, _b, name, func)


def subexpr[**P](
    node: nd.Node, factory: Callable[P, nd.Node], *args: P.args, **kwargs: P.kwargs
) -> nd.SubExpr:
    if isinstance(node, nd.SubExpr):
        parts = node.parts + (factory(*args, **kwargs),)

    else:
        parts = (node, factory(*args, **kwargs))
    return nd.SubExpr(parts)


def and_op(left: nd.Node, right: IntoExpr) -> nd.And:
    return nd.And(left, into_expr(right))


def or_op(left: nd.Node, right: IntoExpr) -> nd.Or:
    return nd.Or(left, into_expr(right))


def pipe_op(left: nd.Node, right: nd.Node) -> nd.Pipe:
    return nd.Pipe(left, right)


def map_apply(base: nd.Node, build: Callable[[Query], Query]) -> nd.MapApply:
    from ._main import Query

    def _b(_: nd.Identity) -> nd.Node:
        return build(Query(nd.Identity())).node

    return nd.MapApply(base, _b)
