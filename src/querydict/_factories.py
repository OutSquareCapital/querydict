from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from . import _nodes as nd
from ._core import EvalFunc, IntoExpr, Kword

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


def field(node: nd.Node, name: str):
    return subexpr(node, nd.Field, name)


def index(node: nd.Node, i: int):
    return subexpr(node, nd.Index, i)


def slice(
    node: nd.Node,
    start: int | None = None,
    end: int | None = None,
    step: int | None = None,
) -> nd.SubExpr:
    return subexpr(node, nd.Slice, start, end, step)


def flatten(node: nd.Node) -> nd.Flatten:
    return nd.Flatten(node.eval())


def not_(node: nd.Node) -> nd.Not:
    return nd.Not(node.eval())


def binary_op(
    node: nd.Node,
    sort: type[nd.BinaryOp],
    right: IntoExpr,
    op: Callable[[Any, Any], bool],
) -> nd.BinaryOp:
    return sort(node.eval(), into_expr(right).eval(), op)


def equality(
    node: nd.Node, right: IntoExpr, op: Callable[[Any, Any], bool]
) -> nd.BinaryOp:
    return binary_op(node, nd.EqBase, right, op)


def comparator(
    node: nd.Node, right: IntoExpr, op: Callable[[Any, Any], bool]
) -> nd.BinaryOp:
    return binary_op(node, nd.Comparator, right, op)


def binary(
    node: nd.Node, op: Callable[[nd.Node, nd.Node], nd.BinaryNode], right: IntoExpr
) -> nd.BinaryNode:
    return op(node, into_expr(right))


def unary(node: nd.Node, op: Callable[[nd.Node], nd.Node]) -> nd.Node:
    return op(node)


def callable(node: nd.Node, func: EvalFunc) -> nd.CallableNode:
    return nd.CallableNode(node.eval(), func)


def projection(
    node: nd.Node, rhs: IntoExpr, func: Callable[[Any], list[Any] | None]
) -> nd.ProjectionBase:
    return nd.ProjectionBase(
        node.eval(), into_expr(rhs).eval(), Kword[func.__name__.upper()], func
    )


def filter_project(base: nd.Node, then: IntoExpr, cond: nd.Node) -> nd.FilterProjection:
    return nd.FilterProjection(base.eval(), into_expr(then).eval(), cond.eval())


def key(
    node: nd.Node,
    name: str,
    func: Callable[[list[Any], Any], Any],
    key: Callable[[Query], Query],
) -> nd.KeyNode:
    from ._main import identity

    def _b(_: nd.Identity) -> nd.Node:
        return key(identity()).node

    return nd.KeyNode(node.eval(), _b, name, func)


def subexpr[**P](
    node: nd.Node, factory: Callable[P, nd.Node], *args: P.args, **kwargs: P.kwargs
) -> nd.SubExpr:
    if isinstance(node, nd.SubExpr):
        parts = node.parts + (factory(*args, **kwargs),)

    else:
        parts = (node, factory(*args, **kwargs))
    return nd.SubExpr(parts)


def and_op(left: nd.Node, right: IntoExpr) -> nd.And:
    return nd.And(left.eval(), into_expr(right).eval())


def or_op(left: nd.Node, right: IntoExpr) -> nd.Or:
    return nd.Or(left.eval(), into_expr(right).eval())


def pipe_op(left: nd.Node, right: nd.Node) -> nd.Pipe:
    return nd.Pipe(left.eval(), right.eval())


def map_apply(base: nd.Node, build: Callable[[Query], Query]) -> nd.MapApply:
    from ._main import Query

    def _b(_: nd.Identity) -> nd.Node:
        return build(Query(nd.Identity())).node

    return nd.MapApply(base.eval(), _b)
