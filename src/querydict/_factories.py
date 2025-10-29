from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from . import _nodes as nd
from ._core import Kword

if TYPE_CHECKING:
    from ._main import Query


def into_expr(obj: Any) -> nd.Node:
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
    node: nd.Node, sort: type[nd.BinaryOp], right: Any, op: Callable[[Any, Any], bool]
):
    return sort(node, into_expr(right), op)


def binary(
    node: nd.Node, op: Callable[[nd.Node, nd.Node], nd.BinaryNode], right: Any
) -> nd.BinaryNode:
    return op(node, into_expr(right))


def unary(node: nd.Node, op: Callable[[nd.Node], nd.Node]) -> nd.Node:
    return op(node)


def callable(node: nd.Node, func: Callable[[Any], Any]) -> nd.CallableNode:
    return nd.CallableNode(node, func)


def projection(
    node: nd.Node, rhs: Any, func: Callable[[Any], list[Any] | None]
) -> nd.ProjectionBase:
    return nd.ProjectionBase(node, into_expr(rhs), Kword[func.__name__.upper()], func)


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
