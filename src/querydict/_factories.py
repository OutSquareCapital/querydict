from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ._core import (
    BinaryNode,
    BinaryOp,
    CallableNode,
    KeyNode,
    Kword,
    Node,
    ProjectionBase,
)
from ._nodes import Identity, LiteralExpr, SubExpr

if TYPE_CHECKING:
    from ._main import Query


def into_expr(obj: Any) -> Node:
    from ._main import Query, field

    match obj:
        case Node():
            return obj
        case Query():
            return obj.node
        case str():
            return field(obj).node
        case _:
            return LiteralExpr(obj)


def new_binary_op(
    node: Node, sort: type[BinaryOp], right: Any, op: Callable[[Any, Any], bool]
):
    return sort(node, into_expr(right), op)


def new_binary(
    node: Node, op: Callable[[Node, Node], BinaryNode], right: Any
) -> BinaryNode:
    return op(node, into_expr(right))


def new_unary(node: Node, op: Callable[[Node], Node]) -> Node:
    return op(node)


def new_callable(node: Node, func: Callable[[Any], Any]) -> CallableNode:
    return CallableNode(node, func)


def new_projection(
    node: Node, rhs: Any, func: Callable[[Any], list[Any] | None]
) -> ProjectionBase:
    return ProjectionBase(node, into_expr(rhs), Kword[func.__name__.upper()], func)


def new_key(
    node: Node,
    name: str,
    func: Callable[[list[Any], Any], Any],
    key: Callable[[Query], Query],
) -> KeyNode:
    from ._main import identity

    def _b(_: Identity) -> Node:
        return key(identity()).node

    return KeyNode(node, _b, name, func)


def new_subexpr[**P](
    node: Node, factory: Callable[P, Node], *args: P.args, **kwargs: P.kwargs
) -> SubExpr:
    if isinstance(node, SubExpr):
        parts = node.parts + (factory(*args, **kwargs),)

    else:
        parts = (node, factory(*args, **kwargs))
    return SubExpr(parts)
