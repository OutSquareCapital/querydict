from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

from . import _nodes as nd
from ._checks import eq, ne
from ._core import BinaryNode, BinaryOp, Comparator, EqBase, Kword, Node

type KeyOp = Callable[[Query], Query]


def into_expr(obj: Any) -> nd.Node:
    match obj:
        case Node():
            return obj
        case Query():
            return obj.node
        case str():
            return field(obj).node
        case _:
            return nd.LiteralExpr(obj)


@dataclass(slots=True, repr=False)
class Query:
    node: nd.Node

    def _new(self, node: nd.Node) -> Self:
        return self.__class__(node)

    def _new_binary_op(
        self, sort: type[BinaryOp], right: Any, op: Callable[[Any, Any], bool]
    ) -> Self:
        return self._new(sort(self.node, into_expr(right), op))

    def _new_binary(
        self, op: Callable[[nd.Node, nd.Node], BinaryNode], right: Any
    ) -> Self:
        return self._new(op(self.node, into_expr(right)))

    def _new_unary(self, op: Callable[[nd.Node], nd.Node]) -> Self:
        return self._new(op(self.node))

    def _new_key(
        self,
        node: Callable[[nd.Node, Callable[[nd.Identity], nd.Node]], nd.KeyNode],
        key: KeyOp,
    ) -> Self:
        def _b(_: nd.Identity) -> nd.Node:
            return key(identity()).node

        return self._new(node(self.node, _b))

    def to_jmespath(self) -> str:
        return self.node.as_jmespath()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.node.as_jmespath() or Kword.CURRENT})"

    def __getattr__(self, name: str) -> Self:
        return self.field(name)

    def field(self, name: str) -> Self:
        if isinstance(self.node, nd.SubExpr):
            node = nd.SubExpr(self.node.parts + (nd.Field(name),))
        else:
            node = nd.SubExpr((self.node, nd.Field(name)))
        return self._new(node)

    def index(self, i: int) -> Self:
        if isinstance(self.node, nd.SubExpr):
            node = nd.SubExpr(self.node.parts + (nd.Index(i),))
        else:
            node = nd.SubExpr((self.node, nd.Index(i)))
        return self._new(node)

    def slice(
        self, start: int | None = None, end: int | None = None, step: int | None = None
    ) -> Self:
        if isinstance(self.node, nd.SubExpr):
            node = nd.SubExpr(self.node.parts + (nd.Slice(start, end, step),))
        else:
            node = nd.SubExpr((self.node, nd.Slice(start, end, step)))
        return self._new(node)

    def project(self, rhs: Any) -> Self:
        return self._new(nd.ArrayProject(self.node, into_expr(rhs)))

    def vproject(self, rhs: Any) -> Self:
        return self._new(nd.ObjectProject(self.node, into_expr(rhs)))

    def flatten(self) -> Self:
        return self._new_unary(nd.Flatten)

    def filter(self, cond: Self, then: Any) -> Self:
        return self._new(nd.FilterProjection(self.node, into_expr(then), cond.node))

    def eq(self, other: Any) -> Self:
        return self._new_binary_op(EqBase, other, eq)

    def ne(self, other: Any) -> Self:
        return self._new_binary_op(EqBase, other, ne)

    def lt(self, other: Any) -> Self:
        return self._new_binary_op(Comparator, other, operator.lt)

    def le(self, other: Any) -> Self:
        return self._new_binary_op(Comparator, other, operator.le)

    def gt(self, other: Any) -> Self:
        return self._new_binary_op(Comparator, other, operator.gt)

    def ge(self, other: Any) -> Self:
        return self._new_binary_op(Comparator, other, operator.ge)

    def and_(self, other: Any) -> Self:
        return self._new_binary(nd.And, other)

    def or_(self, other: Any) -> Self:
        return self._new_binary(nd.Or, other)

    def not_(self) -> Self:
        return self._new_unary(nd.Not)

    def length(self) -> Self:
        return self._new_unary(nd.Length)

    def sort(self) -> Self:
        return self._new_unary(nd.Sort)

    def keys(self) -> Self:
        return self._new_unary(nd.Keys)

    def values(self) -> Self:
        return self._new_unary(nd.Values)

    def to_array(self) -> Self:
        return self._new_unary(nd.Array)

    def to_string(self) -> Self:
        return self._new_unary(nd.String)

    def to_number(self) -> Self:
        return self._new_unary(nd.Number)

    def map_with(self, build: KeyOp) -> Self:
        def _b(_: nd.Identity) -> nd.Node:
            return build(identity()).node

        return self._new(nd.MapApply(self.node, _b))

    def sort_by(self, key: KeyOp) -> Self:
        return self._new_key(nd.SortBy, key)

    def min_by(self, key: KeyOp) -> Self:
        return self._new_key(nd.MinBy, key)

    def max_by(self, key: KeyOp) -> Self:
        return self._new_key(nd.MaxBy, key)

    def pipe(self, rhs: Self) -> Self:
        return self._new(nd.Pipe(self.node, rhs.node))

    def search(self, data: Any) -> Any:
        eval_func = self.node.eval()
        return eval_func(data)


def identity() -> Query:
    return Query(nd.Identity())


def field(name: str) -> Query:
    return Query(nd.Identity()).field(name)


def lit(value: Any) -> Query:
    return Query(nd.LiteralExpr(value))


def select_list(*exprs: Query) -> Query:
    return Query(nd.MultiList(tuple(e.node for e in exprs)))


def select_dict(**items: Query) -> Query:
    return Query(nd.MultiDict(tuple((k, v.node) for k, v in items.items())))
