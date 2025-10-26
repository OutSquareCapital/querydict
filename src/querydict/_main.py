from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

from . import _nodes as nd


@dataclass(slots=True)
class Query:
    node: nd.Node

    def _new(self, node: nd.Node) -> Self:
        return self.__class__(node)

    def field(self, name: str) -> Self:
        if isinstance(self.node, nd.SubExpr):
            node = nd.SubExpr(self.node.parts + (nd.Field(name),))
        else:
            node = nd.SubExpr((self.node, nd.Field(name)))
        return self._new(node)

    def dot(self, rhs: Self) -> Self:
        if isinstance(self.node, nd.SubExpr):
            node = nd.SubExpr(self.node.parts + (rhs.node,))
        else:
            node = nd.SubExpr((self.node, rhs.node))
        return self._new(node)

    def index(self, i: int) -> Self:
        return self._new(nd.SubExpr((self.node, nd.Index(i))))

    def slice(
        self, start: int | None = None, end: int | None = None, step: int | None = None
    ) -> Self:
        return self._new(nd.SubExpr((self.node, nd.Slice(start, end, step))))

    def project(self, rhs: Self) -> Self:
        return self._new(nd.Projection(self.node, rhs.node))

    def vproject(self, rhs: Self) -> Self:
        return self._new(nd.ValueProjection(self.node, rhs.node))

    def flatten(self) -> Self:
        return self._new(nd.Flatten(self.node))

    def filter(self, cond: Self, then: Self | None = None) -> Self:
        return self._new(
            nd.FilterProjection(self.node, (then or identity()).node, cond.node)
        )

    def eq(self, other: Self) -> Self:
        return self._new(nd.Compare("eq", self.node, other.node))

    def ne(self, other: Self) -> Self:
        return self._new(nd.Compare("ne", self.node, other.node))

    def lt(self, other: Self) -> Self:
        return self._new(nd.Compare("lt", self.node, other.node))

    def lte(self, other: Self) -> Self:
        return self._new(nd.Compare("lte", self.node, other.node))

    def gt(self, other: Self) -> Self:
        return self._new(nd.Compare("gt", self.node, other.node))

    def gte(self, other: Self) -> Self:
        return self._new(nd.Compare("gte", self.node, other.node))

    def and_(self, other: Self) -> Self:
        return self._new(nd.And(self.node, other.node))

    def or_(self, other: Self) -> Self:
        return self._new(nd.Or(self.node, other.node))

    def not_(self) -> Self:
        return self._new(nd.Not(self.node))

    def length(self) -> Self:
        return self._new(nd.Length(self.node))

    def sort(self) -> Self:
        return self._new(nd.Sort(self.node))

    def keys(self) -> Self:
        return self._new(nd.Keys(self.node))

    def values(self) -> Self:
        return self._new(nd.Values(self.node))

    def to_array(self) -> Self:
        return self._new(nd.ToArray(self.node))

    def to_string(self) -> Self:
        return self._new(nd.ToString(self.node))

    def to_number(self) -> Self:
        return self._new(nd.ToNumber(self.node))

    def map_with(self, build: Callable[[Query], Query]) -> Self:
        def _b(_: nd.Identity) -> nd.Node:
            return build(identity()).node

        return self._new(nd.MapApply(self.node, _b))

    def sort_by(self, key: Callable[[Query], Query]) -> Self:
        def _b(_: nd.Identity) -> nd.Node:
            return key(identity()).node

        return self._new(nd.SortBy(self.node, _b))

    def min_by(self, key: Callable[[Query], Query]) -> Self:
        def _b(_: nd.Identity) -> nd.Node:
            return key(identity()).node

        return self._new(nd.MinBy(self.node, _b))

    def max_by(self, key: Callable[[Query], Query]) -> Self:
        def _b(_: nd.Identity) -> nd.Node:
            return key(identity()).node

        return self._new(nd.MaxBy(self.node, _b))

    def pipe(self, rhs: Self) -> Self:
        return self._new(nd.Pipe(self.node, rhs.node))

    def search(self, data: Any) -> Any:
        return self.node.eval(data)


def identity() -> Query:
    return Query(nd.Identity())


def lit(value: Any) -> Query:
    return Query(nd.LiteralExpr(value))


def select_list(*exprs: Query) -> Query:
    return Query(nd.MultiList(tuple(e.node for e in exprs)))


def select_dict(**items: Query) -> Query:
    return Query(nd.MultiDict(tuple((k, v.node) for k, v in items.items())))
