from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Concatenate, Self

from . import _factories as fc
from . import _nodes as nd
from . import _strategies as st
from ._core import IntoExpr, Kword, eq, ne


@dataclass(slots=True, repr=False)
class Query:
    node: nd.Node

    def _new[**P](
        self,
        factory: Callable[Concatenate[nd.Node, P], nd.Node],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        return self.__class__(factory(self.node, *args, **kwargs))

    def to_jmespath(self) -> str:
        return self.node.as_jmespath()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.node.as_jmespath() or Kword.CURRENT})"

    def __getattr__(self, name: str) -> Self:
        return self.field(name)

    def field(self, name: str) -> Self:
        return self._new(fc.subexpr, nd.Field, name)

    def index(self, i: int) -> Self:
        return self._new(fc.subexpr, nd.Index, i)

    def slice(
        self, start: int | None = None, end: int | None = None, step: int | None = None
    ) -> Self:
        return self._new(fc.subexpr, nd.Slice, start, end, step)

    def project(self, rhs: IntoExpr) -> Self:
        return self._new(fc.projection, rhs, st.array_project)

    def vproject(self, rhs: IntoExpr) -> Self:
        return self._new(fc.projection, rhs, st.object_project)

    def flatten(self) -> Self:
        return self._new(fc.unary, nd.Flatten)

    def filter(self, cond: Self, then: IntoExpr) -> Self:
        return self._new(fc.filter_project, then, cond.node)

    def eq(self, other: IntoExpr) -> Self:
        return self._new(fc.binary_op, nd.EqBase, other, eq)

    def ne(self, other: IntoExpr) -> Self:
        return self._new(fc.binary_op, nd.EqBase, other, ne)

    def lt(self, other: IntoExpr) -> Self:
        return self._new(fc.binary_op, nd.Comparator, other, operator.lt)

    def le(self, other: IntoExpr) -> Self:
        return self._new(fc.binary_op, nd.Comparator, other, operator.le)

    def gt(self, other: IntoExpr) -> Self:
        return self._new(fc.binary_op, nd.Comparator, other, operator.gt)

    def ge(self, other: IntoExpr) -> Self:
        return self._new(fc.binary_op, nd.Comparator, other, operator.ge)

    def and_(self, other: IntoExpr) -> Self:
        return self._new(fc.and_op, other)

    def or_(self, other: IntoExpr) -> Self:
        return self._new(fc.or_op, other)

    def not_(self) -> Self:
        return self._new(fc.unary, nd.Not)

    def length(self) -> Self:
        return self._new(fc.callable, st.length)

    def sort(self) -> Self:
        return self._new(fc.callable, st.sort)

    def keys(self) -> Self:
        return self._new(fc.callable, st.keys)

    def values(self) -> Self:
        return self._new(fc.callable, st.values)

    def to_array(self) -> Self:
        return self._new(fc.callable, st.to_array)

    def to_string(self) -> Self:
        return self._new(fc.callable, st.to_string)

    def to_number(self) -> Self:
        return self._new(fc.callable, st.to_number)

    def map_with(self, build: Callable[[Query], Query]) -> Self:
        return self._new(fc.map_apply, build)

    def sort_by(self, key: Callable[[Query], Query]) -> Self:
        return self._new(fc.key, "sort_by", st.sort_by, key)

    def min_by(self, key: Callable[[Query], Query]) -> Self:
        return self._new(fc.key, "min_by", st.min_by, key)

    def max_by(self, key: Callable[[Query], Query]) -> Self:
        return self._new(fc.key, "max_by", st.max_by, key)

    def pipe(self, rhs: Self) -> Self:
        return self._new(fc.pipe_op, rhs.node)

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
