from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Concatenate, Self

from . import _funcs as fn
from . import _nodes as nd
from ._checks import eq, ne
from ._core import Comparator, EqBase, Kword
from ._factories import (
    into_expr,
    new_binary_op,
    new_callable,
    new_key,
    new_projection,
    new_subexpr,
    new_unary,
)


@dataclass(slots=True, repr=False)
class Query:
    node: nd.Node

    def _new[**P, T: nd.Node](
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
        return self._new(new_subexpr, nd.Field, name)

    def index(self, i: int) -> Self:
        return self._new(new_subexpr, nd.Index, i)

    def slice(
        self, start: int | None = None, end: int | None = None, step: int | None = None
    ) -> Self:
        return self._new(new_subexpr, nd.Slice, start, end, step)

    def project(self, rhs: Any) -> Self:
        return self._new(new_projection, rhs, fn.array_project)

    def vproject(self, rhs: Any) -> Self:
        return self._new(new_projection, rhs, fn.object_project)

    def flatten(self) -> Self:
        return self._new(new_unary, nd.Flatten)

    def filter(self, cond: Self, then: Any) -> Self:
        return self._new(nd.FilterProjection, into_expr(then), cond.node)

    def eq(self, other: Any) -> Self:
        return self._new(new_binary_op, EqBase, other, eq)

    def ne(self, other: Any) -> Self:
        return self._new(new_binary_op, EqBase, other, ne)

    def lt(self, other: Any) -> Self:
        return self._new(new_binary_op, Comparator, other, operator.lt)

    def le(self, other: Any) -> Self:
        return self._new(new_binary_op, Comparator, other, operator.le)

    def gt(self, other: Any) -> Self:
        return self._new(new_binary_op, Comparator, other, operator.gt)

    def ge(self, other: Any) -> Self:
        return self._new(new_binary_op, Comparator, other, operator.ge)

    def and_(self, other: Any) -> Self:
        return self._new(nd.And, other)

    def or_(self, other: Any) -> Self:
        return self._new(nd.Or, other)

    def not_(self) -> Self:
        return self._new(new_unary, nd.Not)

    def length(self) -> Self:
        return self._new(new_callable, fn.length)

    def sort(self) -> Self:
        return self._new(new_callable, fn.sort)

    def keys(self) -> Self:
        return self._new(new_callable, fn.keys)

    def values(self) -> Self:
        return self._new(new_callable, fn.values)

    def to_array(self) -> Self:
        return self._new(new_callable, fn.to_array)

    def to_string(self) -> Self:
        return self._new(new_callable, fn.to_string)

    def to_number(self) -> Self:
        return self._new(new_callable, fn.to_number)

    def map_with(self, build: Callable[[Query], Query]) -> Self:
        def _b(_: nd.Identity) -> nd.Node:
            return build(identity()).node

        return self._new(nd.MapApply, _b)

    def sort_by(self, key: Callable[[Query], Query]) -> Self:
        return self._new(new_key, "sort_by", fn.sort_by, key)

    def min_by(self, key: Callable[[Query], Query]) -> Self:
        return self._new(new_key, "min_by", fn.min_by, key)

    def max_by(self, key: Callable[[Query], Query]) -> Self:
        return self._new(new_key, "max_by", fn.max_by, key)

    def pipe(self, rhs: Self) -> Self:
        return self._new(nd.Pipe, rhs.node)

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
