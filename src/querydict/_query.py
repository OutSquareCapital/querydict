from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Concatenate, Self

from . import _strategies as st
from ._core import IntoExpr, Node, eq, ne


def into_node(obj: IntoExpr) -> Node:
    from ._main import field
    from ._query import Query

    match obj:
        case Query():
            return obj.node
        case str():
            return field(obj).node
        case _:
            return st.literal(obj)


def build_key_fn(key: Callable[[Query], Query]) -> Node:
    from ._main import identity

    return key(identity()).node


@dataclass(slots=True)
class Query:
    node: Node

    def _new[**P](
        self,
        factory: Callable[Concatenate[Node, P], Node],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        return self.__class__(factory(self.node, *args, **kwargs))

    def __getattr__(self, name: str) -> Self:
        return self.field(name)

    def field(self, name: str) -> Self:
        return self._new(st.sub_expr, st.field(name))

    def index(self, i: int) -> Self:
        return self._new(st.sub_expr, st.index(i))

    def slice(
        self, start: int | None = None, end: int | None = None, step: int | None = None
    ) -> Self:
        return self._new(st.sub_expr, st.slice_on(start, end, step))

    def project(self, rhs: IntoExpr) -> Self:
        return self._new(st.project_base, into_node(rhs), st.array_project)

    def vproject(self, rhs: IntoExpr) -> Self:
        return self._new(st.project_base, into_node(rhs), st.object_project)

    def flatten(self) -> Self:
        return self._new(st.flatten)

    def filter(self, cond: Self, then: IntoExpr) -> Self:
        return self._new(st.filter_projection, into_node(then), cond.node)

    def eq(self, other: IntoExpr) -> Self:
        return self._new(st.eq, into_node(other), eq)

    def ne(self, other: IntoExpr) -> Self:
        return self._new(st.eq, into_node(other), ne)

    def lt(self, other: IntoExpr) -> Self:
        return self._new(st.comparator, into_node(other), operator.lt)

    def le(self, other: IntoExpr) -> Self:
        return self._new(st.comparator, into_node(other), operator.le)

    def gt(self, other: IntoExpr) -> Self:
        return self._new(st.comparator, into_node(other), operator.gt)

    def ge(self, other: IntoExpr) -> Self:
        return self._new(st.comparator, into_node(other), operator.ge)

    def and_(self, other: IntoExpr) -> Self:
        return self._new(st.and_, into_node(other))

    def or_(self, other: IntoExpr) -> Self:
        return self._new(st.associate, into_node(other))

    def not_(self) -> Self:
        return self._new(st.not_)

    def length(self) -> Self:
        return self._new(st.callable_node, st.length)

    def sort(self) -> Self:
        return self._new(st.callable_node, st.sort)

    def keys(self) -> Self:
        return self._new(st.callable_node, st.keys)

    def values(self) -> Self:
        return self._new(st.callable_node, st.values)

    def to_array(self) -> Self:
        return self._new(st.callable_node, st.to_array)

    def to_string(self) -> Self:
        return self._new(st.callable_node, st.to_string)

    def to_number(self) -> Self:
        return self._new(st.callable_node, st.to_number)

    def map_with(self, build: Callable[[Query], Query]) -> Self:
        return self._new(st.map_apply, build_key_fn(build))

    def sort_by(self, key: Callable[[Query], Query]) -> Self:
        return self._new(st.key_node, st.sort_by, build_key_fn(key))

    def min_by(self, key: Callable[[Query], Query]) -> Self:
        return self._new(st.key_node, st.min_by, build_key_fn(key))

    def max_by(self, key: Callable[[Query], Query]) -> Self:
        return self._new(st.key_node, st.max_by, build_key_fn(key))

    def pipe(self, rhs: Self) -> Self:
        return self._new(st.pipe, rhs.node)

    def search(self, data: Any) -> Any:
        return self.node(data)
