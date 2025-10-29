from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cytoolz as cz

from . import _strategies as st
from ._core import EvalFunc, IntoExpr, Kword


class Node(ABC):
    @abstractmethod
    def eval(self) -> EvalFunc:
        """Return a callable that evaluates the node against the given value."""
        raise NotImplementedError


@dataclass(slots=True)
class Identity(Node):
    def eval(self) -> EvalFunc:
        return cz.functoolz.identity


@dataclass(slots=True)
class LiteralExpr(Node):
    value: IntoExpr

    def eval(self) -> EvalFunc:
        return lambda x: cz.functoolz.identity(self.value)


@dataclass(slots=True)
class Field(Node):
    name: str

    def eval(self) -> EvalFunc:
        return st.field(self.name)


@dataclass(slots=True)
class Index(Node):
    i: int

    def eval(self) -> EvalFunc:
        return st.index(self.i)


@dataclass(slots=True)
class Slice(Node):
    start: int | None
    end: int | None
    step: int | None

    def eval(self) -> Callable[[Any], list[Any] | None]:
        return st.slice_on(self.start, self.end, self.step)


@dataclass(slots=True)
class SubExpr(Node):
    parts: tuple[Node, ...]

    def eval(self) -> EvalFunc:
        return st.sub_expr(tuple(p.eval() for p in self.parts))


@dataclass(slots=True)
class MultiList(Node):
    items: tuple[Node, ...]

    def eval(self) -> EvalFunc:
        return st.multi_list(tuple(it.eval() for it in self.items))


@dataclass(slots=True)
class MultiDict(Node):
    mapping: tuple[tuple[str, Node], ...]

    def eval(self) -> EvalFunc:
        return st.multi_dict(tuple((k, n.eval()) for k, n in self.mapping))


@dataclass(slots=True)
class NodeWithBase(Node):
    base: EvalFunc


@dataclass(slots=True)
class ProjectionBase(NodeWithBase):
    rhs: EvalFunc
    separator: Kword
    iter_func: Callable[[Any], list[Any] | None]

    def eval(self) -> Callable[[Any], list[Any] | None]:
        return st.project_base(self.base, self.rhs, self.iter_func)


@dataclass(slots=True)
class FilterProjection(NodeWithBase):
    then: EvalFunc
    cond: EvalFunc

    def eval(self) -> Callable[[Any], list[Any] | None]:
        return st.filter_projection(self.base, self.then, self.cond)


@dataclass(slots=True)
class BinaryNode(NodeWithBase):
    right: EvalFunc

    @property
    def _kword(self) -> str:
        return self.__class__.__name__.upper()


@dataclass(slots=True)
class BinaryOp(BinaryNode):
    op: Callable[[Any, Any], bool]

    @property
    def _kword(self) -> str:
        return self.op.__name__.upper()


@dataclass(slots=True)
class EqBase(BinaryOp):
    def eval(self) -> Callable[[Any], bool]:
        return st.eq(self.base, self.right, self.op)


@dataclass(slots=True)
class Comparator(BinaryOp):
    def eval(self) -> Callable[[Any], bool | None]:
        return st.comparator(self.base, self.right, self.op)


@dataclass(slots=True)
class AssociativeNode(BinaryNode):
    def eval(self) -> EvalFunc:
        return st.associate(self.base, self.right)


@dataclass(slots=True)
class And(AssociativeNode):
    def eval(self) -> EvalFunc:
        return st.and_(self.base, self.right)


@dataclass(slots=True)
class Or(AssociativeNode):
    pass


@dataclass(slots=True)
class CallableNode(NodeWithBase):
    func: EvalFunc

    def eval(self) -> EvalFunc:
        return st.callable_node(self.base, self.func)


@dataclass(slots=True)
class KeyNode(NodeWithBase):
    key_of: Callable[[Identity], Node]
    key_name: str
    func: Callable[[list[Any], Any], Any]

    def eval(self) -> EvalFunc:
        return st.key_node(self.base, self.func, self.key_of(Identity()).eval())


@dataclass(slots=True)
class Pipe(NodeWithBase):
    right: EvalFunc

    def eval(self) -> EvalFunc:
        return st.pipe(self.base, self.right)


@dataclass(slots=True)
class Flatten(NodeWithBase):
    def eval(self) -> EvalFunc:
        return st.flatten(self.base)


@dataclass(slots=True)
class Not(NodeWithBase):
    def eval(self) -> Callable[[Any], bool]:
        return st.not_(self.base)


@dataclass(slots=True)
class MapApply(NodeWithBase):
    build: Callable[[Identity], Node]

    def eval(self) -> Callable[[Any], list[Any] | None]:
        return st.map_apply(self.base, self.build(Identity()).eval())
