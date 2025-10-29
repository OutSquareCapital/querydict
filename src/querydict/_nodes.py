from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._core import (
    IntoExpr,
    Kword,
    is_comparable,
    is_list,
    is_mapping,
    is_number,
    not_empty,
)


class Node(ABC):
    @abstractmethod
    def eval(self) -> Callable[[Any], Any]:
        """Return a callable that evaluates the node against the given value."""
        raise NotImplementedError


@dataclass(slots=True)
class Identity(Node):
    def eval(self) -> Callable[[Any], Any]:
        def _eval(value: Any) -> Any:
            return value

        return _eval


@dataclass(slots=True)
class LiteralExpr(Node):
    value: IntoExpr

    def eval(self) -> Callable[[Any], Any]:
        val = self.value

        def _eval(value: Any) -> Any:
            return val

        return _eval


@dataclass(slots=True)
class Field(Node):
    name: str

    def eval(self) -> Callable[[Any], Any]:
        name = self.name

        def _eval(value: Any) -> Any:
            return value.get(name, None) if is_mapping(value) else None

        return _eval


@dataclass(slots=True)
class Index(Node):
    i: int

    def eval(self) -> Callable[[Any], Any]:
        idx = self.i

        def _eval(value: Any) -> Any:
            if not is_list(value):
                return None
            try:
                return value[idx]
            except IndexError:
                return None

        return _eval


@dataclass(slots=True)
class Slice(Node):
    start: int | None
    end: int | None
    step: int | None

    def eval(self) -> Callable[[Any], list[Any] | None]:
        slc = slice(self.start, self.end, self.step)

        def _eval(value: Any) -> list[Any] | None:
            return value[slc] if is_list(value) else None

        return _eval


@dataclass(slots=True)
class SubExpr(Node):
    parts: tuple[Node, ...]

    def eval(self) -> Callable[[Any], Any]:
        part_evals = tuple(p.eval() for p in self.parts)

        def _eval(value: Any) -> Any:
            out = value
            for p_eval in part_evals:
                out = p_eval(out)
            return out

        return _eval


@dataclass(slots=True)
class MultiList(Node):
    items: tuple[Node, ...]

    def eval(self) -> Callable[[Any], Any]:
        item_evals = tuple(it.eval() for it in self.items)

        def _eval(value: Any) -> Any:
            return [it_eval(value) for it_eval in item_evals]

        return _eval


@dataclass(slots=True)
class MultiDict(Node):
    mapping: tuple[tuple[str, Node], ...]

    def eval(self) -> Callable[[Any], Any]:
        mapping_evals = tuple((k, n.eval()) for k, n in self.mapping)

        def _eval(value: Any) -> Any:
            return {k: n_eval(value) for k, n_eval in mapping_evals}

        return _eval


@dataclass(slots=True)
class NodeWithBase(Node):
    base: Node


@dataclass(slots=True)
class ProjectionBase(NodeWithBase):
    rhs: Node
    separator: Kword
    iter_func: Callable[[Any], list[Any] | None]

    def eval(self) -> Callable[[Any], list[Any] | None]:
        base_eval = self.base.eval()
        rhs_eval = self.rhs.eval()

        def _eval(value: Any) -> list[Any] | None:
            seq = self.iter_func(base_eval(value))
            return (
                [rhs_eval(el) for el in seq if el is not None]
                if seq is not None
                else None
            )

        return _eval


@dataclass(slots=True)
class FilterProjection(NodeWithBase):
    then: Node
    cond: Node

    def eval(self) -> Callable[[Any], list[Any] | None]:
        base_eval = self.base.eval()
        then_eval = self.then.eval()
        cond_eval = self.cond.eval()

        def _eval(value: Any) -> list[Any] | None:
            seq = base_eval(value)
            return (
                [then_eval(el) for el in seq if not_empty(cond_eval(el))]
                if is_list(seq)
                else None
            )

        return _eval


@dataclass(slots=True)
class BinaryNode(NodeWithBase):
    right: Node

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
        left_eval = self.base.eval()
        right_eval = self.right.eval()
        op = self.op

        def _eval(value: Any) -> bool:
            return op(left_eval(value), right_eval(value))

        return _eval


@dataclass(slots=True)
class Comparator(BinaryOp):
    def eval(self) -> Callable[[Any], bool | None]:
        left_eval = self.base.eval()
        right_eval = self.right.eval()
        op = self.op

        def _eval(value: Any) -> bool | None:
            left = left_eval(value)
            right = right_eval(value)
            return (
                op(left, right)
                if (is_comparable(left) and is_comparable(right))
                else None
            )

        return _eval


@dataclass(slots=True)
class AssociativeNode(BinaryNode):
    def eval(self) -> Callable[[Any], Any]:
        left_eval = self.base.eval()
        right_eval = self.right.eval()

        def _eval(value: Any) -> Any:
            left_val = left_eval(value)
            return left_val if not_empty(left_val) else right_eval(value)

        return _eval


@dataclass(slots=True)
class And(AssociativeNode):
    def eval(self) -> Callable[[Any], Any]:
        left_eval = self.base.eval()
        right_eval = self.right.eval()

        def _eval(value: Any) -> Any:
            left_val = left_eval(value)
            return right_eval(value) if not_empty(left_val) else left_val

        return _eval


@dataclass(slots=True)
class Or(AssociativeNode):
    pass


@dataclass(slots=True)
class CallableNode(NodeWithBase):
    func: Callable[[Any], Any]

    def eval(self):
        evaluator = self.base.eval()

        def _eval(value: Any) -> Any:
            return self.func(evaluator(value))

        return _eval


@dataclass(slots=True)
class KeyNode(NodeWithBase):
    key_of: Callable[[Identity], Node]
    key_name: str
    func: Callable[[list[Any], Any], Any]

    def _key_func(self) -> Callable[[Any], Any]:
        key_eval = self.key_of(Identity()).eval()
        return lambda el: key_eval(el)

    def eval(self) -> Callable[[Any], Any]:
        base_eval = self.base.eval()
        key_fn = self._key_func()

        def _eval(value: Any) -> Any:
            arr = base_eval(value)
            return self.func(arr, key_fn) if is_list(arr) else None

        return _eval


@dataclass(slots=True)
class Pipe(NodeWithBase):
    right: Node

    def eval(self) -> Callable[[Any], Any]:
        left_eval = self.base.eval()
        right_eval = self.right.eval()

        def _eval(value: Any) -> Any:
            return right_eval(left_eval(value))

        return _eval


@dataclass(slots=True)
class Flatten(NodeWithBase):
    def eval(self) -> Callable[[Any], Any]:
        base_eval = self.base.eval()

        def _eval(value: Any) -> Any:
            seq = base_eval(value)
            if not is_list(seq):
                return None
            flat: list[Any] = []
            for el in seq:
                if is_list(el):
                    flat.extend(el)
                else:
                    flat.append(el)
            return flat

        return _eval


@dataclass(slots=True)
class Not(NodeWithBase):
    def eval(self) -> Callable[[Any], bool]:
        expr_eval = self.base.eval()

        def _eval(value: Any) -> bool:
            v = expr_eval(value)
            if is_number(v) and v == 0:
                return True
            return not not_empty(v)

        return _eval


@dataclass(slots=True)
class MapApply(NodeWithBase):
    build: Callable[[Identity], Node]

    def eval(self) -> Callable[[Any], list[Any] | None]:
        base_eval = self.base.eval()
        build_eval = self.build(Identity()).eval()

        def _eval(value: Any) -> list[Any] | None:
            arr = base_eval(value)
            return [build_eval(el) for el in arr] if is_list(arr) else None

        return _eval
