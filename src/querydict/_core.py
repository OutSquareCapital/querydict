from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ._checks import is_comparable, is_list, not_empty


class Kword(StrEnum):
    CURRENT = "@"
    REF = "&"
    DOT = "."
    ARRAY_PROJECT = "[*]"
    OBJECT_PROJECT = "*"
    FLATTEN = "[]"
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    AND = "&&"
    OR = "||"
    PIPE = "|"
    SPACE = " "


def as_ref(fnode: Node) -> str:
    s = fnode.as_jmespath()
    match s:
        case s if s.startswith(Kword.REF):
            return s
        case _:
            return Kword.REF + s or Kword.CURRENT


def _is_leading_dot(text: str) -> bool:
    return text.startswith((Kword.DOT, "[", "{", "(", "`", Kword.CURRENT) or text == "")


def ensure_leading_dot(text: str) -> str:
    match text:
        case text if _is_leading_dot(text):
            return text
        case _:
            return Kword.DOT + text


class Node(ABC):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.as_jmespath() or Kword.CURRENT})"

    @abstractmethod
    def eval(self) -> Callable[[Any], Any]:
        """Return a callable that evaluates the node against the given value."""
        raise NotImplementedError

    @abstractmethod
    def as_jmespath(self) -> str:
        """Convert the node to its JMESPath string representation."""
        raise NotImplementedError


@dataclass(slots=True, repr=False)
class ProjectionBase(Node):
    base: Node
    rhs: Node
    SEP: Kword

    @abstractmethod
    def _iter(self, value: Any) -> list[Any] | None:
        raise NotImplementedError

    def eval(self) -> Callable[[Any], list[Any] | None]:
        base_eval = self.base.eval()
        rhs_eval = self.rhs.eval()

        def _eval(value: Any) -> list[Any] | None:
            seq = self._iter(base_eval(value))
            if seq is None:
                return None
            return [rhs_eval(el) for el in seq if el is not None]

        return _eval

    def as_jmespath(self) -> str:
        return f"{self.base.as_jmespath()}{self.SEP}{ensure_leading_dot(self.rhs.as_jmespath())}"


@dataclass(slots=True, repr=False)
class BinaryNode(Node):
    left: Node
    right: Node

    @property
    def _kword(self) -> str:
        return self.__class__.__name__.upper()

    def as_jmespath(self) -> str:
        return (
            f"{self.left.as_jmespath()} {Kword[self._kword]} {self.right.as_jmespath()}"
        )


@dataclass(slots=True, repr=False)
class BinaryOp(BinaryNode):
    OP: Callable[[Any, Any], bool]

    @property
    def _kword(self) -> str:
        return self.OP.__name__.upper()


@dataclass(slots=True, repr=False)
class EqBase(BinaryOp):
    def eval(self) -> Callable[[Any], bool]:
        left_eval = self.left.eval()
        right_eval = self.right.eval()
        op = self.OP

        def _eval(value: Any) -> bool:
            return op(left_eval(value), right_eval(value))

        return _eval


@dataclass(slots=True, repr=False)
class Comparator(BinaryOp):
    def eval(self) -> Callable[[Any], bool | None]:
        left_eval = self.left.eval()
        right_eval = self.right.eval()
        op = self.OP

        def _eval(value: Any) -> bool | None:
            left = left_eval(value)
            right = right_eval(value)
            if not (is_comparable(left) and is_comparable(right)):
                return None
            return op(left, right)

        return _eval


@dataclass(slots=True, repr=False)
class AssociativeNode(BinaryNode):
    def eval(self) -> Callable[[Any], Any]:
        left_eval = self.left.eval()
        right_eval = self.right.eval()

        def _eval(value: Any) -> Any:
            left_val = left_eval(value)
            return left_val if not_empty(left_val) else right_eval(value)

        return _eval


@dataclass(slots=True, repr=False)
class CallableNode(Node):
    inner: Node

    @property
    def func(self) -> str:
        return self.__class__.__name__.lower()

    def as_jmespath(self) -> str:
        return f"{self.func}({self.inner.as_jmespath() or Kword.CURRENT})"


@dataclass(slots=True, repr=False)
class Converter(CallableNode):
    @property
    def func(self) -> str:
        return "to_" + self.__class__.__name__.lower()


@dataclass(slots=True, repr=False)
class Identity(Node):
    def eval(self) -> Callable[[Any], Any]:
        def _eval(value: Any) -> Any:
            return value

        return _eval

    def as_jmespath(self) -> str:
        return ""


@dataclass(slots=True, repr=False)
class KeyNode(Node):
    base: Node
    key_of: Callable[[Identity], Node]
    key_name: str

    def _key_func(self) -> Callable[[Any], Any]:
        key_eval = self.key_of(Identity()).eval()
        return lambda el: key_eval(el)

    def _eval_impl(self, arr: list[Any], key_fn: Callable[[Any], Any]) -> Any: ...

    def eval(self) -> Callable[[Any], Any]:
        base_eval = self.base.eval()
        key_fn = self._key_func()

        def _eval(value: Any) -> Any:
            arr = base_eval(value)
            if not is_list(arr) or not arr:
                return arr if is_list(arr) else None
            return self._eval_impl(arr, key_fn)

        return _eval

    def as_jmespath(self) -> str:
        return f"{self.key_name}({self.base.as_jmespath()}, {as_ref(self.key_of(Identity()))})"
