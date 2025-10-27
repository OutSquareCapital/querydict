from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._checks import equals, falsy, is_comparable

CURRENT = "@"


def as_ref(fnode: Node) -> str:
    s = fnode.as_jmespath()
    match s:
        case s if s.startswith("&"):
            return s
        case s if s == "":
            return f"&{CURRENT}"
        case _:
            return f"&{s}"


def _is_leading_dot(text: str) -> bool:
    return text.startswith((".", "[", "{", "(", "`", CURRENT) or text == "")


def ensure_leading_dot(text: str) -> str:
    match text:
        case text if _is_leading_dot(text):
            return text
        case _:
            return "." + text


class Node(ABC):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.as_jmespath() or CURRENT})"

    @abstractmethod
    def eval(self, value: Any) -> Any:
        """Evaluate the node against the given value."""
        raise NotImplementedError

    @abstractmethod
    def as_jmespath(self) -> str:
        """Convert the node to its JMESPath string representation."""
        raise NotImplementedError


@dataclass(slots=True, repr=False)
class ProjectionBase(Node):
    base: Node
    rhs: Node
    SEP: str

    @abstractmethod
    def _iter(self, value: Any) -> list[Any] | None:
        raise NotImplementedError

    def eval(self, value: Any) -> list[Any] | None:
        seq = self._iter(self.base.eval(value))
        if seq is None:
            return None
        return [self.rhs.eval(el) for el in seq if el is not None]

    def as_jmespath(self) -> str:
        return f"{self.base.as_jmespath()}{self.SEP}{ensure_leading_dot(self.rhs.as_jmespath())}"


@dataclass(slots=True, repr=False)
class BinaryCompare(Node):
    left: Node
    right: Node
    SYMBOL: str

    def as_jmespath(self) -> str:
        return f"{self.left.as_jmespath()} {self.SYMBOL} {self.right.as_jmespath()}"


@dataclass(slots=True, repr=False)
class EqBase(BinaryCompare):
    def _equals(self, value: Any) -> bool:
        return equals(self.left.eval(value), self.right.eval(value))


@dataclass(slots=True, repr=False)
class OrderBase(BinaryCompare):
    OP: Callable[[Any, Any], bool] = operator.lt

    def eval(self, value: Any) -> bool | None:
        left = self.left.eval(value)
        right = self.right.eval(value)
        if not (is_comparable(left) and is_comparable(right)):
            return None
        return self.OP(left, right)


@dataclass(slots=True, repr=False)
class AssociativeNode(Node):
    left: Node
    right: Node
    OP: str

    def eval(self, value: Any) -> Any:
        left = self.left.eval(value)
        return left if falsy(left) else self.right.eval(value)

    def as_jmespath(self) -> str:
        return f"{self.left.as_jmespath()} {self.OP} {self.right.as_jmespath()}"


@dataclass(slots=True, repr=False)
class CallableNode(Node):
    inner: Node

    @property
    def func(self) -> str:
        return self.__class__.__name__.lower()

    def as_jmespath(self) -> str:
        return f"{self.func}({self.inner.as_jmespath() or CURRENT})"


@dataclass(slots=True, repr=False)
class Converter(CallableNode):
    @property
    def func(self) -> str:
        return "to_" + self.__class__.__name__.lower()
