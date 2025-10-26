from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

IDENTITY = "@"


def as_arg(x: Node) -> str:
    s = x.as_jmespath()
    return IDENTITY if s == "" else s


def as_expref(fnode: Node) -> str:
    s = fnode.as_jmespath()
    if s.startswith("."):
        s = IDENTITY + s
    if s == "":
        s = IDENTITY
    return f"&{s}"


def ensure_leading_dot(s: str) -> str:
    if not s:
        return s
    if s[0] in (".", "[", "{", "(", "`", "@"):
        return s
    return "." + s


def as_callable(name: str, *nodes: Node) -> str:
    args = ", ".join(as_arg(n) for n in nodes)
    return f"{name}({args})"


class Node(ABC):
    def __repr__(self) -> str:
        jp = self.as_jmespath() or "@"
        return f"{self.__class__.__name__}({jp})"

    @abstractmethod
    def eval(self, value: Any) -> Any:
        """Evaluate the node against the given value."""
        raise NotImplementedError

    @abstractmethod
    def as_jmespath(self) -> str:
        """Convert the node to its JMESPath string representation."""
        raise NotImplementedError
