from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ._nodes import Node
OP_MAP: dict[str, str] = {
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "lte": "<=",
    "gt": ">",
    "gte": ">=",
}

Comparator = Literal["eq", "ne", "lt", "lte", "gt", "gte"]


def as_arg(x: Node) -> str:
    s = x.as_jmespath()
    return "@" if s == "" else s


def as_expref(fnode: Node) -> str:
    s = fnode.as_jmespath()
    if s.startswith("."):
        s = "@" + s
    if s == "":
        s = "@"
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
