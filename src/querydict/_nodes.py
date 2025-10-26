from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeIs

# ------------------------------ Helpers -------------------------------------


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_mapping(x: Any) -> TypeIs[Mapping[Any, Any]]:
    return isinstance(x, Mapping)


def _is_list(x: Any) -> TypeIs[list[Any]]:
    return isinstance(x, list)


def _comparable(x: Any) -> bool:
    return _is_number(x) or isinstance(x, str)


def _equals(x: Any, y: Any) -> bool:
    # Special-case 0/1 vs bool like original jmespath semantics
    if _is_number(x) and x in (0, 1):
        return not isinstance(y, bool)
    if _is_number(y) and y in (0, 1):
        return not isinstance(x, bool)
    return x == y


def _falsy(v: Any) -> bool:
    return v in ("", [], {}) or v is None or v is False


def _truthy(v: Any) -> bool:
    return not _falsy(v)


# ------------------------------- Nodes --------------------------------------


class Node:
    def eval(self, value: Any) -> Any:  # abstract
        raise NotImplementedError


# ---- leaves ----


@dataclass(slots=True)
class Identity(Node):
    def eval(self, value: Any) -> Any:
        return value


@dataclass(slots=True)
class LiteralExpr(Node):
    value: Any

    def eval(self, value: Any) -> Any:
        return self.value


@dataclass(slots=True)
class Field(Node):
    name: str

    def eval(self, value: Any) -> Any:
        if _is_mapping(value):
            return value.get(self.name)
        return None


@dataclass(slots=True)
class Index(Node):
    i: int

    def eval(self, value: Any) -> Any:
        if not _is_list(value):
            return None
        try:
            return value[self.i]
        except IndexError:
            return None


@dataclass(slots=True)
class Slice(Node):
    start: int | None
    end: int | None
    step: int | None

    def eval(self, value: Any) -> Any:
        if not _is_list(value):
            return None
        return value[slice(self.start, self.end, self.step)]


# ---- structural ----


@dataclass(slots=True)
class SubExpr(Node):
    parts: tuple[Node, ...]

    def eval(self, value: Any) -> Any:
        out = value
        for p in self.parts:
            out = p.eval(out)
        return out


@dataclass(slots=True)
class Pipe(Node):
    left: Node
    right: Node

    def eval(self, value: Any) -> Any:
        return self.right.eval(self.left.eval(value))


# ---- projections ----


@dataclass(slots=True)
class Projection(Node):
    base: Node
    rhs: Node

    def eval(self, value: Any) -> Any:
        seq = self.base.eval(value)
        if not _is_list(seq):
            return None
        out: list[Any] = []
        for el in seq:
            v = self.rhs.eval(el)
            if v is not None:
                out.append(v)
        return out


@dataclass(slots=True)
class ValueProjection(Node):
    base: Node
    rhs: Node

    def eval(self, value: Any) -> Any:
        obj = self.base.eval(value)
        if not _is_mapping(obj):
            return None
        out: list[Any] = []
        for el in obj.values():
            v = self.rhs.eval(el)
            if v is not None:
                out.append(v)
        return out


@dataclass(slots=True)
class Flatten(Node):
    base: Node

    def eval(self, value: Any) -> Any:
        seq = self.base.eval(value)
        if not _is_list(seq):
            return None
        flat: list[Any] = []
        for el in seq:
            if _is_list(el):
                flat.extend(el)
            else:
                flat.append(el)
        return flat


@dataclass(slots=True)
class FilterProjection(Node):
    base: Node
    then: Node
    cond: Node

    def eval(self, value: Any) -> Any:
        seq = self.base.eval(value)
        if not _is_list(seq):
            return None
        out: list[Any] = []
        for el in seq:
            keep = self.cond.eval(el)
            if _truthy(keep):
                v = self.then.eval(el)
                if v is not None:
                    out.append(v)
        return out


# ---- logic / compare ----

Comparator = Literal["eq", "ne", "lt", "lte", "gt", "gte"]


@dataclass(slots=True)
class Compare(Node):
    op: Comparator
    left: Node
    right: Node

    def eval(self, value: Any) -> Any:
        l = self.left.eval(value)
        r = self.right.eval(value)
        if self.op in ("eq", "ne"):
            eq = _equals(l, r)
            return eq if self.op == "eq" else (not eq)
        if not (_comparable(l) and _comparable(r)):
            return None
        if self.op == "lt":
            return l < r
        if self.op == "lte":
            return l <= r
        if self.op == "gt":
            return l > r
        if self.op == "gte":
            return l >= r
        raise ValueError(self.op)


@dataclass(slots=True)
class And(Node):
    left: Node
    right: Node

    def eval(self, value: Any) -> Any:
        l = self.left.eval(value)
        return l if _falsy(l) else self.right.eval(value)


@dataclass(slots=True)
class Or(Node):
    left: Node
    right: Node

    def eval(self, value: Any) -> Any:
        l = self.left.eval(value)
        return self.right.eval(value) if _falsy(l) else l


@dataclass(slots=True)
class Not(Node):
    expr: Node

    def eval(self, value: Any) -> Any:
        v = self.expr.eval(value)
        if _is_number(v) and v == 0:
            return False
        return not v


# ---- built-in ops as nodes (no registry) ----


@dataclass(slots=True)
class Length(Node):
    inner: Node

    def eval(self, value: Any) -> Any:
        x = self.inner.eval(value)
        if isinstance(x, (str, list, dict)):
            return len(x)
        return None


@dataclass(slots=True)
class Sort(Node):
    inner: Node

    def eval(self, value: Any) -> Any:
        xs = self.inner.eval(value)
        if isinstance(xs, list):
            try:
                return sorted(xs)
            except Exception:
                return xs
        return None


@dataclass(slots=True)
class Keys(Node):
    inner: Node

    def eval(self, value: Mapping[Any, Any] | Any) -> Any:
        x = self.inner.eval(value)
        return list(x.keys()) if _is_mapping(x) else None


@dataclass(slots=True)
class Values(Node):
    inner: Node

    def eval(self, value: Any) -> Any:
        x = self.inner.eval(value)
        return list(x.values()) if _is_mapping(x) else None


@dataclass(slots=True)
class ToArray(Node):
    inner: Node

    def eval(self, value: Any) -> Any:
        x = self.inner.eval(value)
        return x if _is_list(x) else [x]


@dataclass(slots=True)
class ToString(Node):
    inner: Node

    def eval(self, value: Any) -> Any:
        import json as _json

        x = self.inner.eval(value)
        return (
            x
            if isinstance(x, str)
            else _json.dumps(x, separators=(",", ":"), default=str)
        )


@dataclass(slots=True)
class ToNumber(Node):
    inner: Node

    def eval(self, value: Any) -> Any:
        x = self.inner.eval(value)
        if isinstance(x, (list, dict, bool)) or x is None:
            return None
        if isinstance(x, (int, float)):
            return x
        try:
            return int(x)
        except Exception:
            try:
                return float(x)
            except Exception:
                return None


@dataclass(slots=True)
class MapApply(Node):
    base: Node
    build: Callable[[Identity], Node]  # builder from element-root

    def eval(self, value: Any) -> Any:
        arr = self.base.eval(value)
        if not _is_list(arr):
            return None
        out: list[Any] = []
        for el in arr:
            expr = self.build(Identity())
            out.append(expr.eval(el))
        return out


@dataclass(slots=True)
class SortBy(Node):
    base: Node
    key_of: Callable[[Identity], Node]

    def eval(self, value: Any) -> Any:
        arr = self.base.eval(value)
        if not _is_list(arr) or not arr:
            return arr if _is_list(arr) else None

        def key(el: Any) -> Any:
            expr = self.key_of(Identity())
            return expr.eval(el)

        try:
            return sorted(arr, key=key)
        except Exception:
            return arr


@dataclass(slots=True)
class MinBy(Node):
    base: Node
    key_of: Callable[[Identity], Node]

    def eval(self, value: Any) -> Any:
        arr = self.base.eval(value)
        if not isinstance(arr, list) or not arr:
            return None

        def key(el: Any) -> Any:
            expr = self.key_of(Identity())
            return expr.eval(el)

        try:
            return min(arr, key=key)
        except Exception:
            return None


@dataclass(slots=True)
class MaxBy(Node):
    base: Node
    key_of: Callable[[Identity], Node]

    def eval(self, value: Any) -> Any:
        arr = self.base.eval(value)
        if not isinstance(arr, list) or not arr:
            return None

        def key(el: Any) -> Any:
            expr = self.key_of(Identity())
            return expr.eval(el)

        try:
            return max(arr, key=key)
        except Exception:
            return None


# ---- multi-selects ----


@dataclass(slots=True)
class MultiList(Node):
    items: tuple[Node, ...]

    def eval(self, value: Any) -> Any:
        return [it.eval(value) for it in self.items]


@dataclass(slots=True)
class MultiDict(Node):
    mapping: tuple[tuple[str, Node], ...]

    def eval(self, value: Any) -> Any:
        d: dict[str, Any] = {}
        for k, n in self.mapping:
            d[k] = n.eval(value)
        return d
