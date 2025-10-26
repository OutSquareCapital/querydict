from __future__ import annotations

import operator
from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ._checks import (
    equals,
    falsy,
    is_comparable,
    is_list,
    is_mapping,
    is_number,
    is_sized,
    truthy,
)
from ._core import Node, as_callable, as_expref, ensure_leading_dot


@dataclass(slots=True, repr=False)
class Identity(Node):
    def eval(self, value: Any) -> Any:
        return value

    def as_jmespath(self) -> str:
        return ""


@dataclass(slots=True, repr=False)
class LiteralExpr(Node):
    value: Any

    def eval(self, value: Any) -> Any:
        return self.value

    def as_jmespath(self) -> str:
        import json as _json

        return f"`{_json.dumps(self.value, separators=(',', ':'), default=str)}`"


@dataclass(slots=True, repr=False)
class Field(Node):
    name: str

    def eval(self, value: Any) -> Any:
        if is_mapping(value):
            return value.get(self.name)
        return None

    def as_jmespath(self) -> str:
        return f".{self.name}"


@dataclass(slots=True, repr=False)
class Index(Node):
    i: int

    def eval(self, value: Any) -> Any:
        if not is_list(value):
            return None
        try:
            return value[self.i]
        except IndexError:
            return None

    def as_jmespath(self) -> str:
        return f"[{self.i}]"


@dataclass(slots=True, repr=False)
class Slice(Node):
    start: int | None
    end: int | None
    step: int | None

    def eval(self, value: Any) -> Any:
        if not is_list(value):
            return None
        return value[slice(self.start, self.end, self.step)]

    def as_jmespath(self) -> str:
        start = "" if self.start is None else str(self.start)
        end = "" if self.end is None else str(self.end)
        step = "" if self.step is None else str(self.step)
        if self.step is None:
            return f"[{start}:{end}]"
        return f"[{start}:{end}:{step}]"


@dataclass(slots=True, repr=False)
class SubExpr(Node):
    parts: tuple[Node, ...]

    def eval(self, value: Any) -> Any:
        out = value
        for p in self.parts:
            out = p.eval(out)
        return out

    def as_jmespath(self) -> str:
        s = "".join(p.as_jmespath() for p in self.parts)
        return s[1:] if s.startswith(".") else s


@dataclass(slots=True, repr=False)
class Pipe(Node):
    left: Node
    right: Node

    def eval(self, value: Any) -> Any:
        return self.right.eval(self.left.eval(value))

    def as_jmespath(self) -> str:
        return f"{self.left.as_jmespath()} | {self.right.as_jmespath()}"


@dataclass(slots=True, repr=False)
class _ProjectionBase(Node):
    base: Node
    rhs: Node
    SEP: str

    @abstractmethod
    def _iter(self, value: Any) -> list[Any] | None:
        raise NotImplementedError

    def eval(self, value: Any) -> Any:
        seq = self._iter(self.base.eval(value))
        if seq is None:
            return None
        return [self.rhs.eval(el) for el in seq if el is not None]

    def as_jmespath(self) -> str:
        return f"{self.base.as_jmespath()}{self.SEP}{ensure_leading_dot(self.rhs.as_jmespath())}"


@dataclass(slots=True, repr=False)
class Projection(_ProjectionBase):
    SEP: str = "[*]"

    def _iter(self, value: Any) -> list[Any] | None:
        return value if is_list(value) else None


@dataclass(slots=True, repr=False)
class ValueProjection(_ProjectionBase):
    SEP: str = "*"

    def _iter(self, value: Any) -> list[Any] | None:
        return list(value.values()) if is_mapping(value) else None


@dataclass(slots=True, repr=False)
class FilterProjection(Node):
    base: Node
    then: Node
    cond: Node

    def eval(self, value: Any) -> Any:
        seq = self.base.eval(value)
        if not is_list(seq):
            return None
        return [self.then.eval(el) for el in seq if truthy(self.cond.eval(el))]

    def as_jmespath(self) -> str:
        base = self.base.as_jmespath()
        cond = self.cond.as_jmespath()
        if cond.startswith("."):
            cond = cond[1:]
        then = ensure_leading_dot(self.then.as_jmespath())
        return f"{base}[?{cond}]{then}"


@dataclass(slots=True, repr=False)
class Flatten(Node):
    base: Node

    def eval(self, value: Any) -> Any:
        seq = self.base.eval(value)
        if not is_list(seq):
            return None
        flat: list[Any] = []
        for el in seq:
            if is_list(el):
                flat.extend(el)
            else:
                flat.append(el)
        return flat

    def as_jmespath(self) -> str:
        return f"{self.base.as_jmespath()}[]"


@dataclass(slots=True, repr=False)
class _BinaryCompare(Node):
    left: Node
    right: Node
    SYMBOL: str

    def as_jmespath(self) -> str:
        return f"{self.left.as_jmespath()} {self.SYMBOL} {self.right.as_jmespath()}"


@dataclass(slots=True, repr=False)
class _EqBase(_BinaryCompare):
    def _equals(self, value: Any) -> bool:
        return equals(self.left.eval(value), self.right.eval(value))


@dataclass(slots=True, repr=False)
class Eq(_EqBase):
    SYMBOL: str = "=="

    def eval(self, value: Any) -> Any:
        return self._equals(value)


@dataclass(slots=True, repr=False)
class Ne(_EqBase):
    SYMBOL: str = "!="

    def eval(self, value: Any) -> Any:
        return not self._equals(value)


@dataclass(slots=True, repr=False)
class _OrderBase(_BinaryCompare):
    OP: Callable[[Any, Any], bool] = operator.lt

    def eval(self, value: Any) -> Any:
        left = self.left.eval(value)
        right = self.right.eval(value)
        if not (is_comparable(left) and is_comparable(right)):
            return None
        return self.OP(left, right)


@dataclass(slots=True, repr=False)
class Lt(_OrderBase):
    SYMBOL: str = "<"
    OP: Callable[[Any, Any], bool] = operator.lt


@dataclass(slots=True, repr=False)
class Lte(_OrderBase):
    SYMBOL: str = "<="
    OP: Callable[[Any, Any], bool] = operator.le


@dataclass(slots=True, repr=False)
class Gt(_OrderBase):
    SYMBOL: str = ">"
    OP: Callable[[Any, Any], bool] = operator.gt


@dataclass(slots=True, repr=False)
class Gte(_OrderBase):
    SYMBOL: str = ">="
    OP: Callable[[Any, Any], bool] = operator.ge


@dataclass(slots=True, repr=False)
class And(Node):
    left: Node
    right: Node

    def eval(self, value: Any) -> Any:
        left = self.left.eval(value)
        return left if falsy(left) else self.right.eval(value)

    def as_jmespath(self) -> str:
        return f"{self.left.as_jmespath()} && {self.right.as_jmespath()}"


@dataclass(slots=True, repr=False)
class Or(Node):
    left: Node
    right: Node

    def eval(self, value: Any) -> Any:
        left = self.left.eval(value)
        return self.right.eval(value) if falsy(left) else left

    def as_jmespath(self) -> str:
        return f"{self.left.as_jmespath()} || {self.right.as_jmespath()}"


@dataclass(slots=True, repr=False)
class Not(Node):
    expr: Node

    def eval(self, value: Any) -> Any:
        v = self.expr.eval(value)
        if is_number(v) and v == 0:
            return False
        return not v

    def as_jmespath(self) -> str:
        return f"!{self.expr.as_jmespath()}"


@dataclass(slots=True, repr=False)
class CallableNode(Node):
    inner: Node

    @property
    def func(self) -> str:
        return self.__class__.__name__.lower()

    def as_jmespath(self) -> str:
        return as_callable(self.func, self.inner)


@dataclass(slots=True, repr=False)
class Length(CallableNode):
    def eval(self, value: Any) -> Any:
        x = self.inner.eval(value)
        if is_sized(x):
            return len(x)
        return None


@dataclass(slots=True, repr=False)
class Sort(CallableNode):
    def eval(self, value: Any) -> Any:
        xs = self.inner.eval(value)
        if is_list(xs):
            try:
                return sorted(xs)
            except Exception:
                return xs
        return None


@dataclass(slots=True, repr=False)
class Keys(CallableNode):
    def eval(self, value: Mapping[Any, Any] | Any) -> Any:
        x = self.inner.eval(value)
        return list(x.keys()) if is_mapping(x) else None


@dataclass(slots=True, repr=False)
class Values(CallableNode):
    def eval(self, value: Any) -> Any:
        x = self.inner.eval(value)
        return list(x.values()) if is_mapping(x) else None


@dataclass(slots=True, repr=False)
class Converter(CallableNode):
    @property
    def func(self) -> str:
        return "to_" + self.__class__.__name__.lower()


@dataclass(slots=True, repr=False)
class Array(Converter):
    def eval(self, value: Any) -> Any:
        x = self.inner.eval(value)
        return x if is_list(x) else [x]


@dataclass(slots=True, repr=False)
class String(Converter):
    def eval(self, value: Any) -> Any:
        import json as _json

        x = self.inner.eval(value)
        return (
            x
            if isinstance(x, str)
            else _json.dumps(x, separators=(",", ":"), default=str)
        )


@dataclass(slots=True, repr=False)
class Number(Converter):
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


@dataclass(slots=True, repr=False)
class MapApply(Node):
    base: Node
    build: Callable[[Identity], Node]

    def eval(self, value: Any) -> Any:
        arr = self.base.eval(value)
        if not is_list(arr):
            return None
        return [self.build(Identity()).eval(el) for el in arr]

    def as_jmespath(self) -> str:
        fn = as_expref(self.build(Identity()))
        return f"map({fn}, {self.base.as_jmespath()})"


@dataclass(slots=True, repr=False)
class SortBy(Node):
    base: Node
    key_of: Callable[[Identity], Node]

    def eval(self, value: Any) -> Any:
        arr = self.base.eval(value)
        if not is_list(arr) or not arr:
            return arr if is_list(arr) else None

        def key(el: Any) -> Any:
            return self.key_of(Identity()).eval(el)

        try:
            return sorted(arr, key=key)
        except Exception:
            return arr

    def as_jmespath(self) -> str:
        key = as_expref(self.key_of(Identity()))
        return f"sort_by({self.base.as_jmespath()}, {key})"


@dataclass(slots=True, repr=False)
class MinBy(Node):
    base: Node
    key_of: Callable[[Identity], Node]

    def eval(self, value: Any) -> Any:
        arr = self.base.eval(value)
        if not is_list(arr) or not arr:
            return None

        def key(el: Any) -> Any:
            return self.key_of(Identity()).eval(el)

        try:
            return min(arr, key=key)
        except Exception:
            return None

    def as_jmespath(self) -> str:
        key = as_expref(self.key_of(Identity()))
        return f"min_by({self.base.as_jmespath()}, {key})"


@dataclass(slots=True, repr=False)
class MaxBy(Node):
    base: Node
    key_of: Callable[[Identity], Node]

    def eval(self, value: Any) -> Any:
        arr = self.base.eval(value)
        if not is_list(arr) or not arr:
            return None

        def key(el: Any) -> Any:
            expr = self.key_of(Identity())
            return expr.eval(el)

        try:
            return max(arr, key=key)
        except Exception:
            return None

    def as_jmespath(self) -> str:
        key = as_expref(self.key_of(Identity()))
        return f"max_by({self.base.as_jmespath()}, {key})"


@dataclass(slots=True, repr=False)
class MultiList(Node):
    items: tuple[Node, ...]

    def eval(self, value: Any) -> Any:
        return [it.eval(value) for it in self.items]

    def as_jmespath(self) -> str:
        inner = ", ".join(it.as_jmespath() or "@" for it in self.items)
        return f"[{inner}]"


@dataclass(slots=True, repr=False)
class MultiDict(Node):
    mapping: tuple[tuple[str, Node], ...]

    def eval(self, value: Any) -> Any:
        d: dict[str, Any] = {}
        for k, n in self.mapping:
            d[k] = n.eval(value)
        return d

    def as_jmespath(self) -> str:
        items = ", ".join(f"{k}: {n.as_jmespath() or '@'}" for k, n in self.mapping)
        return f"{{{items}}}"
