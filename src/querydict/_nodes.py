from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ._checks import is_list, is_mapping, is_number, is_sized, not_empty
from ._core import (
    AssociativeNode,
    CallableNode,
    Converter,
    Identity,
    KeyNode,
    Kword,
    Node,
    ProjectionBase,
    as_ref,
    ensure_leading_dot,
)


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
        return Kword.DOT + self.name


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

    def eval(self, value: Any) -> list[Any] | None:
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
        return s[1:] if s.startswith(Kword.DOT) else s


@dataclass(slots=True, repr=False)
class Pipe(Node):
    left: Node
    right: Node

    def eval(self, value: Any) -> Any:
        return self.right.eval(self.left.eval(value))

    def as_jmespath(self) -> str:
        return f"{self.left.as_jmespath()} {Kword.PIPE} {self.right.as_jmespath()}"


@dataclass(slots=True, repr=False)
class ArrayProject(ProjectionBase):
    SEP: Kword = Kword.ARRAY_PROJECT

    def _iter(self, value: Any) -> list[Any] | None:
        return value if is_list(value) else None


@dataclass(slots=True, repr=False)
class ObjectProject(ProjectionBase):
    SEP: Kword = Kword.OBJECT_PROJECT

    def _iter(self, value: Any) -> list[Any] | None:
        return list(value.values()) if is_mapping(value) else None


@dataclass(slots=True, repr=False)
class FilterProjection(Node):
    base: Node
    then: Node
    cond: Node

    def eval(self, value: Any) -> list[Any] | None:
        seq = self.base.eval(value)
        if not is_list(seq):
            return None
        return [self.then.eval(el) for el in seq if not_empty(self.cond.eval(el))]

    def as_jmespath(self) -> str:
        base = self.base.as_jmespath()
        cond = self.cond.as_jmespath()
        cond = cond[1:] if cond.startswith(Kword.DOT) else cond
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
        return self.base.as_jmespath() + Kword.FLATTEN


@dataclass(slots=True, repr=False)
class And(AssociativeNode):
    def eval(self, value: Any) -> Any:
        left = self.left.eval(value)
        return self.right.eval(value) if not_empty(left) else left


# ...existing code...
@dataclass(slots=True, repr=False)
class Or(AssociativeNode):
    pass


@dataclass(slots=True, repr=False)
class Not(Node):
    expr: Node

    def eval(self, value: Any) -> bool:
        v = self.expr.eval(value)
        if is_number(v) and v == 0:
            return False
        return not v

    def as_jmespath(self) -> str:
        return f"!{self.expr.as_jmespath()}"


@dataclass(slots=True, repr=False)
class Length(CallableNode):
    def eval(self, value: Any) -> int | None:
        x = self.inner.eval(value)
        if is_sized(x):
            return len(x)
        return None


@dataclass(slots=True, repr=False)
class Sort(CallableNode):
    def eval(self, value: Any) -> list[Any] | None:
        xs = self.inner.eval(value)
        if is_list(xs):
            try:
                return sorted(xs)
            except Exception:
                return xs
        return None


@dataclass(slots=True, repr=False)
class Keys(CallableNode):
    def eval(self, value: Mapping[Any, Any] | Any) -> list[Any] | None:
        x = self.inner.eval(value)
        return list(x.keys()) if is_mapping(x) else None


@dataclass(slots=True, repr=False)
class Values(CallableNode):
    def eval(self, value: Any) -> list[Any] | None:
        x = self.inner.eval(value)
        return list(x.values()) if is_mapping(x) else None


@dataclass(slots=True, repr=False)
class Array(Converter):
    def eval(self, value: Any) -> list[Any]:
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
    def eval(self, value: Any) -> int | float | None:
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

    def eval(self, value: Any) -> list[Any] | None:
        arr = self.base.eval(value)
        if not is_list(arr):
            return None
        return [self.build(Identity()).eval(el) for el in arr]

    def as_jmespath(self) -> str:
        fn = as_ref(self.build(Identity()))
        return f"map({fn}, {self.base.as_jmespath()})"


@dataclass(slots=True, repr=False)
class SortBy(KeyNode):
    key_name: str = "sort_by"

    def _eval_impl(self, arr: list[Any]) -> list[Any]:
        try:
            return sorted(arr, key=self._key)
        except Exception:
            return arr


@dataclass(slots=True, repr=False)
class MinBy(KeyNode):
    key_name: str = "min_by"

    def _eval_impl(self, arr: list[Any]) -> list[Any] | None:
        try:
            return min(arr, key=self._key)
        except Exception:
            return None


@dataclass(slots=True, repr=False)
class MaxBy(KeyNode):
    key_name: str = "max_by"

    def _eval_impl(self, arr: list[Any]) -> list[Any] | None:
        try:
            return max(arr, key=self._key)
        except Exception:
            return None


@dataclass(slots=True, repr=False)
class MultiList(Node):
    items: tuple[Node, ...]

    def eval(self, value: Any) -> Any:
        return [it.eval(value) for it in self.items]

    def as_jmespath(self) -> str:
        inner = ", ".join(it.as_jmespath() or Kword.CURRENT for it in self.items)
        return f"[{inner}]"


@dataclass(slots=True, repr=False)
class MultiDict(Node):
    mapping: tuple[tuple[str, Node], ...]

    def eval(self, value: Any) -> Any:
        return {k: n.eval(value) for k, n in self.mapping}

    def as_jmespath(self) -> str:
        items = ", ".join(
            f"{k}: {n.as_jmespath() or Kword.CURRENT}" for k, n in self.mapping
        )
        return f"{{{items}}}"
