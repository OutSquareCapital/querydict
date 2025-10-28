from __future__ import annotations

import json as _json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._checks import is_list, is_mapping, is_number, not_empty
from ._core import AssociativeNode, Identity, Kword, Node, as_ref, ensure_leading_dot


@dataclass(slots=True, repr=False)
class LiteralExpr(Node):
    value: Any

    def eval(self) -> Callable[[Any], Any]:
        val = self.value

        def _eval(value: Any) -> Any:
            return val

        return _eval

    def as_jmespath(self) -> str:
        return f"`{_json.dumps(self.value, separators=(',', ':'), default=str)}`"


@dataclass(slots=True, repr=False)
class Field(Node):
    name: str

    def eval(self) -> Callable[[Any], Any]:
        name = self.name

        def _eval(value: Any) -> Any:
            if is_mapping(value):
                return value.get(name)
            return None

        return _eval

    def as_jmespath(self) -> str:
        return Kword.DOT + self.name


@dataclass(slots=True, repr=False)
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

    def as_jmespath(self) -> str:
        return f"[{self.i}]"


@dataclass(slots=True, repr=False)
class Slice(Node):
    start: int | None
    end: int | None
    step: int | None

    def eval(self) -> Callable[[Any], list[Any] | None]:
        slc = slice(self.start, self.end, self.step)

        def _eval(value: Any) -> list[Any] | None:
            if not is_list(value):
                return None
            return value[slc]

        return _eval

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

    def eval(self) -> Callable[[Any], Any]:
        part_evals = tuple(p.eval() for p in self.parts)

        def _eval(value: Any) -> Any:
            out = value
            for p_eval in part_evals:
                out = p_eval(out)
            return out

        return _eval

    def as_jmespath(self) -> str:
        s = "".join(p.as_jmespath() for p in self.parts)
        return s[1:] if s.startswith(Kword.DOT) else s


@dataclass(slots=True, repr=False)
class Pipe(Node):
    left: Node
    right: Node

    def eval(self) -> Callable[[Any], Any]:
        left_eval = self.left.eval()
        right_eval = self.right.eval()

        def _eval(value: Any) -> Any:
            return right_eval(left_eval(value))

        return _eval

    def as_jmespath(self) -> str:
        return f"{self.left.as_jmespath()} {Kword.PIPE} {self.right.as_jmespath()}"


@dataclass(slots=True, repr=False)
class FilterProjection(Node):
    base: Node
    then: Node
    cond: Node

    def eval(self) -> Callable[[Any], list[Any] | None]:
        base_eval = self.base.eval()
        then_eval = self.then.eval()
        cond_eval = self.cond.eval()

        def _eval(value: Any) -> list[Any] | None:
            seq = base_eval(value)
            if not is_list(seq):
                return None
            return [then_eval(el) for el in seq if not_empty(cond_eval(el))]

        return _eval

    def as_jmespath(self) -> str:
        base = self.base.as_jmespath()
        cond = self.cond.as_jmespath()
        cond = cond[1:] if cond.startswith(Kword.DOT) else cond
        then = ensure_leading_dot(self.then.as_jmespath())
        return f"{base}[?{cond}]{then}"


@dataclass(slots=True, repr=False)
class Flatten(Node):
    base: Node

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

    def as_jmespath(self) -> str:
        return self.base.as_jmespath() + Kword.FLATTEN


@dataclass(slots=True, repr=False)
class And(AssociativeNode):
    def eval(self) -> Callable[[Any], Any]:
        left_eval = self.left.eval()
        right_eval = self.right.eval()

        def _eval(value: Any) -> Any:
            left_val = left_eval(value)
            return right_eval(value) if not_empty(left_val) else left_val

        return _eval


@dataclass(slots=True, repr=False)
class Or(AssociativeNode):
    pass


@dataclass(slots=True, repr=False)
class Not(Node):
    expr: Node

    def eval(self) -> Callable[[Any], bool]:
        expr_eval = self.expr.eval()

        def _eval(value: Any) -> bool:
            v = expr_eval(value)
            if is_number(v) and v == 0:
                return True
            return not not_empty(v)

        return _eval

    def as_jmespath(self) -> str:
        return f"!{self.expr.as_jmespath()}"


@dataclass(slots=True, repr=False)
class MapApply(Node):
    base: Node
    build: Callable[[Identity], Node]

    def eval(self) -> Callable[[Any], list[Any] | None]:
        base_eval = self.base.eval()
        build_eval = self.build(Identity()).eval()

        def _eval(value: Any) -> list[Any] | None:
            arr = base_eval(value)
            if not is_list(arr):
                return None
            return [build_eval(el) for el in arr]

        return _eval

    def as_jmespath(self) -> str:
        fn = as_ref(self.build(Identity()))
        return f"map({fn}, {self.base.as_jmespath()})"


@dataclass(slots=True, repr=False)
class MultiList(Node):
    items: tuple[Node, ...]

    def eval(self) -> Callable[[Any], Any]:
        item_evals = tuple(it.eval() for it in self.items)

        def _eval(value: Any) -> Any:
            return [it_eval(value) for it_eval in item_evals]

        return _eval

    def as_jmespath(self) -> str:
        inner = ", ".join(it.as_jmespath() or Kword.CURRENT for it in self.items)
        return f"[{inner}]"


@dataclass(slots=True, repr=False)
class MultiDict(Node):
    mapping: tuple[tuple[str, Node], ...]

    def eval(self) -> Callable[[Any], Any]:
        mapping_evals = tuple((k, n.eval()) for k, n in self.mapping)

        def _eval(value: Any) -> Any:
            return {k: n_eval(value) for k, n_eval in mapping_evals}

        return _eval

    def as_jmespath(self) -> str:
        items = ", ".join(
            f"{k}: {n.as_jmespath() or Kword.CURRENT}" for k, n in self.mapping
        )
        return f"{{{items}}}"
