from __future__ import annotations

from collections.abc import Callable, Mapping, Sized
from typing import TYPE_CHECKING, Any, TypeIs

if TYPE_CHECKING:
    from ._main import Query
    from ._nodes import Node

type IntoExpr = Node | Query | str | int | float | bool | None

type EvalFunc = Callable[[Any], Any]


def is_sized(x: object) -> TypeIs[Sized]:
    return isinstance(x, Sized)


def is_number(x: object) -> TypeIs[int | float]:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_mapping(x: object) -> TypeIs[Mapping[Any, Any]]:
    return isinstance(x, Mapping)


def is_list(x: object) -> TypeIs[list[Any]]:
    return isinstance(x, list)


def is_comparable(x: object) -> TypeIs[int | float | str]:
    return is_number(x) or isinstance(x, str)


def eq(x: object, y: object) -> bool:
    if is_number(x) and x in (0, 1):
        return not isinstance(y, bool)
    if is_number(y) and y in (0, 1):
        return not isinstance(x, bool)
    return x == y


def ne(x: object, y: object) -> bool:
    return not eq(x, y)


def is_empty(v: object) -> bool:
    return v in ("", [], {}) or v is None or v is False


def not_empty(v: object) -> bool:
    return not is_empty(v)
