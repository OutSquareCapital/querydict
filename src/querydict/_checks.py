from collections.abc import Mapping, Sized
from typing import Any, TypeIs


def is_sized(x: Any) -> TypeIs[Sized]:
    return isinstance(x, Sized)


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_mapping(x: Any) -> TypeIs[Mapping[Any, Any]]:
    return isinstance(x, Mapping)


def is_list(x: Any) -> TypeIs[list[Any]]:
    return isinstance(x, list)


def is_comparable(x: Any) -> bool:
    return is_number(x) or isinstance(x, str)


def eq(x: Any, y: Any) -> bool:
    if is_number(x) and x in (0, 1):
        return not isinstance(y, bool)
    if is_number(y) and y in (0, 1):
        return not isinstance(x, bool)
    return x == y


def ne(x: Any, y: Any) -> bool:
    return not eq(x, y)


def is_empty(v: Any) -> bool:
    return v in ("", [], {}) or v is None or v is False


def not_empty(v: Any) -> bool:
    return not is_empty(v)
