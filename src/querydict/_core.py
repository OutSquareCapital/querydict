from __future__ import annotations

from collections.abc import Mapping, Sized
from enum import StrEnum
from typing import Any, TypeIs


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


def _is_leading_dot(text: str) -> bool:
    return text.startswith((Kword.DOT, "[", "{", "(", "`", Kword.CURRENT) or text == "")


def ensure_leading_dot(text: str) -> str:
    match text:
        case text if _is_leading_dot(text):
            return text
        case _:
            return Kword.DOT + text


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
