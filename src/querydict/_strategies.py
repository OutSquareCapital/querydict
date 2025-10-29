import json
from collections.abc import Callable
from typing import Any

from ._core import is_list, is_mapping, is_sized


def length(value: Any) -> int | None:
    return len(value) if is_sized(value) else None


def sort(value: list[Any]) -> list[Any] | None:
    match value:
        case list():
            try:
                return sorted(value)
            except Exception:
                return value
        case _:
            return None


def keys(value: Any) -> list[Any] | None:
    return list(value.keys()) if is_mapping(value) else None


def values(value: Any) -> list[Any] | None:
    return list(value.values()) if is_mapping(value) else None


def _convert_obj(value: Any) -> int | float | None:
    try:
        return int(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return None


def to_number(value: Any) -> int | float | None:
    match value:
        case (list() | dict() | bool()) | None:
            return None
        case int() | float():
            return value
        case _:
            return _convert_obj(value)


def to_string(value: Any) -> Any:
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, separators=(",", ":"), default=str)
    )


def to_array(value: Any) -> list[Any]:
    return value if is_list(value) else [value]


def array_project(value: Any) -> list[Any] | None:
    return value if is_list(value) else None


def object_project(value: Any) -> list[Any] | None:
    return list(value.values()) if is_mapping(value) else None


def sort_by(arr: list[Any], key_fn: Callable[[Any], Any]) -> list[Any]:
    try:
        return sorted(arr, key=key_fn)
    except Exception:
        return arr


def min_by(arr: list[Any], key_fn: Callable[[Any], Any]) -> Any | None:
    try:
        return min(arr, key=key_fn)
    except Exception:
        return None


def max_by(arr: list[Any], key_fn: Callable[[Any], Any]) -> Any | None:
    try:
        return max(arr, key=key_fn)
    except Exception:
        return None
