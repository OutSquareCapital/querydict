import json
from collections.abc import Callable
from typing import Any

from ._core import is_comparable, is_list, is_mapping, is_number, is_sized, not_empty


def eq(
    left_eval: Callable[[Any], Any],
    right_eval: Callable[[Any], Any],
    op: Callable[[Any, Any], bool],
) -> Callable[[Any], bool]:
    def _eval(value: Any) -> bool:
        return op(left_eval(value), right_eval(value))

    return _eval


def not_(expr_eval: Callable[[Any], Any]) -> Callable[[Any], bool]:
    def _eval(value: Any) -> bool:
        v = expr_eval(value)
        if is_number(v) and v == 0:
            return True
        return not not_empty(v)

    return _eval


def index(i: int) -> Callable[[Any], Any]:
    def _eval(value: Any) -> Any:
        if not is_list(value):
            return None
        try:
            return value[i]
        except IndexError:
            return None

    return _eval


def slice_on(
    start: int | None, end: int | None, step: int | None
) -> Callable[[Any], list[Any] | None]:
    slc = slice(start, end, step)

    def _eval(value: Any) -> list[Any] | None:
        return value[slc] if is_list(value) else None

    return _eval


def multi_list(items: tuple[Callable[[Any], Any], ...]) -> Callable[[Any], Any]:
    def _eval(value: Any) -> Any:
        return [it_eval(value) for it_eval in items]

    return _eval


def key_node(
    base_eval: Callable[[Any], Any],
    func: Callable[[list[Any], Any], Any],
    key_eval: Callable[[Any], Any],
) -> Callable[[Any], Any]:
    def _key_func() -> Callable[[Any], Any]:
        return lambda el: key_eval(el)

    key_fn = _key_func()

    def _eval(value: Any) -> Any:
        arr = base_eval(value)
        return func(arr, key_fn) if is_list(arr) else None

    return _eval


def multi_dict(
    mapping: tuple[tuple[str, Callable[[Any], Any]], ...],
) -> Callable[[Any], Any]:
    def _eval(value: Any) -> Any:
        return {k: n_eval(value) for k, n_eval in mapping}

    return _eval


def project_base(
    base_eval: Callable[[Any], Any],
    rhs_eval: Callable[[Any], Any],
    iter_func: Callable[[Any], list[Any] | None],
) -> Callable[[Any], list[Any] | None]:
    def _eval(value: Any) -> list[Any] | None:
        seq = iter_func(base_eval(value))
        return (
            [rhs_eval(el) for el in seq if el is not None] if seq is not None else None
        )

    return _eval


def filter_projection(
    base_eval: Callable[[Any], Any],
    then_eval: Callable[[Any], Any],
    cond_eval: Callable[[Any], Any],
) -> Callable[[Any], list[Any] | None]:
    def _eval(value: Any) -> list[Any] | None:
        seq = base_eval(value)
        return filter_project(seq, then_eval, cond_eval)

    return _eval


def callable_node(
    base: Callable[[Any], Any], func: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    def _eval(value: Any) -> Any:
        return func(base(value))

    return _eval


def identity() -> Callable[[Any], Any]:
    def _eval(value: Any) -> Any:
        return value

    return _eval


def literal() -> Callable[[Any], Any]:
    def _eval(value: Any) -> Any:
        return value

    return _eval


def field(name: str) -> Callable[[Any], Any]:
    def _eval(value: Any) -> Any:
        return value.get(name, None) if is_mapping(value) else None

    return _eval


def associate(
    left_eval: Callable[[Any], Any], right_eval: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    def _eval(value: Any) -> Any:
        left_val = left_eval(value)
        return left_val if not_empty(left_val) else right_eval(value)

    return _eval


def flatten(base_eval: Callable[[Any], Any]) -> Callable[[Any], Any]:
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


def map_apply(
    base_eval: Callable[[Any], Any], build_eval: Callable[[Any], Any]
) -> Callable[[Any], list[Any] | None]:
    def _eval(value: Any) -> list[Any] | None:
        arr = base_eval(value)
        return [build_eval(el) for el in arr] if is_list(arr) else None

    return _eval


def comparator(
    left_eval: Callable[[Any], Any],
    right_eval: Callable[[Any], Any],
    op: Callable[[Any, Any], bool],
) -> Callable[[Any], bool | None]:
    def _eval(value: Any) -> bool | None:
        left = left_eval(value)
        right = right_eval(value)
        return (
            op(left, right) if (is_comparable(left) and is_comparable(right)) else None
        )

    return _eval


def and_(
    left_eval: Callable[[Any], Any], right_eval: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    def _eval(value: Any) -> Any:
        left_val = left_eval(value)
        return right_eval(value) if not_empty(left_val) else left_val

    return _eval


def pipe(
    left_eval: Callable[[Any], Any], right_eval: Callable[[Any], Any]
) -> Callable[[Any], Any]:
    def _eval(value: Any) -> Any:
        return right_eval(left_eval(value))

    return _eval


def filter_project(
    value: Any, then_eval: Callable[[Any], Any], cond_eval: Callable[[Any], Any]
) -> list[Any] | None:
    return (
        [then_eval(el) for el in value if not_empty(cond_eval(el))]
        if is_list(value)
        else None
    )


def sub_expr(part_evals: tuple[Callable[[Any], Any], ...]) -> Callable[[Any], Any]:
    def _eval(value: Any) -> Any:
        out = value
        for p_eval in part_evals:
            out = p_eval(out)
        return out

    return _eval


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
