from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jmespath

import querydict as qd


@dataclass(slots=True, frozen=True)
class Case:
    name: str
    build: Callable[[], qd.Query]
    data: Any


def check(c: Case) -> None:
    q = c.build()
    expr = q.to_jmespath()
    got = q.search(c.data)
    want = jmespath.search(expr, c.data)
    assert got == want, f"{c.name}: \n{got=!r} != \n{want=!r}  \nexpr={expr!r}"
    print(f"✔ {c.name}, \nexpr: \n  {expr}, \nresult: \n  {got!r}")


DATA_USER: dict[str, Any] = {
    "users": [
        {"name": "Ada", "age": 36},
        {"name": "Bob", "age": 17},
        {"name": "Cy", "age": 20},
    ]
}

DATA_MIXED: dict[str, Any] = {
    "foo": {"bar": [{"baz": 1}, {"baz": 2}]},
    "stats": {"a": 3, "b": 1, "c": 2},
    "arr": [3, 1, 2, 2],
    "nested": [[1, 2], [3], 4],
}

DATA_EDGE: dict[str, Any] = {
    "numbers": [0, 1, 2],
    "truth": [True, False],
    "obj": {"x": {"y": {"z": 5}}},
}

CASES: list[Case] = [
    Case(
        name="dot-field-index-dot",
        build=lambda: qd.field("foo").bar.index(0).baz,
        data=DATA_MIXED,
    ),
    Case(
        name="simple-field",
        build=lambda: qd.field("users").index(0).name,
        data=DATA_USER,
    ),
    Case(
        name="slice",
        build=lambda: qd.field("arr").slice(1, 3),
        data=DATA_MIXED,
    ),
    Case(
        name="projection",
        build=lambda: qd.field("foo").bar.project("baz"),
        data=DATA_MIXED,
    ),
    Case(
        name="value-projection-sort",
        build=lambda: qd.field("stats").values().sort(),
        data=DATA_MIXED,
    ),
    Case(
        name="filter-then-name",
        build=lambda: (
            qd.field("users").filter(
                qd.field("age").ge(18),
                then="name",
            )
        ),
        data=DATA_USER,
    ),
    Case(
        name="multi-select-dict",
        build=lambda: qd.select_dict(
            a=qd.field("stats").a,
            b=qd.field("stats").b,
        ),
        data=DATA_MIXED,
    ),
    Case(
        name="pipe-length",
        build=lambda: qd.field("foo").bar.length(),
        data=DATA_MIXED,
    ),
    Case(
        name="numbers-vs-bool-eq",
        build=lambda: qd.field("numbers").index(0).eq(False),
        data=DATA_EDGE,
    ),
    Case(
        name="and-or-not",
        build=lambda: (
            qd.field("obj").x.y.z.gt(1).and_(qd.field("obj").x.y.z.eq(5).not_()).or_(0)
        ),
        data=DATA_EDGE,
    ),
    # higher-order
    Case(
        name="map_with-length",
        build=lambda: (
            qd.field("users").project("name").map_with(lambda e: e.length())
        ),
        data=DATA_USER,
    ),
    Case(
        name="sort_by-age",
        build=lambda: qd.field("users").sort_by(lambda e: e.age),
        data=DATA_USER,
    ),
    Case(
        name="min_by-age",
        build=lambda: qd.field("users").min_by(lambda e: e.age),
        data=DATA_USER,
    ),
    Case(
        name="max_by-age",
        build=lambda: qd.field("users").max_by(lambda e: e.age),
        data=DATA_USER,
    ),
    # conversions
    Case(
        name="to_array-wrap",
        build=lambda: qd.field("stats").a.to_array(),
        data=DATA_MIXED,
    ),
    Case(
        name="to_string-json",
        build=lambda: qd.field("stats").to_string(),
        data=DATA_MIXED,
    ),
    Case(
        name="to_number-valid",
        build=lambda: qd.lit("42").to_number(),
        data=DATA_MIXED,
    ),
    Case(
        name="flatten-nested",
        build=lambda: qd.field("nested").flatten(),
        data=DATA_MIXED,
    ),
]


def main() -> None:
    print(f"Running {len(CASES)} cases…\n")
    for c in CASES:
        check(c)
    print("\nAll good.")


if __name__ == "__main__":
    main()
