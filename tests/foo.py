# foo.py (version "as_jmespath")
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jmespath

import querydict as qd  # conforme


@dataclass(slots=True, frozen=True)
class Case:
    name: str
    build: Callable[[], qd.Query]
    data: Any


def check(c: Case) -> None:
    q = c.build()
    expr = q.node.as_jmespath()
    got = q.search(c.data)
    want = jmespath.search(expr, c.data)
    assert got == want, f"{c.name}: {got=!r} != {want=!r}  (expr={expr!r})"
    print(f"✔ {c.name}  [{expr}]")


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
        build=lambda: qd.identity().foo.bar.index(0).baz,
        data=DATA_MIXED,
    ),
    Case(
        name="simple-field",
        build=lambda: qd.identity().users.index(0).name,
        data=DATA_USER,
    ),
    Case(
        name="slice",
        build=lambda: qd.identity().arr.slice(1, 3),
        data=DATA_MIXED,
    ),
    Case(
        name="projection",
        build=lambda: qd.identity().foo.bar.project(qd.identity().baz),
        data=DATA_MIXED,
    ),
    Case(
        name="value-projection-sort",
        build=lambda: qd.identity().stats.values().sort(),
        data=DATA_MIXED,
    ),
    Case(
        name="filter-then-name",
        build=lambda: (
            qd.identity().users.filter(
                qd.identity().age.gte(18),
                then=qd.identity().name,
            )
        ),
        data=DATA_USER,
    ),
    Case(
        name="multi-select-dict",
        build=lambda: qd.select_dict(
            a=qd.identity().stats.a,
            b=qd.identity().stats.b,
        ),
        data=DATA_MIXED,
    ),
    Case(
        name="pipe-length",
        build=lambda: qd.identity().foo.bar.pipe(qd.identity().length()),
        data=DATA_MIXED,
    ),
    Case(
        name="numbers-vs-bool-eq",
        build=lambda: qd.identity().numbers.index(0).eq(False),
        data=DATA_EDGE,
    ),
    Case(
        name="and-or-not",
        build=lambda: (
            qd.identity()
            .obj.x.y.z.gt(1)
            .and_(qd.identity().obj.x.y.z.eq(5).not_())
            .or_(0)
        ),
        data=DATA_EDGE,
    ),
    # higher-order
    Case(
        name="map_with-length",
        build=lambda: (
            qd.identity()
            .users.project(qd.identity().name)
            .map_with(lambda e: e.length())
        ),
        data=DATA_USER,
    ),
    Case(
        name="sort_by-age",
        build=lambda: qd.identity().users.sort_by(lambda e: e.age),
        data=DATA_USER,
    ),
    Case(
        name="min_by-age",
        build=lambda: qd.identity().users.min_by(lambda e: e.age),
        data=DATA_USER,
    ),
    Case(
        name="max_by-age",
        build=lambda: qd.identity().users.max_by(lambda e: e.age),
        data=DATA_USER,
    ),
    # conversions
    Case(
        name="to_array-wrap",
        build=lambda: qd.identity().stats.a.to_array(),
        data=DATA_MIXED,
    ),
    Case(
        name="to_string-json",
        build=lambda: qd.identity().stats.to_string(),
        data=DATA_MIXED,
    ),
    Case(
        name="to_number-valid",
        build=lambda: qd.lit("42").to_number(),
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
