from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jmespath

import querydict as qd


@dataclass(slots=True, frozen=True)
class Case:
    name: str
    expr: str
    build: Callable[[], qd.Query]
    data: Any


def check(c: Case) -> None:
    got = c.build().search(c.data)
    want = jmespath.search(c.expr, c.data)
    assert got == want, f"{c.name}: {got=!r} != {want=!r}"
    print(f"✔ {c.name}")


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
        expr="foo.bar[0].baz",
        build=lambda: (qd.identity().field("foo").field("bar").index(0).field("baz")),
        data=DATA_MIXED,
    ),
    Case(
        name="simple-field",
        expr="users[0].name",
        build=lambda: (qd.identity().field("users").index(0).field("name")),
        data=DATA_USER,
    ),
    Case(
        name="slice",
        expr="arr[1:3]",
        build=lambda: (qd.identity().field("arr").slice(1, 3)),
        data=DATA_MIXED,
    ),
    Case(
        name="projection",
        expr="foo.bar[*].baz",
        build=lambda: (
            qd.identity().field("foo").field("bar").project(qd.identity().field("baz"))
        ),
        data=DATA_MIXED,
    ),
    Case(
        name="value-projection-sort",
        expr="values(stats) | sort(@)",
        build=lambda: (qd.identity().field("stats").values().sort()),
        data=DATA_MIXED,
    ),
    Case(
        name="filter-then-name",
        expr="users[?age >= `18`].name",
        build=lambda: (
            qd.identity()
            .field("users")
            .filter(
                qd.identity().field("age").gte(qd.lit(18)),
                then=qd.identity().field("name"),
            )
        ),
        data=DATA_USER,
    ),
    Case(
        name="multi-select-dict",
        expr="{a: stats.a, b: stats.b}",
        build=lambda: (
            qd.select_dict(
                a=qd.identity().field("stats").field("a"),
                b=qd.identity().field("stats").field("b"),
            )
        ),
        data=DATA_MIXED,
    ),
    Case(
        name="pipe-length",
        expr="foo.bar | length(@)",
        build=lambda: (
            qd.identity().field("foo").field("bar").pipe(qd.identity().length())
        ),
        data=DATA_MIXED,
    ),
    Case(
        name="numbers-vs-bool-eq",
        expr="numbers[0] == `False`",
        build=lambda: (qd.identity().field("numbers").index(0).eq(qd.lit(False))),
        data=DATA_EDGE,
    ),
    Case(
        name="and-or-not",
        expr="(obj.x.y.z > `1`) && !(obj.x.y.z == `5`) || `0`",
        build=lambda: (
            qd.identity()
            .field("obj")
            .field("x")
            .field("y")
            .field("z")
            .gt(qd.lit(1))
            .and_(
                qd.identity()
                .field("obj")
                .field("x")
                .field("y")
                .field("z")
                .eq(qd.lit(5))
                .not_()
            )
            .or_(qd.lit(0))
        ),
        data=DATA_EDGE,
    ),
    Case(
        name="map_with-length",
        expr="map(&length(@), users[*].name)",
        build=lambda: (
            qd.identity()
            .field("users")
            .project(qd.identity().field("name"))
            .map_with(lambda e: e.length())
        ),
        data=DATA_USER,
    ),
    Case(
        name="sort_by-age",
        expr="sort_by(users, &age)",
        build=lambda: (qd.identity().field("users").sort_by(lambda e: e.field("age"))),
        data=DATA_USER,
    ),
    Case(
        name="min_by-age",
        expr="min_by(users, &age)",
        build=lambda: (qd.identity().field("users").min_by(lambda e: e.field("age"))),
        data=DATA_USER,
    ),
    Case(
        name="max_by-age",
        expr="max_by(users, &age)",
        build=lambda: (qd.identity().field("users").max_by(lambda e: e.field("age"))),
        data=DATA_USER,
    ),
    Case(
        name="to_array-wrap",
        expr="to_array(stats.a)",
        build=lambda: (qd.identity().field("stats").field("a").to_array()),
        data=DATA_MIXED,
    ),
    Case(
        name="to_string-json",
        expr="to_string(stats)",
        build=lambda: (qd.identity().field("stats").to_string()),
        data=DATA_MIXED,
    ),
    Case(
        name="to_number-valid",
        expr="to_number('42')",
        build=lambda: (qd.lit("42").to_number()),
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
