from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jmespath
import pytest

import querydict as qd


@dataclass(slots=True, frozen=True)
class Case:
    name: str
    expr: str
    build: Callable[[], qd.Query]
    data: Any


def run_case(c: Case) -> None:
    q = c.build()
    got = q.search(c.data)
    want = jmespath.search(c.expr, c.data)
    assert got == want, f"{c.name}: {got=} != {want=}"


DATA_USER = {
    "users": [
        {"name": "Ada", "age": 36},
        {"name": "Bob", "age": 17},
        {"name": "Cy", "age": 20},
    ]
}

DATA_MIXED = {
    "foo": {"bar": [{"baz": 1}, {"baz": 2}]},
    "stats": {"a": 3, "b": 1, "c": 2},
    "arr": [3, 1, 2, 2],
    "nested": [[1, 2], [3], 4],
}

DATA_EDGE = {
    "numbers": [0, 1, 2],
    "truth": [True, False],
    "obj": {"x": {"y": {"z": 5}}},
}

CASES: list[Case] = [
    Case(
        name="dot-field-index-dot",
        expr="foo.bar[0].baz",
        build=lambda: qd.identity().field("foo").field("bar").index(0).field("baz"),
        data=DATA_MIXED,
    ),
    Case(
        name="simple-field",
        expr="users[0].name",
        build=lambda: qd.identity().field("users").index(0).field("name"),
        data=DATA_USER,
    ),
    Case(
        name="slice",
        expr="arr[1:3]",
        build=lambda: qd.identity().field("arr").slice(1, 3),
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
        build=lambda: qd.identity().field("stats").values().sort(),
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
        name="flatten",
        expr="flatten(nested)",
        build=lambda: qd.identity().field("nested").flatten(),
        data=DATA_MIXED,
    ),
    Case(
        name="multi-select-dict",
        expr="{a: stats.a, b: stats.b}",
        build=lambda: qd.select_dict(
            a=qd.identity().field("stats").field("a"),
            b=qd.identity().field("stats").field("b"),
        ),
        data=DATA_MIXED,
    ),
    Case(
        name="pipe-length",
        expr="foo.bar | length(@)",
        build=lambda: qd.identity()
        .field("foo")
        .field("bar")
        .pipe(qd.identity().length()),
        data=DATA_MIXED,
    ),
    Case(
        name="numbers-vs-bool-eq",
        expr="numbers[0] == `False`",
        build=lambda: qd.identity().field("numbers").index(0).eq(qd.lit(False)),
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
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_equivalence(case: Case) -> None:
    run_case(case)


def test_map_with_expref_equiv() -> None:
    # map(&length(@), users[*].name)  ~=  users[*].name |> map_with(lambda e: e.length())
    q = (
        qd.identity()
        .field("users")
        .project(qd.identity().field("name"))
        .map_with(lambda e: e.length())
    )
    got = q.search(DATA_USER)
    want = jmespath.search("map(&length(@), users[*].name)", DATA_USER)
    assert got == want
