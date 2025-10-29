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
    js_query: str
    data: dict[str, Any]

    def check(self) -> None:
        got = self.build().search(self.data)
        want = jmespath.search(self.js_query, self.data)
        assert got == want, (
            f"{self.name}: \n{got=!r} != \n{want=!r}  \nexpr={self.js_query!r}"
        )
        print(
            f"✔ {self.name}, \nquerydict: \n  {self.build()}, \njmes: \n  {self.js_query}, \nresult: \n  {got!r}"
        )


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
        js_query="foo.bar[0].baz",
        data=DATA_MIXED,
    ),
    Case(
        name="simple-field",
        build=lambda: qd.field("users").index(0).name,
        js_query="users[0].name",
        data=DATA_USER,
    ),
    Case(
        name="slice",
        build=lambda: qd.field("arr").slice(1, 3),
        js_query="arr[1:3]",
        data=DATA_MIXED,
    ),
    Case(
        name="projection",
        build=lambda: qd.field("foo").bar.project("baz"),
        js_query="foo.bar[].baz",
        data=DATA_MIXED,
    ),
    Case(
        name="value-projection-sort",
        build=lambda: qd.field("stats").values().sort(),
        js_query="stats.* | sort(@)",
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
        js_query="users[?age >= `18`].name",
        data=DATA_USER,
    ),
    Case(
        name="multi-select-dict",
        build=lambda: qd.select_dict(
            a=qd.field("stats").a,
            b=qd.field("stats").b,
        ),
        js_query="{a: stats.a, b: stats.b}",
        data=DATA_MIXED,
    ),
    Case(
        name="pipe-length",
        build=lambda: qd.field("foo").bar.length(),
        js_query="length(foo.bar)",
        data=DATA_MIXED,
    ),
    Case(
        name="numbers-vs-bool-eq",
        build=lambda: qd.field("numbers").index(0).eq(False),
        js_query="numbers[0] == `false`",
        data=DATA_EDGE,
    ),
    Case(
        name="and-or-not",
        build=lambda: (
            qd.field("obj").x.y.z.gt(1).and_(qd.field("obj").x.y.z.eq(5).not_()).or_(0)
        ),
        js_query="obj.x.y.z > `1` && obj.x.y.z != `5` || `0`",
        data=DATA_EDGE,
    ),
    # higher-order
    Case(
        name="map_with-length",
        build=lambda: (
            qd.field("users").project("name").map_with(lambda e: e.length())
        ),
        js_query="map(&length(@), users[*].name)",
        data=DATA_USER,
    ),
    Case(
        name="sort_by-age",
        build=lambda: qd.field("users").sort_by(lambda e: e.age),
        js_query="sort_by(users, &age)",
        data=DATA_USER,
    ),
    Case(
        name="min_by-age",
        build=lambda: qd.field("users").min_by(lambda e: e.age),
        js_query="min_by(users, &age)",
        data=DATA_USER,
    ),
    Case(
        name="max_by-age",
        build=lambda: qd.field("users").max_by(lambda e: e.age),
        js_query="max_by(users, &age)",
        data=DATA_USER,
    ),
    # conversions
    Case(
        name="to_array-wrap",
        build=lambda: qd.field("stats").a.to_array(),
        js_query="to_array(stats.a)",
        data=DATA_MIXED,
    ),
    Case(
        name="to_string-json",
        build=lambda: qd.field("stats").to_string(),
        js_query="to_string(stats)",
        data=DATA_MIXED,
    ),
    Case(
        name="to_number-valid",
        build=lambda: qd.lit("42").to_number(),
        js_query='to_number(`"42"`)',
        data=DATA_MIXED,
    ),
    Case(
        name="flatten-nested",
        build=lambda: qd.field("nested").flatten(),
        js_query="nested[]",
        data=DATA_MIXED,
    ),
]


def main() -> None:
    print(f"Running {len(CASES)} cases…\n")
    for c in CASES:
        c.check()
    print("\nAll good.")


if __name__ == "__main__":
    main()
