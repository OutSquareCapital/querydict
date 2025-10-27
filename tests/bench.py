# Fichier: bench.py
from __future__ import annotations

import random
import string
import timeit
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

import jmespath
import polars as pl

import querydict as qd

type JsonData = dict[str, Any]
type QueryFunc = Callable[[JsonData], Any]


class BenchmarkResult(TypedDict):
    size: int
    case_name: str
    qrydict: float
    jmespth: float
    jmwapth_compiled: float


@dataclass(slots=True, frozen=True)
class BenchmarkCase:
    name: str
    qd_query: qd.Query
    jp_expr: str


def rand_str(k: int) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=k))


def generate_data(n: int) -> JsonData:
    users: list[dict[str, Any]] = []
    for i in range(n):
        user = {
            "id": i,
            "name": rand_str(10),
            "age": random.randint(18, 65),
            "active": random.choice([True, False]),
            "tags": [rand_str(5) for _ in range(random.randint(1, 5))],
        }
        users.append(user)
    return {"users": users}


BENCHMARKS: list[BenchmarkCase] = [
    BenchmarkCase(
        name="Projection simple (names)",
        qd_query=qd.identity().users.project(qd.identity().name),
        jp_expr="users[*].name",
    ),
    BenchmarkCase(
        name="Filtre complexe (active & >30)",
        qd_query=qd.identity()
        .users.filter(
            qd.identity().age.gt(30).and_(qd.identity().active.eq(True)),
        )
        .name,
        jp_expr="users[?age > `30` && active == `true`].name",
    ),
    BenchmarkCase(
        name="Tri (sort_by age)",
        qd_query=qd.identity().users.sort_by(lambda e: e.age),
        jp_expr="sort_by(users, &age)",
    ),
]

DATA_SIZES: list[int] = [100, 1_000, 10_000]


def time_query(query_func: QueryFunc, data: JsonData, number: int) -> float:
    timer = timeit.Timer(lambda: query_func(data))
    total_time = timer.timeit(number=number)
    return total_time / number


def main(runs: int) -> pl.DataFrame:
    print(f"Lancement des benchmarks (Runs par test: {runs})\n")
    results: list[BenchmarkResult] = []
    for size in DATA_SIZES:
        data = generate_data(size)

        for case in BENCHMARKS:
            jp_compiled = jmespath.compile(case.jp_expr)
            jp_expr = case.jp_expr

            t_qd = time_query(case.qd_query.search, data, runs)
            t_jp_c = time_query(jp_compiled.search, data, runs)
            t_jp_e = time_query(lambda d: jmespath.search(jp_expr, d), data, runs)

            results.append(
                BenchmarkResult(
                    size=size,
                    case_name=case.name,
                    qrydict=t_qd,
                    jmespth=t_jp_c,
                    jmwapth_compiled=t_jp_e,
                )
            )
    return (
        pl.LazyFrame(results)
        .unpivot(index=["size", "case_name"], variable_name="lib", value_name="time")
        .group_by("size", "case_name")
        .agg(
            pl.all().exclude("time"),
            pl.col("time").mul(1000).round(3).alias("time"),
            pl.col("time").rank().alias("rank"),
        )
        .sort("case_name", "size")
        .collect()
    )


if __name__ == "__main__":
    main(2).pipe(print)
