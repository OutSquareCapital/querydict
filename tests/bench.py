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


@dataclass(slots=True, frozen=True)
class BenchmarkCase:
    name: str
    qd_query: qd.Query


def rand_str(k: int) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=k))


def generate_user(i: int) -> dict[str, Any]:
    return {
        "id": i,
        "name": rand_str(10),
        "age": random.randint(18, 65),
        "active": random.choice([True, False]),
        "tags": [rand_str(5) for _ in range(random.randint(1, 5))],
    }


def generate_data(n: int) -> JsonData:
    return {"users": [generate_user(i) for i in range(n)]}


BENCHMARKS: list[BenchmarkCase] = [
    BenchmarkCase(
        name="Projection simple (names)", qd_query=qd.field("users").project("name")
    ),
    BenchmarkCase(
        name="Filtre complexe (active & >30)",
        qd_query=qd.field("users").filter(
            qd.field("age").gt(30).and_(qd.field("active").eq(True)), then="name"
        ),
    ),
    BenchmarkCase(
        name="Tri (sort_by age)", qd_query=qd.field("users").sort_by(lambda e: e.age)
    ),
]

DATA_SIZES: list[int] = [100, 1_000, 10_000]


def time_query(query_func: QueryFunc, data: JsonData, number: int) -> float:
    timer = timeit.Timer(lambda: query_func(data))
    total_time = timer.timeit(number=number)
    return total_time / number


def format_results(results: list[BenchmarkResult]) -> pl.DataFrame:
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


def add_case(
    case: BenchmarkCase, size: int, runs: int, data: JsonData
) -> BenchmarkResult:
    jp_expr = case.qd_query.to_jmespath()
    jp_compiled = jmespath.compile(jp_expr)
    assert jp_compiled.search(data) == case.qd_query.search(data)

    t_qd = time_query(case.qd_query.search, data, runs)
    t_jp_c = time_query(jp_compiled.search, data, runs)

    return BenchmarkResult(size=size, case_name=case.name, qrydict=t_qd, jmespth=t_jp_c)


def main(runs: int) -> pl.DataFrame:
    print(f"Lancement des benchmarks (Runs par test: {runs})\n")
    results: list[BenchmarkResult] = []
    for size in DATA_SIZES:
        data = generate_data(size)

        for case in BENCHMARKS:
            results.append(add_case(case, size, runs, data))

    return format_results(results)


if __name__ == "__main__":
    main(2).pipe(print)
