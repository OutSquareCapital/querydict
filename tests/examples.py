# type: ignore

import random
from enum import StrEnum, auto
from typing import Any, TypedDict

import factory as fc
import factory.fuzzy as fz
from faker import Faker

import querydict as qd

LIMIT = 1000


class Tags(StrEnum):
    NEW = auto()
    POPULAR = auto()
    LIMITED = auto()
    EXCLUSIVE = auto()


TAGS_LIST = [tag.value for tag in Tags]

fake = Faker()


class UserFactory(fc.base.DictFactory):
    id = fc.declarations.Sequence(lambda n: n + 1)
    name = fc.declarations.LazyAttribute(lambda _: fake.name())
    age = fc.declarations.LazyAttribute(lambda _: fake.random_int(min=18, max=65))
    active = fc.declarations.LazyAttribute(lambda _: fake.pybool())


class ProductFactory(fc.base.DictFactory):
    product_id = fc.declarations.Sequence(lambda n: n + 1)
    name = fc.declarations.LazyAttribute(lambda _: fake.word().capitalize())
    price = fc.declarations.LazyAttribute(
        lambda _: round(fake.pyfloat(min_value=5.0, max_value=100.0, right_digits=2), 2)
    )
    in_stock = fc.declarations.LazyAttribute(lambda _: fake.pybool())
    tag = fz.FuzzyChoice(TAGS_LIST)


class SaleRecordFactory(fc.DictFactory):
    order_id = fc.Sequence(lambda n: n + 1)
    customer = fc.Trait()
    product = fc.Trait()
    customer_id = fc.LazyAttribute(lambda o: o.customer["id"])
    product_id = fc.LazyAttribute(lambda o: o.product["product_id"])
    items = fc.LazyAttribute(lambda _: fake.random_int(min=1, max=10))
    amount = fc.LazyAttribute(lambda o: round(o.product["price"] * o.items, 2))
    shipped = fc.LazyAttribute(lambda _: fake.pybool())


class DataBase(TypedDict):
    users: list[dict[str, Any]]
    sales: list[dict[str, Any]]
    products: list[dict[str, Any]]
    tags: dict[str, int]


def generate_data(n: int) -> DataBase:
    product_count = 20
    users: list[dict[str, Any]] = UserFactory.build_batch(n)
    products: list[dict[str, Any]] = ProductFactory.build_batch(product_count)
    sales = [
        SaleRecordFactory.build(
            customer=random.choice(users), product=random.choice(products)
        )
        for _ in range(n * 2)
    ]

    return DataBase(
        users=users,
        sales=sales,
        products=products,
        tags=dict((tag.value, i) for i, tag in enumerate(Tags)),
    )


def main():
    data = generate_data(100)
    print(qd.field("users").project("name").search(data))


if __name__ == "__main__":
    main()
