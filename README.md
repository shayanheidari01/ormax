# Ormax 1.4

Ormax is a lightweight asynchronous ORM for Python 3.9+ with a Django-style query API, model validation, foreign-key relationships, nested transactions, and database adapters for SQLite, PostgreSQL, MySQL/MariaDB, Microsoft SQL Server, Oracle, and Amazon Aurora.

> **Project status:** Beta. SQLite is covered by the bundled test suite. Other adapters require their database servers and drivers and should be integration-tested in the target environment before production use.

## Highlights

- Async model CRUD and chainable `QuerySet` operations
- SQLite support with an internal standard-library fallback; `aiosqlite` remains optional
- Connection pooling for pool-based backends
- Nested transactions through savepoints
- Task-safe transaction connection pinning
- Forward and reverse foreign-key loading
- Callable and mutable-safe defaults
- Abstract model inheritance and `class Meta` configuration
- Field projection with `only()`, `defer()`, `values()`, and `values_list()`
- Atomic updates with `F()` expressions
- Database index creation from `index=True` and backend-specific `using_index()` hints
- Inline type information marked with `py.typed`

## Installation

```bash
pip install ormax
```

Install the driver extra for the selected backend:

```bash
pip install "ormax[sqlite]"      # optional: Ormax has a built-in SQLite fallback
pip install "ormax[postgresql]"
pip install "ormax[mysql]"
pip install "ormax[mssql]"
pip install "ormax[oracle]"
pip install "ormax[all]"
```

For local development:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest -q
```

## Quick start

```python
import asyncio

from ormax import CharField, Database, ForeignKeyField, Model


class Author(Model):
    name = CharField(max_length=100, index=True)

    class Meta:
        table_name = "authors"


class Book(Model):
    title = CharField(max_length=200)
    author = ForeignKeyField(Author, related_name="books", null=False)


async def main():
    db = Database("sqlite:///example.db", models=[Author, Book])

    async with db:
        await db.create_tables()

        author = await Author.create(name="Ursula K. Le Guin")
        await Book.create(title="A Wizard of Earthsea", author=author)

        book = await Book.objects().select_related("author").get()
        print(book.title, book.author.name)


asyncio.run(main())
```

## Models and fields

Ormax adds an auto-incrementing `id` field when a model has no explicit primary key.

```python
import uuid

from ormax import (
    BooleanField,
    CharField,
    DateTimeField,
    JSONField,
    Model,
    UUIDField,
)


class Timestamped(Model):
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Article(Timestamped):
    token = UUIDField(default=uuid.uuid4, unique=True)
    title = CharField(max_length=200, index=True)
    metadata = JSONField(default=dict)
    published = BooleanField(default=False)

    class Meta:
        table_name = "articles"
```

Callable defaults are invoked per instance. Mutable defaults are deep-copied, so separate model instances do not share the same list or dictionary.

Supported public field classes:

- Text: `CharField`, `TextField`, `EmailField`, `URLField`, `SlugField`
- Numeric: `IntegerField`, `BigIntegerField`, `SmallIntegerField`, `PositiveIntegerField`, `PositiveSmallIntegerField`, `FloatField`, `DecimalField`
- Identity: `AutoField`, `BigAutoField`, `SmallAutoField`, `UUIDField`
- Date/time: `DateTimeField`, `DateField`, `TimeField`
- Structured/special: `BooleanField`, `JSONField`, `BinaryField`, `IPAddressField`
- Relationship: `ForeignKeyField`

Both `null=` and `nullable=` are accepted. Prefer `null=` in new code for brevity.

## Query API

```python
# Basic retrieval
all_books = await Book.objects().all()
first_book = await Book.objects().order_by("title").first()
book = await Book.objects().get(id=1)

# Lookups
results = await Book.objects().filter(title__icontains="earth").all()
results = await Book.objects().exclude(id__in=[1, 2]).all()
results = await Book.objects().filter(id__range=(10, 20)).all()

# Projection
rows = await Book.objects().values("id", "title")
titles = await Book.objects().values_list("title", flat=True)
partial = await Book.objects().only("id", "title").all()
without_title = await Book.objects().defer("title").all()

# Pagination and metadata
page = await Book.objects().order_by("id").offset(20).limit(10).all()
count = await Book.objects().count()
exists = await Book.objects().filter(title="Missing").exists()

# Updates and deletes return the affected-row count where the driver exposes it
updated = await Book.objects().filter(id=1).update(title="New title")
deleted = await Book.objects().filter(id=1).delete()
```

Supported lookup suffixes include `exact`, `iexact`, `contains`, `icontains`, `gt`, `gte`, `lt`, `lte`, `in`, `startswith`, `istartswith`, `endswith`, `iendswith`, `isnull`, and `range`.

`get()` raises the public `DoesNotExist` or `MultipleObjectsReturned` exceptions.

### Atomic expressions

```python
from ormax import F

await Account.objects().filter(id=1).update(balance=F("balance") - 10)
```

### Query timeout

```python
rows = await Book.objects().timeout(2.5).all()
```

The timeout is enforced by `asyncio.wait_for` around the driver operation.

## Relationships

```python
class Author(Model):
    name = CharField(max_length=100)


class Book(Model):
    title = CharField(max_length=200)
    author = ForeignKeyField(
        Author,
        related_name="books",
        null=False,
        on_delete="CASCADE",
    )
```

Forward loading:

```python
book = await Book.objects().get(id=1)
author = await book.get_related("author")

book = await Book.objects().select_related("author").get(id=1)
print(book.author.name)
```

Reverse loading:

```python
author = await Author.objects().prefetch_related("books").get(id=1)
print(len(author.books))
books = await author.books.all()
```

`select_related()` currently performs batched eager loading rather than a SQL `JOIN`. `prefetch_related()` loads reverse relations in separate batched queries.

## Transactions

```python
async with db.transaction():
    author = await Author.create(name="Octavia E. Butler")

    try:
        async with db.transaction():
            await Book.create(title="Temporary", author=author)
            raise RuntimeError("roll back nested work")
    except RuntimeError:
        pass

    await Book.create(title="Kindred", author=author)
```

The outer transaction commits or rolls back as one unit. Nested contexts use savepoints. Pool-based adapters pin all statements in a transaction to the same physical connection.

SQLite uses one connection and serializes concurrent outer transactions. Do not share an active transaction context with a child `asyncio` task; Ormax raises `DatabaseError` rather than allowing ambiguous cross-task transaction behavior.

## Indexes

Fields declared with `index=True` or `db_index=True` create a non-unique index during `create_tables()`:

```python
class Event(Model):
    external_id = CharField(max_length=64, index=True)
```

The generated name is `idx_<table>_<field>`. An explicit hint can be requested on supported backends:

```python
rows = await Event.objects().using_index("idx_event_external_id").all()
```

Hints are emitted for SQLite, MySQL/MariaDB/Aurora, and MSSQL. PostgreSQL rejects explicit hints because it does not support this syntax natively.

## Raw SQL

Use `?` placeholders in application code; Ormax converts them for the selected adapter:

```python
row = await db.fetch_one(
    "SELECT * FROM books WHERE title = ?",
    ("Kindred",),
)
```

`annotate()`, `having()`, `extra()`, and `raw()` accept SQL fragments. They are advanced APIs: never insert untrusted user input into those fragments. Bind values through parameters.

## Database URLs

```text
sqlite:///path/to/database.db
sqlite:///:memory:
postgresql://user:password@host:5432/database
mysql://user:password@host:3306/database
mariadb://user:password@host:3306/database
mssql://<aioodbc DSN or connection string>
microsoft://<aioodbc DSN or connection string>
oracle://user:password@host:1521/service
aurora://user:password@host:3306/database
```

## Lifecycle hooks

Hooks may be synchronous or asynchronous:

```python
class AuditRecord(Model):
    message = CharField(max_length=255)

    def pre_save(self):
        self.message = self.message.strip()

    async def post_save(self, created: bool):
        if created:
            await publish_event(self.pk)
```

## Current limitations

- Schema migrations are not included; `create_tables()` is a creation helper, not a migration engine.
- `union()`, `intersection()`, and `difference()` are reserved but not implemented.
- `select_related()` uses batched queries, not SQL joins.
- Pool-backed adapters are structurally covered by unit tests but require live integration testing against their database servers.
- Raw SQL-fragment APIs require trusted SQL.

## Development and release checks

```bash
python -m compileall -q ormax tests
pytest -q
python -m build
python -m twine check dist/*
```

See [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md), and the generated static documentation under [`docs/`](docs/).

## License

MIT. See [LICENSE](LICENSE).
