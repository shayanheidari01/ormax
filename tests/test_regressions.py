import uuid

import pytest

from ormax import (
    AutoField,
    BooleanField,
    CharField,
    Database,
    ForeignKeyField,
    IntegerField,
    JSONField,
    Model,
    MultipleObjectsReturned,
    UUIDField,
)


@pytest.mark.asyncio
async def test_sqlite_works_without_third_party_driver():
    db = Database("sqlite:///:memory:")
    await db.connect()
    try:
        assert db.is_connected
        assert db.connection.aiosqlite is not None
    finally:
        await db.disconnect()


def test_meta_inheritance_callable_and_mutable_defaults():
    class Timestamped(Model):
        tags = JSONField(default=list)

        class Meta:
            abstract = True

    class Article(Timestamped):
        title = CharField(max_length=50)

        class Meta:
            table_name = "articles"

    assert Article.get_table_name() == "articles"
    assert "tags" in Article._fields
    assert Article._meta.get("abstract") is not True

    first = Article(title="A")
    second = Article(title="B")
    first.tags.append("news")
    assert second.tags == []

    with pytest.raises(TypeError, match="Unknown field"):
        Article(title="A", typo=True)


@pytest.mark.asyncio
async def test_field_round_trip_values_and_projection():
    db = Database("sqlite:///:memory:")

    class Record(Model):
        id = AutoField()
        token = UUIDField(default=uuid.uuid4)
        payload = JSONField(default=dict)
        enabled = BooleanField(default=True)
        name = CharField(max_length=30)

        class Meta:
            table_name = "records"

    db.register_model(Record)
    await db.connect()
    await db.create_tables()
    try:
        created = await Record.create(name="one", payload={"x": 1})
        fetched = await Record.objects().get(id=created.id)
        assert isinstance(fetched.token, uuid.UUID)
        assert fetched.payload == {"x": 1}
        assert fetched.enabled is True

        queryset = Record.objects().order_by("id")
        names = await queryset.values_list("name", flat=True)
        assert names == ["one"]
        assert (await queryset.all())[0].payload == {"x": 1}

        projected = await Record.objects().only("id", "name").get(id=created.id)
        assert projected.id == created.id
        assert projected.name == "one"
        assert "payload" not in projected._data

        deferred = await Record.objects().defer("payload").get(id=created.id)
        assert "payload" not in deferred._data
    finally:
        await db.disconnect()


@pytest.mark.asyncio
async def test_relationship_loading_and_model_instance_filters():
    db = Database("sqlite:///:memory:")

    class Author(Model):
        name = CharField(max_length=50)

    class Book(Model):
        title = CharField(max_length=100)
        author = ForeignKeyField(Author, related_name="books", null=False)

    db.register_model(Author, Book)
    await db.connect()
    await db.create_tables()
    try:
        author = await Author.create(name="Ada")
        await Book.create(title="Notes", author=author)

        direct = await Book.objects().filter(author=author).get()
        assert direct.title == "Notes"
        loaded_author = await direct.get_related("author")
        assert loaded_author.name == "Ada"

        eager = await Book.objects().select_related("author").get()
        assert eager.author.name == "Ada"

        prefetched = await Author.objects().prefetch_related("books").get()
        assert len(prefetched.books) == 1
        assert (await prefetched.books.all())[0].title == "Notes"
    finally:
        await db.disconnect()


@pytest.mark.asyncio
async def test_sync_lifecycle_hooks_and_multiple_objects_error():
    db = Database("sqlite:///:memory:")
    events = []

    class Item(Model):
        name = CharField(max_length=20)

        def pre_save(self):
            events.append("pre")

        def post_save(self, created):
            events.append(f"post:{created}")

    db.register_model(Item)
    await db.connect()
    await db.create_tables()
    try:
        await Item.create(name="a")
        await Item.create(name="b")
        assert events[:2] == ["pre", "post:True"]
        with pytest.raises(MultipleObjectsReturned):
            await Item.objects().get()
    finally:
        await db.disconnect()


@pytest.mark.asyncio
async def test_nested_transaction_savepoint_rollback():
    db = Database("sqlite:///:memory:")

    class Entry(Model):
        value = IntegerField()

    db.register_model(Entry)
    await db.connect()
    await db.create_tables()
    try:
        async with db.transaction():
            await Entry.create(value=1)
            with pytest.raises(RuntimeError):
                async with db.transaction():
                    await Entry.create(value=2)
                    raise RuntimeError("rollback nested")
            await Entry.create(value=3)

        assert await Entry.objects().values_list("value", flat=True) == [1, 3]
    finally:
        await db.disconnect()

@pytest.mark.asyncio
async def test_concurrent_sqlite_transactions_are_serialized():
    import asyncio

    db = Database("sqlite:///:memory:")

    class Counter(Model):
        value = IntegerField()

    db.register_model(Counter)
    await db.connect()
    await db.create_tables()
    try:
        async def worker(value):
            async with db.transaction():
                await Counter.create(value=value)
                await asyncio.sleep(0)

        await asyncio.gather(worker(1), worker(2))
        assert sorted(await Counter.objects().values_list("value", flat=True)) == [1, 2]
    finally:
        await db.disconnect()


@pytest.mark.asyncio
async def test_transaction_context_rejects_child_task_inheritance():
    import asyncio

    db = Database("sqlite:///:memory:")
    await db.connect()
    try:
        async with db.transaction():
            task = asyncio.create_task(db.fetch_one("SELECT 1 AS value"))
            with pytest.raises(Exception, match="child asyncio task"):
                await task
    finally:
        await db.disconnect()

@pytest.mark.asyncio
async def test_public_does_not_exist_and_sqlite_index_hint():
    from ormax import DoesNotExist

    db = Database("sqlite:///:memory:")

    class IndexedItem(Model):
        name = CharField(max_length=20, index=True)

    db.register_model(IndexedItem)
    await db.connect()
    await db.create_tables()
    try:
        await IndexedItem.create(name="a")
        with pytest.raises(DoesNotExist):
            await IndexedItem.objects().get(name="missing")

        rows = await IndexedItem.objects().using_index("idx_indexeditem_name").all()
        assert [row.name for row in rows] == ["a"]

        with pytest.raises(ValueError):
            IndexedItem.objects().using_index("bad; DROP TABLE")
        with pytest.raises(ValueError):
            IndexedItem.objects().timeout(0)
    finally:
        await db.disconnect()

@pytest.mark.asyncio
async def test_postgresql_transaction_pins_one_pool_connection():
    from ormax.core import PostgreSQLConnection

    class FakeConnection:
        def __init__(self, name):
            self.name = name
            self.calls = []

        async def execute(self, query, *params):
            self.calls.append(("execute", query, params))
            return "OK"

        async def fetchrow(self, query, *params):
            self.calls.append(("fetchrow", query, params))
            return {"value": 1}

        async def fetch(self, query, *params):
            self.calls.append(("fetch", query, params))
            return [{"value": 1}]

    class AcquireContext:
        def __init__(self, pool):
            self.pool = pool
            self.connection = None

        def __await__(self):
            async def acquire():
                self.connection = self.pool.connections[0]
                self.pool.acquired.append(self.connection.name)
                return self.connection

            return acquire().__await__()

        async def __aenter__(self):
            return await self

        async def __aexit__(self, *_):
            await self.pool.release(self.connection)

    class FakePool:
        def __init__(self):
            self.connections = [FakeConnection("primary"), FakeConnection("secondary")]
            self.acquired = []
            self.released = []

        def acquire(self):
            return AcquireContext(self)

        async def release(self, connection):
            self.released.append(connection.name)

    db = Database("postgresql://user:pass@localhost/db")
    connection = PostgreSQLConnection(db.connection_string)
    pool = FakePool()
    connection.pool = pool
    connection.database = db
    db.connection = connection
    db._connected = True

    async with db.transaction():
        await db.execute("UPDATE example SET value = ?", (2,))
        assert (await db.fetch_one("SELECT 1 AS value"))["value"] == 1
        assert (await db.fetch_all("SELECT 1 AS value"))[0]["value"] == 1

    assert pool.acquired == ["primary"]
    assert pool.released == ["primary"]
    primary_calls = [call[1] for call in pool.connections[0].calls]
    assert primary_calls[0] == "BEGIN"
    assert primary_calls[-1] == "COMMIT"
    assert pool.connections[1].calls == []

@pytest.mark.asyncio
async def test_backend_routing_and_portable_ddl_generation():
    from ormax.core import MySQLConnection

    assert isinstance(Database("mariadb://user:pass@localhost/db").connection, MySQLConnection)

    class RecordingConnection:
        def __init__(self):
            self.queries = []

        async def execute(self, query, params=None):
            self.queries.append(query)
            return 0

    class DDLModel(Model):
        code = CharField(max_length=20, index=True)

    mssql = Database("microsoft://Driver=Fake")
    mssql.register_model(DDLModel)
    mssql.connection = RecordingConnection()
    await mssql._create_table_without_foreign_keys(DDLModel)
    assert "OBJECT_ID" in mssql.connection.queries[0]
    await mssql.drop_table_by_name(DDLModel.get_table_name())
    assert "OBJECT_ID" in mssql.connection.queries[-1]
    assert "CASCADE" not in mssql.connection.queries[-1]

    oracle = Database("oracle://user:pass@localhost/service")
    oracle.register_model(DDLModel)
    oracle.connection = RecordingConnection()
    await oracle._create_table_without_foreign_keys(DDLModel)
    assert "EXECUTE IMMEDIATE" in oracle.connection.queries[0]
    assert "GENERATED BY DEFAULT AS IDENTITY" in oracle.connection.queries[0]
    await oracle.drop_table_by_name(DDLModel.get_table_name())
    assert "SQLCODE != -942" in oracle.connection.queries[-1]

@pytest.mark.asyncio
async def test_official_oracle_async_adapter_and_bind_conversion(monkeypatch):
    import sys
    import types

    from ormax.core import OracleConnection

    class FakeCursor:
        def __init__(self):
            self.description = [("VALUE",)]
            self.rowcount = 1
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        async def execute(self, query, params):
            self.calls.append((query, params))

        async def fetchone(self):
            return (7,)

        async def fetchall(self):
            return [(7,), (8,)]

    class FakeOracleConnection:
        def __init__(self):
            self.cursor_instance = FakeCursor()
            self.commits = 0
            self.rollbacks = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def cursor(self):
            return self.cursor_instance

        async def commit(self):
            self.commits += 1

        async def rollback(self):
            self.rollbacks += 1

    class FakeOraclePool:
        def __init__(self):
            self.connection = FakeOracleConnection()
            self.releases = 0
            self.closed = False

        def acquire(self):
            return self.connection

        async def release(self, connection):
            assert connection is self.connection
            self.releases += 1

        async def close(self):
            self.closed = True

    pool = FakeOraclePool()
    module = types.SimpleNamespace(create_pool_async=lambda **kwargs: pool)
    monkeypatch.setitem(sys.modules, "oracledb", module)

    connection = OracleConnection("oracle://user:pass@localhost:1521/service")
    await connection.connect()
    assert connection.pool is pool

    row = await connection.fetch_one("SELECT ? AS value FROM dual", (7,))
    assert row == {"VALUE": 7}
    assert pool.connection.cursor_instance.calls[-1] == (
        "SELECT :1 AS value FROM dual",
        (7,),
    )

    await connection.execute("UPDATE demo SET value = ?", (6,))
    assert pool.connection.commits == 1

    await connection.start_transaction()
    await connection.execute("UPDATE demo SET value = ?", (8,))
    await connection.commit_transaction()
    assert pool.releases == 1
    assert pool.connection.commits == 2  # one auto-commit before tx, one tx commit

    await connection.disconnect()
    assert pool.closed is True
