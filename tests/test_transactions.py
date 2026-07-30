import pytest
import pytest_asyncio
from ormax import Database, Model
from ormax.fields import AutoField, CharField

class User(Model):
    id = AutoField()
    username = CharField(max_length=50)

@pytest_asyncio.fixture
async def db():
    database = Database("sqlite:///:memory:")
    await database.connect()
    database.register_model(User)
    await database.create_tables()
    yield database
    await database.disconnect()

@pytest.mark.asyncio
async def test_transaction_commit(db):
    async with db.transaction():
        await User.create(username="user1")
        await User.create(username="user2")
        
    assert await User.objects().count() == 2

@pytest.mark.asyncio
async def test_transaction_rollback(db):
    try:
        async with db.transaction():
            await User.create(username="user1")
            raise RuntimeError("Force rollback")
    except RuntimeError:
        pass
        
    assert await User.objects().count() == 0
