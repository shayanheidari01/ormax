import pytest
import pytest_asyncio
from ormax import Database, Model, F
from ormax.fields import AutoField, CharField, IntegerField

class Account(Model):
    id = AutoField()
    holder = CharField(max_length=50)
    balance = IntegerField(default=0)

@pytest_asyncio.fixture
async def db():
    database = Database("sqlite:///:memory:")
    await database.connect()
    database.register_model(Account)
    await database.create_tables()
    yield database
    await database.disconnect()

@pytest.mark.asyncio
async def test_f_expression_update(db):
    acc = await Account.create(holder="Alice", balance=100)
    
    # Update balance using F() expression: balance = balance + 50
    await Account.objects().filter(id=acc.id).update(balance=F("balance") + 50)
    
    updated = await Account.objects().get(id=acc.id)
    assert updated.balance == 150

    # Subtract 30 using F() expression: balance = balance - 30
    await Account.objects().filter(id=acc.id).update(balance=F("balance") - 30)
    
    updated2 = await Account.objects().get(id=acc.id)
    assert updated2.balance == 120


@pytest.mark.asyncio
async def test_unary_expressions(db):
    pos_expr = +F("balance")
    neg_expr = -F("balance")
    sql_pos, _ = pos_expr.resolve_expression()
    sql_neg, _ = neg_expr.resolve_expression()
    assert sql_pos == "+(balance)"
    assert sql_neg == "-(balance)"

