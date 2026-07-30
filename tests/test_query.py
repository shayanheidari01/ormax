import pytest
import pytest_asyncio
from ormax import Database, Model
from ormax.fields import AutoField, CharField, IntegerField, FloatField, BooleanField
from ormax.exceptions import DoesNotExist

class Item(Model):
    id = AutoField()
    name = CharField(max_length=100)
    category = CharField(max_length=50)
    price = FloatField()
    in_stock = BooleanField(default=True)

@pytest_asyncio.fixture
async def db():
    database = Database("sqlite:///:memory:")
    await database.connect()
    database.register_model(Item)
    await database.create_tables()
    yield database
    await database.disconnect()

@pytest.mark.asyncio
async def test_queryset_crud_and_lookups(db):
    # Test Create
    item1 = await Item.create(name="Laptop", category="Electronics", price=1200.50, in_stock=True)
    item2 = await Item.create(name="Mouse", category="Electronics", price=25.00, in_stock=True)
    item3 = await Item.create(name="Desk", category="Furniture", price=300.00, in_stock=False)

    # Test count & all
    assert await Item.objects().count() == 3

    # Test Exact & Filter
    electronics = await Item.objects().filter(category="Electronics").all()
    assert len(electronics) == 2

    # Test Greater Than / Less Than Lookups
    expensive = await Item.objects().filter(price__gt=500.0).all()
    assert len(expensive) == 1
    assert expensive[0].name == "Laptop"

    cheaper = await Item.objects().filter(price__lte=300.0).all()
    assert len(cheaper) == 2

    # Test iContains Lookup
    m_items = await Item.objects().filter(name__icontains="lap").all()
    assert len(m_items) == 1
    assert m_items[0].name == "Laptop"

    # Test In Lookup
    cat_items = await Item.objects().filter(category__in=["Furniture", "Toys"]).all()
    assert len(cat_items) == 1

    # Test Order By & Limit
    ordered = await Item.objects().order_by("-price").limit(2).all()
    assert len(ordered) == 2
    assert ordered[0].name == "Laptop"
    assert ordered[1].name == "Desk"

@pytest.mark.asyncio
async def test_get_or_create_and_update_or_create(db):
    # Test get_or_create (create case)
    item, created = await Item.get_or_create(name="Keyboard", defaults={"category": "Electronics", "price": 45.0})
    assert created is True
    assert item.name == "Keyboard"

    # Test get_or_create (get case)
    item_existing, created_2 = await Item.get_or_create(name="Keyboard", defaults={"category": "Other", "price": 99.0})
    assert created_2 is False
    assert item_existing.id == item.id
    assert item_existing.price == 45.0

    # Test update_or_create (update case)
    item_updated, created_3 = await Item.update_or_create(name="Keyboard", defaults={"price": 50.0})
    assert created_3 is False
    assert item_updated.id == item.id
    assert item_updated.price == 50.0

@pytest.mark.asyncio
async def test_queryset_aggregate(db):
    await Item.create(name="A", category="X", price=10.0)
    await Item.create(name="B", category="X", price=20.0)
    await Item.create(name="C", category="X", price=30.0)

    agg = await Item.objects().aggregate(avg_p="AVG(price)", max_p="MAX(price)", total="SUM(price)")
    assert agg["avg_p"] == 20.0
    assert agg["max_p"] == 30.0
    assert agg["total"] == 60.0

@pytest.mark.asyncio
async def test_unregistered_model_save():
    class UnregisteredItem(Model):
        id = AutoField()
        name = CharField(max_length=100)

    item = UnregisteredItem(name="Unsaved")
    from ormax.exceptions import DatabaseError
    with pytest.raises(DatabaseError, match="Model not registered with database"):
        await item.save()

    with pytest.raises(DatabaseError, match="Model not registered with database"):
        await UnregisteredItem.bulk_create([{"name": "Unsaved"}])

    with pytest.raises(DatabaseError, match="Model not registered with database"):
        await UnregisteredItem._bulk_insert_batch([{"name": "Unsaved"}])


