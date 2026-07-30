import pytest
from typing import List
from ormax import Database, Model, IntegerField, CharField, ValidationError, DoesNotExist


@pytest.mark.asyncio
async def test_database_connection_and_raw_query(tmp_path):
    db_file = tmp_path / "core_test.db"
    db = Database(f"sqlite:///{db_file}")
    await db.connect()
    assert db.is_connected is True

    await db.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
    await db.execute("INSERT INTO test_table (name) VALUES (?)", ("Alice",))
    
    rows = await db.fetch_all("SELECT * FROM test_table")
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"

    row = await db.fetch_one("SELECT * FROM test_table WHERE id = ?", (1,))
    assert row is not None
    assert row["name"] == "Alice"

    await db.disconnect()
    assert db.is_connected is False


@pytest.mark.asyncio
async def test_model_signals_and_lifecycle(tmp_path):
    db_file = tmp_path / "signals_test.db"
    db = Database(f"sqlite:///{db_file}")

    events: List[str] = []

    class SignalModel(Model):
        name = CharField(max_length=50)

        class Meta:
            table_name = "signal_models"

        async def pre_save(self):
            events.append(f"pre_save:{self.name}")

        async def post_save(self, created: bool):
            events.append(f"post_save:{created}:{self.name}")

        async def pre_delete(self):
            events.append(f"pre_delete:{self.name}")

        async def post_delete(self):
            events.append(f"post_delete:{self.name}")

    db.register_model(SignalModel)
    await db.connect()
    await SignalModel.create_table()

    obj = SignalModel(name="Item1")
    await obj.save()
    assert events == ["pre_save:Item1", "post_save:True:Item1"]

    events.clear()
    obj.name = "Item1_Updated"
    await obj.save()
    assert events == ["pre_save:Item1_Updated", "post_save:False:Item1_Updated"]

    events.clear()
    await obj.delete()
    assert events == ["pre_delete:Item1_Updated", "post_delete:Item1_Updated"]

    await db.disconnect()


@pytest.mark.asyncio
async def test_custom_primary_key(tmp_path):
    db_file = tmp_path / "custom_pk.db"
    db = Database(f"sqlite:///{db_file}")

    class CodeItem(Model):
        code = CharField(max_length=10, primary_key=True)
        title = CharField(max_length=50)

        class Meta:
            table_name = "code_items"

    db.register_model(CodeItem)
    await db.connect()
    await CodeItem.create_table()

    item = await CodeItem.objects().create(code="ABC", title="Alphabet")
    assert item.pk == "ABC"

    fetched = await CodeItem.objects().get(code="ABC")
    assert fetched.title == "Alphabet"

    await db.disconnect()


@pytest.mark.asyncio
async def test_model_clean_and_validation(tmp_path):
    db_file = tmp_path / "clean_test.db"
    db = Database(f"sqlite:///{db_file}")

    class ValidatedModel(Model):
        age = IntegerField()

        class Meta:
            table_name = "validated_models"

        def clean(self):
            if self.age < 0:
                raise ValidationError("Age cannot be negative")

    db.register_model(ValidatedModel)
    await db.connect()
    await ValidatedModel.create_table()

    valid_item = ValidatedModel(age=25)
    valid_item.full_clean()  # Should not raise

    invalid_item = ValidatedModel(age=-5)
    with pytest.raises(ValidationError, match="Age cannot be negative"):
        invalid_item.full_clean()

    await db.disconnect()
