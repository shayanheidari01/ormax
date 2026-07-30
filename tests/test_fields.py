import pytest
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal
from ormax.fields import (
    Field, CharField, IntegerField, FloatField, DecimalField, BooleanField,
    DateTimeField, DateField, EmailField, UUIDField, JSONField, ForeignKeyField,
    PositiveIntegerField, SlugField
)
from ormax.exceptions import ValidationError

def test_char_field_validation():
    field = CharField(max_length=10, min_length=2)
    assert field.validate("hello") == "hello"
    
    with pytest.raises(ValidationError):
        field.validate("a")  # too short
        
    with pytest.raises(ValidationError):
        field.validate("this string is too long")  # too long


def test_integer_and_float_fields():
    int_field = IntegerField(min_value=0, max_value=100)
    assert int_field.validate(50) == 50
    with pytest.raises(ValidationError):
        int_field.validate(-1)
    with pytest.raises(ValidationError):
        int_field.validate(101)

    pos_int_field = PositiveIntegerField()
    assert pos_int_field.validate(10) == 10
    with pytest.raises(ValidationError):
        pos_int_field.validate(-5)

    float_field = FloatField()
    assert float_field.validate(12.34) == 12.34
    assert float_field.to_python("12.34") == 12.34


def test_decimal_field_precision():
    field = DecimalField(max_digits=10, decimal_places=2)
    val = Decimal("123.45")
    assert field.to_python(123.45) == Decimal("123.45")
    assert field.to_database(val) == "123.45"  # Exact precision preserved as string


def test_boolean_field_validation():
    field = BooleanField()
    assert field.validate("true") is True
    assert field.validate("false") is False
    assert field.validate(1) is True
    assert field.validate(0) is False


def test_email_field_validation():
    field = EmailField()
    assert field.validate("user@example.com") == "user@example.com"
    
    with pytest.raises(ValidationError):
        field.validate("invalid-email")


def test_slug_field_validation():
    field = SlugField()
    assert field.validate("hello-world_123") == "hello-world_123"
    with pytest.raises(ValidationError):
        field.validate("Hello World!")


def test_datetime_and_date_fields():
    dt_field = DateTimeField(auto_now=False, auto_now_add=False, use_tz=False)
    now = datetime.now()
    assert dt_field.validate(now) == now
    assert isinstance(dt_field.to_python(now.isoformat()), datetime)

    dt_field_tz = DateTimeField(auto_now=False, auto_now_add=False, use_tz=True)
    validated_tz = dt_field_tz.validate(now)
    assert validated_tz.tzinfo == timezone.utc

    d_field = DateField()
    today = date.today()
    assert d_field.validate(today) == today
    assert d_field.to_python(today.isoformat()) == today


def test_uuid_field_validation():
    field = UUIDField()
    u = uuid.uuid4()
    assert field.validate(u) == u
    assert field.to_database(u) == str(u)


def test_json_field():
    field = JSONField()
    data = {"key": "value", "numbers": [1, 2, 3]}
    db_val = field.to_database(data)
    assert isinstance(db_val, str)
    assert field.to_python(db_val) == data
