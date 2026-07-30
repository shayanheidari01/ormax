import pytest
from ormax.exceptions import DatabaseError, ValidationError, DoesNotExist


def test_exceptions_instantiation_and_inheritance():
    db_err = DatabaseError("Database connection failed")
    assert isinstance(db_err, Exception)
    assert str(db_err) == "Database connection failed"

    val_err = ValidationError("Field validation error")
    assert isinstance(val_err, Exception)
    assert str(val_err) == "Field validation error"

    dne_err = DoesNotExist("Object not found")
    assert isinstance(dne_err, Exception)
    assert str(dne_err) == "Object not found"
