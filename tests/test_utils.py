import pytest
import asyncio
from datetime import datetime, date, time
from decimal import Decimal

from ormax.utils import (
    OrmaxJSONEncoder, json_dumps, json_loads,
    generate_secure_token, generate_secure_password, hash_password, verify_password,
    sanitize_input, sanitize_dict,
    camel_to_snake, snake_to_camel, pluralize, singularize,
    get_class_name, get_module_name,
    deep_merge_dicts, flatten_dict, unflatten_dict, diff_dicts,
    chunk_list, unique_list, paginate_list, get_percentage,
    retry_async, timeout_async, cached_property,
    mask_string, truncate_string, humanize_bytes, humanize_number,
    create_database_url, parse_database_url, parse_sqlite_path, get_database_type_from_url,
    format_placeholders, escape_sql_identifier, format_sql_query,
    get_system_info, deprecated, requires_package,
    Timer, LRUCache, BoundedSemaphore, PerformanceMonitor,
    batch_process_async, is_model_instance, model_to_dict
)
from ormax import Model, Field, IntegerField, CharField


def test_json_encoder_and_dumps_loads():
    data = {
        "datetime": datetime(2026, 1, 1, 12, 0, 0),
        "date": date(2026, 1, 1),
        "time": time(12, 0, 0),
        "decimal": Decimal("10.5"),
        "number": 42
    }
    encoded = json_dumps(data)
    assert "2026-01-01T12:00:00" in encoded
    assert "10.5" in encoded

    decoded = json_loads(encoded)
    assert decoded["number"] == 42
    assert decoded["decimal"] == 10.5


def test_security_helpers():
    token = generate_secure_token(16)
    assert len(token) == 32  # hex representation of 16 bytes

    password = generate_secure_password(16)
    assert len(password) == 16

    hashed, salt = hash_password("secret_password")
    assert verify_password("secret_password", hashed, salt) is True
    assert verify_password("wrong_password", hashed, salt) is False


def test_sanitization():
    unsafe_str = "<script>alert('xss\x00')</script>"
    cleaned = sanitize_input(unsafe_str)
    assert "\x00" not in cleaned
    assert "&lt;script&gt;" in cleaned

    cleaned_dict = sanitize_dict({"key": "<b\x1a>"})
    assert cleaned_dict["key"] == "&lt;b&gt;"


def test_string_case_and_inflection():
    assert camel_to_snake("CamelCaseString") == "camel_case_string"
    assert snake_to_camel("snake_case_string") == "snakeCaseString"

    assert pluralize("city") == "cities"
    assert pluralize("box") == "boxes"
    assert pluralize("cat") == "cats"

    assert singularize("cities") == "city"
    assert singularize("boxes") == "box"
    assert singularize("cats") == "cat"


class DummyClass:
    pass


def test_reflection_helpers():
    obj = DummyClass()
    assert get_class_name(obj) == "DummyClass"
    assert get_class_name(DummyClass) == "DummyClass"
    assert get_module_name(obj) == __name__


def test_dict_utilities():
    d1 = {"a": 1, "b": {"c": 2}}
    d2 = {"b": {"d": 3}, "e": 4}
    merged = deep_merge_dicts(d1, d2)
    assert merged == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}

    nested = {"a": {"b": 1, "c": {"d": 2}}}
    flattened = flatten_dict(nested)
    assert flattened == {"a_b": 1, "a_c_d": 2}

    unflattened = unflatten_dict(flattened)
    assert unflattened == nested

    diff = diff_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4})
    assert diff["a"] == (1, None)
    assert diff["b"] == (2, 3)
    assert diff["c"] == (None, 4)


def test_list_and_number_utilities():
    assert list(chunk_list([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert unique_list([1, 2, 2, 3, 1]) == [1, 2, 3]

    items, pages, total = paginate_list([1, 2, 3, 4, 5, 6, 7], page=2, per_page=3)
    assert items == [4, 5, 6]
    assert pages == 3
    assert total == 7

    assert get_percentage(50, 200) == 25.0
    assert get_percentage(10, 0) == 0.0

    assert mask_string("1234567890", visible_chars=4) == "******7890"
    assert truncate_string("Hello World", max_length=8) == "Hello..."

    assert humanize_bytes(1024) == "1.0 KB"
    assert humanize_number(1500) == "1.5K"


def test_db_url_helpers():
    url = create_database_url("sqlite", "", "", "", database="test.db")
    assert url == "sqlite:///test.db"

    parsed = parse_database_url("sqlite:///test.db")
    assert parsed["scheme"] == "sqlite"
    assert parsed["database"] == "test.db"

    assert parse_sqlite_path("sqlite:///tmp/test.db") == "/tmp/test.db"
    assert parse_sqlite_path("sqlite://:memory:") == ":memory:"

    assert get_database_type_from_url("postgresql://user:pass@localhost/db") == "postgresql"
    assert format_placeholders("SELECT * FROM table WHERE a = ? AND b = ?", "postgresql://...") == "SELECT * FROM table WHERE a = $1 AND b = $2"
    assert escape_sql_identifier("user_name") == '"user_name"'
    assert format_sql_query("SELECT * FROM t WHERE a = ? AND b = ?", ("foo", 42)) == "SELECT * FROM t WHERE a = 'foo' AND b = 42"


def test_lru_cache_and_perf_monitor():
    cache = LRUCache(maxsize=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)  # Evicts "b"
    assert cache.get("b") is None
    assert cache.get("c") == 3

    monitor = PerformanceMonitor()
    monitor.record("op1", 0.1)
    monitor.record("op1", 0.3)
    stats = monitor.get_stats("op1")
    assert stats["count"] == 2
    assert stats["average"] == 0.2


@pytest.mark.asyncio
async def test_async_utilities():
    counter = 0

    @retry_async(max_attempts=3, delay=0.01)
    async def flaky():
        nonlocal counter
        counter += 1
        if counter < 2:
            raise ValueError("Failed")
        return "success"

    res = await flaky()
    assert res == "success"
    assert counter == 2

    @timeout_async(seconds=1)
    async def fast():
        return "fast"

    assert await fast() == "fast"

    async def double(x):
        return x * 2

    batch_res = await batch_process_async([1, 2, 3, 4], double, batch_size=2)
    assert batch_res == [2, 4, 6, 8]


def test_decorators_and_timer():
    @deprecated("Use new_func instead")
    def old_func():
        return 42

    assert old_func() == 42

    @requires_package("os")
    def func_requiring_os():
        return True

    assert func_requiring_os() is True

    with Timer("TestTimer"):
        pass
