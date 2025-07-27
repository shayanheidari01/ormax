# database_orm/fields.py
import re
from typing import Any, Optional, Union
from datetime import datetime
from .exceptions import ValidationError


class Field:
    def __init__(self, 
                 field_type: str = "VARCHAR(255)",
                 primary_key: bool = False,
                 auto_increment: bool = False,
                 nullable: bool = True,
                 default: Any = None,
                 unique: bool = False,
                 index: bool = False,
                 foreign_key: Optional[str] = None):
        self.field_type = field_type
        self.primary_key = primary_key
        self.auto_increment = auto_increment
        self.nullable = nullable
        self.default = default
        self.unique = unique
        self.index = index
        self.foreign_key = foreign_key
    
    def get_sql_type(self) -> str:
        return self.field_type
    
    def get_default_sql(self) -> str:
        if self.default is None:
            return "NULL"
        elif isinstance(self.default, str):
            return f"'{self.default}'"
        else:
            return str(self.default)
    
    def validate(self, value: Any) -> Any:
        if value is None and not self.nullable:
            if not (self.primary_key and self.auto_increment):
                raise ValidationError("Field cannot be null")
        return value


class CharField(Field):
    def __init__(self, max_length: int = 255, **kwargs):
        super().__init__(field_type=f"VARCHAR({max_length})", **kwargs)
        self.max_length = max_length
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if not isinstance(value, str):
                value = str(value)
            if len(value) > self.max_length:
                raise ValidationError(f"Value exceeds maximum length of {self.max_length}")
        return value


class TextField(Field):
    def __init__(self, **kwargs):
        super().__init__(field_type="TEXT", **kwargs)


class IntegerField(Field):
    def __init__(self, **kwargs):
        # Handle auto_increment parameter properly
        auto_increment = kwargs.pop('auto_increment', False)
        super().__init__(field_type="INTEGER", auto_increment=auto_increment, **kwargs)
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ValidationError("Value must be an integer")
        return value


class BigIntegerField(Field):
    def __init__(self, **kwargs):
        auto_increment = kwargs.pop('auto_increment', False)
        super().__init__(field_type="BIGINT", auto_increment=auto_increment, **kwargs)
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ValidationError("Value must be an integer")
        return value


class FloatField(Field):
    def __init__(self, **kwargs):
        super().__init__(field_type="FLOAT", **kwargs)
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ValidationError("Value must be a float")
        return value


class BooleanField(Field):
    def __init__(self, **kwargs):
        super().__init__(field_type="BOOLEAN", **kwargs)
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            elif isinstance(value, (int, float)):
                return bool(value)
            else:
                raise ValidationError("Value must be a boolean")
        return value


class DateTimeField(Field):
    def __init__(self, auto_now: bool = False, auto_now_add: bool = False, **kwargs):
        super().__init__(field_type="DATETIME", **kwargs)
        self.auto_now = auto_now
        self.auto_now_add = auto_now_add
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if isinstance(value, str):
                try:
                    value = datetime.fromisoformat(value)
                except ValueError:
                    raise ValidationError("Invalid datetime format")
            elif not isinstance(value, datetime):
                raise ValidationError("Value must be a datetime")
        return value


class DateField(Field):
    def __init__(self, **kwargs):
        super().__init__(field_type="DATE", **kwargs)
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if isinstance(value, str):
                try:
                    datetime.strptime(value, '%Y-%m-%d')
                except ValueError:
                    raise ValidationError("Invalid date format, expected YYYY-MM-DD")
            elif hasattr(value, 'date'):
                value = value.date()
        return value


class JSONField(Field):
    def __init__(self, **kwargs):
        super().__init__(field_type="JSON", **kwargs)
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if isinstance(value, str):
                try:
                    import json
                    json.loads(value)
                except json.JSONDecodeError:
                    raise ValidationError("Invalid JSON string")
            elif not isinstance(value, (dict, list)):
                raise ValidationError("Value must be a dict, list, or JSON string")
        return value


class EmailField(CharField):
    def __init__(self, **kwargs):
        super().__init__(max_length=254, **kwargs)
        self.email_regex = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if not self.email_regex.match(value):
                raise ValidationError("Invalid email format")
        return value


class URLField(CharField):
    def __init__(self, **kwargs):
        super().__init__(max_length=2048, **kwargs)
        self.url_regex = re.compile(
            r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
        )
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if not self.url_regex.match(value):
                raise ValidationError("Invalid URL format")
        return value