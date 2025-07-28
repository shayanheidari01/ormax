# ormax/fields.py (نسخه اصلاح شده)
import re
import uuid
from typing import Any, Optional, Union, Callable, Dict, List
from datetime import datetime, date
from decimal import Decimal
from ipaddress import IPv4Address, IPv6Address
from .exceptions import ValidationError


class Field:
    """
    Base field class for all ORM fields
    """
    __slots__ = ('field_type', 'primary_key', 'auto_increment', 'nullable', 
                 'default', 'unique', 'index', 'foreign_key', 'choices',
                 'help_text', 'validators', 'db_column', 'db_index')
    
    def __init__(self, 
                 field_type: str = "VARCHAR(255)",
                 primary_key: bool = False,
                 auto_increment: bool = False,
                 nullable: bool = True,
                 default: Any = None,
                 unique: bool = False,
                 index: bool = False,
                 foreign_key: Optional[str] = None,
                 choices: Optional[List[tuple]] = None,
                 help_text: Optional[str] = None,
                 validators: Optional[List[Callable]] = None,
                 db_column: Optional[str] = None,
                 db_index: bool = False):
        self.field_type = field_type
        self.primary_key = primary_key
        self.auto_increment = auto_increment
        self.nullable = nullable
        self.default = default
        self.unique = unique
        self.index = index or db_index
        self.foreign_key = foreign_key
        self.choices = choices
        self.help_text = help_text
        self.validators = validators or []
        self.db_column = db_column
        self.db_index = db_index
    
    def get_sql_type(self) -> str:
        """Get SQL type for this field"""
        return self.field_type
    
    def get_default_sql(self) -> str:
        """Get SQL representation of default value"""
        if self.default is None:
            return "NULL"
        elif isinstance(self.default, str):
            return f"'{self.default}'"
        elif isinstance(self.default, bool):
            return "TRUE" if self.default else "FALSE"
        else:
            return str(self.default)
    
    def validate(self, value: Any) -> Any:
        """Validate field value"""
        if value is None and not self.nullable:
            if not (self.primary_key and self.auto_increment):
                raise ValidationError("Field cannot be null")
        
        # Check choices
        if self.choices and value is not None:
            valid_choices = [choice[0] for choice in self.choices]
            if value not in valid_choices:
                raise ValidationError(f"Value '{value}' is not in valid choices: {valid_choices}")
        
        # Run custom validators
        for validator in self.validators:
            try:
                validator(value)
            except Exception as e:
                raise ValidationError(str(e))
        
        return value
    
    def clean(self, value: Any) -> Any:
        """Clean and normalize field value"""
        return self.validate(value)
    
    def to_python(self, value: Any) -> Any:
        """Convert database value to Python value"""
        return value
    
    def to_database(self, value: Any) -> Any:
        """Convert Python value to database value"""
        return value


class CharField(Field):
    """
    Character field with maximum length
    """
    __slots__ = ('max_length', 'min_length')
    
    def __init__(self, max_length: int = 255, min_length: int = 0, **kwargs):
        super().__init__(field_type=f"VARCHAR({max_length})", **kwargs)
        self.max_length = max_length
        self.min_length = min_length
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if not isinstance(value, str):
                try:
                    value = str(value)
                except Exception:
                    raise ValidationError("Value cannot be converted to string")
            
            if len(value) > self.max_length:
                raise ValidationError(f"Value exceeds maximum length of {self.max_length}")
            
            if len(value) < self.min_length:
                raise ValidationError(f"Value is shorter than minimum length of {self.min_length}")
        
        return value


class TextField(Field):
    """
    Long text field
    """
    def __init__(self, **kwargs):
        super().__init__(field_type="TEXT", **kwargs)


class IntegerField(Field):
    """
    Integer field
    """
    __slots__ = ('min_value', 'max_value')
    
    def __init__(self, min_value: Optional[int] = None, max_value: Optional[int] = None, **kwargs):
        auto_increment = kwargs.pop('auto_increment', False)
        super().__init__(field_type="INTEGER", auto_increment=auto_increment, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ValidationError("Value must be an integer")
            
            if self.min_value is not None and value < self.min_value:
                raise ValidationError(f"Value must be greater than or equal to {self.min_value}")
            
            if self.max_value is not None and value > self.max_value:
                raise ValidationError(f"Value must be less than or equal to {self.max_value}")
        
        return value


class BigIntegerField(Field):
    """
    Big integer field
    """
    __slots__ = ('min_value', 'max_value')
    
    def __init__(self, min_value: Optional[int] = None, max_value: Optional[int] = None, **kwargs):
        auto_increment = kwargs.pop('auto_increment', False)
        super().__init__(field_type="BIGINT", auto_increment=auto_increment, **kwargs)
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ValidationError("Value must be an integer")
            
            if self.min_value is not None and value < self.min_value:
                raise ValidationError(f"Value must be greater than or equal to {self.min_value}")
            
            if self.max_value is not None and value > self.max_value:
                raise ValidationError(f"Value must be less than or equal to {self.max_value}")
        
        return value


class SmallIntegerField(Field):
    """
    Small integer field (16-bit)
    """
    __slots__ = ('min_value', 'max_value')
    
    def __init__(self, min_value: Optional[int] = None, max_value: Optional[int] = None, **kwargs):
        super().__init__(field_type="SMALLINT", **kwargs)
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ValidationError("Value must be an integer")
            
            if self.min_value is not None and value < self.min_value:
                raise ValidationError(f"Value must be greater than or equal to {self.min_value}")
            
            if self.max_value is not None and value > self.max_value:
                raise ValidationError(f"Value must be less than or equal to {self.max_value}")
        
        return value


class FloatField(Field):
    """
    Floating point number field
    """
    __slots__ = ('min_value', 'max_value')
    
    def __init__(self, min_value: Optional[float] = None, max_value: Optional[float] = None, **kwargs):
        super().__init__(field_type="FLOAT", **kwargs)
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ValidationError("Value must be a float")
            
            if self.min_value is not None and value < self.min_value:
                raise ValidationError(f"Value must be greater than or equal to {self.min_value}")
            
            if self.max_value is not None and value > self.max_value:
                raise ValidationError(f"Value must be less than or equal to {self.max_value}")
        
        return value


class DecimalField(Field):
    """
    Decimal field with precision and scale
    """
    __slots__ = ('max_digits', 'decimal_places', 'min_value', 'max_value')
    
    def __init__(self, max_digits: int = 10, decimal_places: int = 2, 
                 min_value: Optional[Union[Decimal, float, str]] = None,
                 max_value: Optional[Union[Decimal, float, str]] = None, **kwargs):
        super().__init__(field_type=f"DECIMAL({max_digits},{decimal_places})", **kwargs)
        self.max_digits = max_digits
        self.decimal_places = decimal_places
        self.min_value = Decimal(str(min_value)) if min_value is not None else None
        self.max_value = Decimal(str(max_value)) if max_value is not None else None
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            try:
                if not isinstance(value, Decimal):
                    value = Decimal(str(value))
            except Exception:
                raise ValidationError("Value must be a decimal number")
            
            # Check precision
            if value.as_tuple().exponent < -self.decimal_places:
                raise ValidationError(f"Value has more decimal places than allowed ({self.decimal_places})")
            
            if len(str(value).replace('.', '').replace('-', '')) > self.max_digits:
                raise ValidationError(f"Value has more digits than allowed ({self.max_digits})")
            
            if self.min_value is not None and value < self.min_value:
                raise ValidationError(f"Value must be greater than or equal to {self.min_value}")
            
            if self.max_value is not None and value > self.max_value:
                raise ValidationError(f"Value must be less than or equal to {self.max_value}")
        
        return value
    
    def to_python(self, value: Any) -> Any:
        if value is not None and not isinstance(value, Decimal):
            return Decimal(str(value))
        return value
    
    def to_database(self, value: Any) -> Any:
        """Convert Python Decimal to database value"""
        if isinstance(value, Decimal):
            return float(value)  # یا str(value) برای دقت بیشتر
        return value


class BooleanField(Field):
    """
    Boolean field
    """
    def __init__(self, **kwargs):
        super().__init__(field_type="BOOLEAN", **kwargs)
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                lower_value = value.lower()
                if lower_value in ('true', '1', 'yes', 'on'):
                    return True
                elif lower_value in ('false', '0', 'no', 'off'):
                    return False
                else:
                    raise ValidationError("Invalid boolean string value")
            elif isinstance(value, (int, float)):
                return bool(value)
            else:
                raise ValidationError("Value must be a boolean")
        return value


class DateTimeField(Field):
    """
    DateTime field
    """
    __slots__ = ('auto_now', 'auto_now_add')
    
    def __init__(self, auto_now: bool = False, auto_now_add: bool = False, **kwargs):
        super().__init__(field_type="DATETIME", **kwargs)
        self.auto_now = auto_now
        self.auto_now_add = auto_now_add
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if isinstance(value, str):
                try:
                    # Try ISO format first
                    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                except ValueError:
                    # Try other common formats
                    formats = [
                        '%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%d %H:%M:%S.%f',
                        '%Y-%m-%dT%H:%M:%S',
                        '%Y-%m-%dT%H:%M:%S.%f',
                        '%Y-%m-%d'
                    ]
                    parsed = False
                    for fmt in formats:
                        try:
                            value = datetime.strptime(value, fmt)
                            parsed = True
                            break
                        except ValueError:
                            continue
                    
                    if not parsed:
                        raise ValidationError("Invalid datetime format")
            elif isinstance(value, (int, float)):
                # Assume timestamp
                value = datetime.fromtimestamp(value)
            elif not isinstance(value, datetime):
                raise ValidationError("Value must be a datetime")
        return value
    
    def to_python(self, value: Any) -> Any:
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                return value
        return value


class DateField(Field):
    """
    Date field
    """
    def __init__(self, **kwargs):
        super().__init__(field_type="DATE", **kwargs)
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if isinstance(value, str):
                try:
                    value = datetime.strptime(value, '%Y-%m-%d').date()
                except ValueError:
                    raise ValidationError("Invalid date format, expected YYYY-MM-DD")
            elif isinstance(value, datetime):
                value = value.date()
            elif not isinstance(value, date):
                raise ValidationError("Value must be a date")
        return value
    
    def to_python(self, value: Any) -> Any:
        if isinstance(value, str) and value:
            try:
                return datetime.strptime(value, '%Y-%m-%d').date()
            except ValueError:
                return value
        return value


class TimeField(Field):
    """
    Time field
    """
    def __init__(self, **kwargs):
        super().__init__(field_type="TIME", **kwargs)
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if isinstance(value, str):
                try:
                    # Try various time formats
                    formats = ['%H:%M:%S', '%H:%M:%S.%f', '%H:%M']
                    parsed = False
                    for fmt in formats:
                        try:
                            datetime.strptime(value, fmt)
                            parsed = True
                            break
                        except ValueError:
                            continue
                    
                    if not parsed:
                        raise ValidationError("Invalid time format")
                except ValueError:
                    raise ValidationError("Invalid time format")
            elif not isinstance(value, datetime) and hasattr(value, 'hour'):
                # Already a time object
                pass
            else:
                raise ValidationError("Value must be a time")
        return value


class JSONField(Field):
    """
    JSON field
    """
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
    
    def to_python(self, value: Any) -> Any:
        if isinstance(value, str) and value:
            try:
                import json
                return json.loads(value)
            except:
                return value
        return value
    
    def to_database(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            import json
            return json.dumps(value)
        return value


class EmailField(CharField):
    """
    Email field with validation
    """
    __slots__ = ('email_regex',)
    
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
    """
    URL field with validation
    """
    __slots__ = ('url_regex',)
    
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


class UUIDField(Field):
    """
    UUID field
    """
    def __init__(self, **kwargs):
        super().__init__(field_type="VARCHAR(36)", **kwargs)
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if isinstance(value, uuid.UUID):
                return value
            elif isinstance(value, str):
                try:
                    return uuid.UUID(value)
                except ValueError:
                    raise ValidationError("Invalid UUID format")
            else:
                raise ValidationError("Value must be a UUID or UUID string")
        return value
    
    def to_python(self, value: Any) -> Any:
        if isinstance(value, str) and value:
            try:
                return uuid.UUID(value)
            except ValueError:
                return value
        return value
    
    def to_database(self, value: Any) -> Any:
        if isinstance(value, uuid.UUID):
            return str(value)
        return value


class IPAddressField(Field):
    """
    IP Address field (IPv4 and IPv6)
    """
    def __init__(self, protocol: str = 'both', **kwargs):
        super().__init__(field_type="VARCHAR(39)", **kwargs)  # 39 chars for IPv6
        self.protocol = protocol.lower()
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if isinstance(value, (IPv4Address, IPv6Address)):
                if self.protocol == 'ipv4' and not isinstance(value, IPv4Address):
                    raise ValidationError("Only IPv4 addresses are allowed")
                elif self.protocol == 'ipv6' and not isinstance(value, IPv6Address):
                    raise ValidationError("Only IPv6 addresses are allowed")
                return value
            elif isinstance(value, str):
                try:
                    ip = IPv4Address(value)
                    if self.protocol == 'ipv6':
                        raise ValidationError("Only IPv6 addresses are allowed")
                    return ip
                except ValueError:
                    try:
                        ip = IPv6Address(value)
                        if self.protocol == 'ipv4':
                            raise ValidationError("Only IPv4 addresses are allowed")
                        return ip
                    except ValueError:
                        raise ValidationError("Invalid IP address format")
            else:
                raise ValidationError("Value must be an IP address")
        return value


class SlugField(CharField):
    """
    Slug field (URL-friendly string)
    """
    __slots__ = ('slug_regex',)
    
    def __init__(self, **kwargs):
        # اصلاح اینجا: ابتدا max_length را استخراج کنیم یا مقدار پیش‌فرض بدهیم
        max_length = kwargs.pop('max_length', 50)
        super().__init__(max_length=max_length, **kwargs)
        self.slug_regex = re.compile(r'^[-a-zA-Z0-9_]+$')
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if not isinstance(value, str):
                value = str(value)
            if not self.slug_regex.match(value):
                raise ValidationError("Slug can only contain letters, numbers, hyphens, and underscores")
        return value


class PositiveIntegerField(IntegerField):
    """
    Integer field that only accepts positive values
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('min_value', 0)
        super().__init__(**kwargs)


class PositiveSmallIntegerField(SmallIntegerField):
    """
    Small integer field that only accepts positive values
    """
    def __init__(self, **kwargs):
        kwargs.setdefault('min_value', 0)
        super().__init__(**kwargs)


class AutoField(IntegerField):
    """
    Auto-incrementing integer field
    """
    def __init__(self, **kwargs):
        kwargs['primary_key'] = True
        kwargs['auto_increment'] = True
        kwargs['nullable'] = False
        super().__init__(**kwargs)


class BigAutoField(BigIntegerField):
    """
    Auto-incrementing big integer field
    """
    def __init__(self, **kwargs):
        kwargs['primary_key'] = True
        kwargs['auto_increment'] = True
        kwargs['nullable'] = False
        super().__init__(**kwargs)


class BinaryField(Field):
    """
    Binary data field
    """
    def __init__(self, max_length: Optional[int] = None, **kwargs):
        field_type = f"VARBINARY({max_length})" if max_length else "BLOB"
        super().__init__(field_type=field_type, **kwargs)
        self.max_length = max_length
    
    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if not isinstance(value, (bytes, bytearray)):
                raise ValidationError("Value must be bytes or bytearray")
            if self.max_length and len(value) > self.max_length:
                raise ValidationError(f"Value exceeds maximum length of {self.max_length}")
        return value


# Convenience aliases
SmallAutoField = AutoField  # For consistency