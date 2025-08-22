# ormax/fields.py
import re
import uuid
from typing import Any, Optional, Union, Callable, List, Type
from datetime import datetime, date, timezone
from decimal import Decimal
from ipaddress import IPv4Address, IPv6Address

import ormax
from .exceptions import ValidationError
from .utils import parse_datetime as utils_parse_datetime


class Field:
    """
    Base field class for all ORM fields
    """

    __slots__ = (
        "field_type",
        "primary_key",
        "auto_increment",
        "nullable",
        "default",
        "unique",
        "index",
        "foreign_key",
        "choices",
        "help_text",
        "validators",
        "db_column",
        "db_index",
    )

    def __init__(
        self,
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
        db_index: bool = False,
    ):
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

    def get_sql_type(self, database=None) -> str:
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
                raise ValidationError(
                    f"Value '{value}' is not in valid choices: {valid_choices}"
                )
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
    
    def pre_save(self, model_instance, add: bool) -> Any:
        """
        Hook for processing field before saving.
        
        Args:
            model_instance: The model instance being saved.
            add: True if creating a new instance, False if updating.
            
        Returns:
            The processed value or None if no change is needed.
        """
        return None


class CharField(Field):
    """
    Character field with maximum length and database-specific optimizations
    """
    __slots__ = ("max_length", "min_length", "_original_max_length")

    def __init__(self, max_length: int = 255, min_length: int = 0, **kwargs):
        # ذخیره طول اصلی برای استفاده در اعتبارسنجی
        self._original_max_length = max_length
        self.min_length = min_length
        
        # برای فیلدهای UNIQUE در MySQL، طول را به 191 محدود می‌کنیم
        adjusted_max_length = max_length
        super().__init__(field_type=f"VARCHAR({adjusted_max_length})", **kwargs)
        self.max_length = adjusted_max_length

    def get_sql_type(self, database=None) -> str:
        """Get SQL type with database-specific optimizations"""
        # برای فیلدهای UNIQUE در MySQL، طول را به 191 محدود می‌کنیم
        if database and "mysql" in database.connection_string and self.unique:
            return f"VARCHAR(191)"
        # برای سایر پایگاه‌داده‌ها از طول اصلی استفاده می‌کنیم
        return f"VARCHAR({self._original_max_length})"

    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if not isinstance(value, str):
                try:
                    value = str(value)
                except Exception:
                    raise ValidationError("Value cannot be converted to string")
            # برای اعتبارسنجی، از طول اصلی استفاده می‌کنیم
            if len(value) > self._original_max_length:
                raise ValidationError(
                    f"Value exceeds maximum length of {self._original_max_length}"
                )
            if len(value) < self.min_length:
                raise ValidationError(
                    f"Value is shorter than minimum length of {self.min_length}"
                )
        return value


class TextField(Field):
    """
    Long text field with optional min/max length validation.

    Example:
        class Article(Model):
            content = TextField(max_length=5000, min_length=10)

        # Usage
        article = await Article.create(content="This is a long text...")
    """

    __slots__ = ("max_length", "min_length")

    def __init__(self, max_length: int = None, min_length: int = 0, **kwargs):
        """
        :param max_length: Maximum allowed length (None for unlimited)
        :param min_length: Minimum required length
        """
        super().__init__(field_type="TEXT", **kwargs)
        self.max_length = max_length
        self.min_length = min_length

    def validate(self, value: Any) -> Any:
        """
        Validate that the value can be stored as text and matches length constraints.
        """
        value = super().validate(value)
        if value is not None:
            # Convert to string if not already
            if not isinstance(value, str):
                try:
                    value = str(value)
                except Exception:
                    raise ValidationError("Value cannot be converted to string")

            # Check length constraints
            if self.max_length is not None and len(value) > self.max_length:
                raise ValidationError(
                    f"Value exceeds maximum length of {self.max_length}"
                )
            if len(value) < self.min_length:
                raise ValidationError(
                    f"Value is shorter than minimum length of {self.min_length}"
                )

        return value


class IntegerField(Field):
    """
    Integer field
    """

    __slots__ = ("min_value", "max_value")

    def __init__(
        self, min_value: Optional[int] = None, max_value: Optional[int] = None, **kwargs
    ):
        auto_increment = kwargs.pop("auto_increment", False)
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
                raise ValidationError(
                    f"Value must be greater than or equal to {self.min_value}"
                )
            if self.max_value is not None and value > self.max_value:
                raise ValidationError(
                    f"Value must be less than or equal to {self.max_value}"
                )
        return value


class BigIntegerField(Field):
    """
    Big integer field
    """

    __slots__ = ("min_value", "max_value")

    def __init__(
        self, min_value: Optional[int] = None, max_value: Optional[int] = None, **kwargs
    ):
        auto_increment = kwargs.pop("auto_increment", False)
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
                raise ValidationError(
                    f"Value must be greater than or equal to {self.min_value}"
                )
            if self.max_value is not None and value > self.max_value:
                raise ValidationError(
                    f"Value must be less than or equal to {self.max_value}"
                )
        return value


class SmallIntegerField(Field):
    """
    Small integer field (16-bit)
    """

    __slots__ = ("min_value", "max_value")

    def __init__(
        self, min_value: Optional[int] = None, max_value: Optional[int] = None, **kwargs
    ):
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
                raise ValidationError(
                    f"Value must be greater than or equal to {self.min_value}"
                )
            if self.max_value is not None and value > self.max_value:
                raise ValidationError(
                    f"Value must be less than or equal to {self.max_value}"
                )
        return value


class FloatField(Field):
    """
    Floating point number field
    """

    __slots__ = ("min_value", "max_value")

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        **kwargs,
    ):
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
                raise ValidationError(
                    f"Value must be greater than or equal to {self.min_value}"
                )
            if self.max_value is not None and value > self.max_value:
                raise ValidationError(
                    f"Value must be less than or equal to {self.max_value}"
                )
        return value


class DecimalField(Field):
    """
    Decimal field with precision and scale
    """

    __slots__ = ("max_digits", "decimal_places", "min_value", "max_value")

    def __init__(
        self,
        max_digits: int = 10,
        decimal_places: int = 2,
        min_value: Optional[Union[Decimal, float, str]] = None,
        max_value: Optional[Union[Decimal, float, str]] = None,
        **kwargs,
    ):
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
                raise ValidationError(
                    f"Value has more decimal places than allowed ({self.decimal_places})"
                )
            if len(str(value).replace(".", "").replace("-", "")) > self.max_digits:
                raise ValidationError(
                    f"Value has more digits than allowed ({self.max_digits})"
                )
            if self.min_value is not None and value < self.min_value:
                raise ValidationError(
                    f"Value must be greater than or equal to {self.min_value}"
                )
            if self.max_value is not None and value > self.max_value:
                raise ValidationError(
                    f"Value must be less than or equal to {self.max_value}"
                )
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
                if lower_value in ("true", "1", "yes", "on"):
                    return True
                elif lower_value in ("false", "0", "no", "off"):
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
    DateTime field with comprehensive timezone support and optimized validation
    """
    __slots__ = ("auto_now", "auto_now_add", "use_tz", "database_type")

    def __init__(
        self, 
        auto_now: bool = False, 
        auto_now_add: bool = False,
        use_tz: bool = True,  # Enable timezone support by default
        **kwargs
    ):
        """
        Initialize DateTimeField with enhanced options.
        
        Args:
            auto_now: Automatically set to current time on each save
            auto_now_add: Automatically set to current time on creation
            use_tz: Whether to use timezone-aware datetime objects
        """
        # Determine appropriate database type based on connection
        self.database_type = "TIMESTAMP"  # Default for most databases
        
        # Store timezone preference
        self.use_tz = use_tz
        
        super().__init__(
            field_type=self.database_type,
            **kwargs
        )
        self.auto_now = auto_now
        self.auto_now_add = auto_now_add
        
        # Validate incompatible options
        if auto_now and auto_now_add:
            raise ValueError("Cannot set both auto_now and auto_now_add to True")

    def get_sql_type(self, database=None) -> str:
        """Get SQL type with database-specific optimizations"""
        # Determine appropriate database type based on connection
        if database:
            if "mysql" in database.connection_string:
                return "DATETIME"
            elif "sqlite" in database.connection_string:
                return "TIMESTAMP"
            elif "postgresql" in database.connection_string:
                return "TIMESTAMPTZ" if self.use_tz else "TIMESTAMP"
            elif "mssql" in database.connection_string:
                return "DATETIMEOFFSET" if self.use_tz else "DATETIME2"
        return self.database_type

    def validate(self, value: Any) -> Any:
        """Validate and normalize datetime values with proper timezone handling"""
        # Skip validation for auto fields (they'll be set by ORM)
        if (self.auto_now or self.auto_now_add) and value is None:
            return value
            
        # Handle None values according to nullability
        if value is None:
            if not self.nullable and not (self.auto_now or self.auto_now_add):
                raise ValidationError("DateTimeField cannot be null")
            return None

        # Case 1: Already a datetime object
        if isinstance(value, datetime):
            return self._normalize_datetime(value)
            
        # Case 2: String representation
        if isinstance(value, str):
            try:
                # Use utility function for robust parsing
                dt = utils_parse_datetime(value)
                return self._normalize_datetime(dt)
            except (ValueError, TypeError) as e:
                raise ValidationError(f"Invalid datetime string: {str(e)}")
                
        # Case 3: Timestamp (int/float)
        if isinstance(value, (int, float)):
            try:
                dt = datetime.fromtimestamp(value, tz=timezone.utc if self.use_tz else None)
                return self._normalize_datetime(dt)
            except (OSError, TypeError, ValueError) as e:
                raise ValidationError(f"Invalid timestamp: {str(e)}")
                
        # Case 4: Date object (convert to datetime)
        if isinstance(value, date) and not isinstance(value, datetime):
            dt = datetime.combine(value, datetime.min.time())
            if self.use_tz:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
            
        raise ValidationError(
            "Value must be a datetime, date, timestamp, or valid datetime string"
        )

    def _normalize_datetime(self, dt: datetime) -> datetime:
        """Normalize datetime to proper timezone format"""
        # Ensure datetime is timezone-aware if required
        if self.use_tz and dt.tzinfo is None:
            # Assume UTC if no timezone is provided
            dt = dt.replace(tzinfo=timezone.utc)
        elif not self.use_tz and dt.tzinfo is not None:
            # Convert to naive datetime by removing timezone
            dt = dt.replace(tzinfo=None)
            
        return dt

    def to_python(self, value: Any) -> Optional[datetime]:
        """Convert database value to Python datetime object"""
        if value is None:
            return None
            
        # Already a datetime object
        if isinstance(value, datetime):
            return self._normalize_datetime(value)
            
        # String value from database
        if isinstance(value, str):
            try:
                # Handle ISO format with timezone
                if "T" in value and ("+" in value or "Z" in value or "-" in value[-6:]):
                    # Use utility function for robust parsing
                    return utils_parse_datetime(value)
                # Handle MySQL DATETIME format without timezone
                elif " " in value:
                    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]
                    for fmt in formats:
                        try:
                            dt = datetime.strptime(value, fmt)
                            return self._normalize_datetime(dt)
                        except ValueError:
                            continue
                # Handle date-only format
                return utils_parse_datetime(value)
            except (ValueError, TypeError):
                pass
                
        # Timestamp value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc if self.use_tz else None)
            
        # Date object
        if isinstance(value, date) and not isinstance(value, datetime):
            dt = datetime.combine(value, datetime.min.time())
            return self._normalize_datetime(dt)
            
        return value

    def to_database(self, value: Any) -> Any:
        """Convert Python value to database representation"""
        if value is None:
            # Handle auto fields
            if self.auto_now or self.auto_now_add:
                now = datetime.now(tz=timezone.utc if self.use_tz else None)
                return self._normalize_datetime(now)
            return None
            
        # Convert to appropriate format for database
        if isinstance(value, datetime):
            normalized = self._normalize_datetime(value)
            # For databases that require specific formatting
            if "mysql" in self._database.connection_string:
                return normalized.strftime("%Y-%m-%d %H:%M:%S")
            return normalized
            
        return value

    def __get__(self, instance, owner):
        """Descriptor protocol for proper field access"""
        if instance is None:
            return self
            
        value = instance._data.get(self.name)
        # For auto fields, return current time if value is None
        if value is None and (self.auto_now or self.auto_now_add):
            return datetime.now(tz=timezone.utc if self.use_tz else None)
        return value

    def __set__(self, instance, value):
        """Descriptor protocol for setting field values"""
        if instance is None:
            return
            
        # Handle auto fields - they should not be manually set
        if self.auto_now or self.auto_now_add:
            if value is not None:
                raise ValueError(
                    f"Cannot manually set value for auto_now/auto_now_add field '{self.name}'"
                )
            # Auto fields will be handled in pre_save
            instance._data[self.name] = None
            return
            
        # Validate and store the value
        validated_value = self.validate(value)
        instance._data[self.name] = validated_value

    def pre_save(self, model_instance, add: bool) -> Any:
        """
        Handle auto_now and auto_now_add before saving.
        Called by the ORM before saving the model.
        """
        if self.auto_now or (self.auto_now_add and add):
            return datetime.now(tz=timezone.utc if self.use_tz else None)
        # Call parent implementation (if any)
        return super().pre_save(model_instance, add)

    def __get__(self, instance, owner):
        """Descriptor protocol for proper field access"""
        if instance is None:
            return self
            
        value = instance._data.get(self.name)
        # For auto fields, return current time if value is None
        if value is None and (self.auto_now or self.auto_now_add):
            return datetime.now(tz=timezone.utc if self.use_tz else None)
        return value

    def __set__(self, instance, value):
        """Descriptor protocol for setting field values"""
        if instance is None:
            return
            
        # Handle auto fields - they should not be manually set
        if self.auto_now or self.auto_now_add:
            if value is not None:
                raise ValueError(
                    f"Cannot manually set value for auto_now/auto_now_add field '{self.name}'"
                )
            # Auto fields will be handled in pre_save
            instance._data[self.name] = None
            return
            
        # Validate and store the value
        validated_value = self.validate(value)
        instance._data[self.name] = validated_value

    def pre_save(self, model_instance, add: bool) -> Any:
        """
        Handle auto_now and auto_now_add before saving.
        Called by the ORM before saving the model.
        """
        if self.auto_now or (self.auto_now_add and add):
            return datetime.now(tz=timezone.utc if self.use_tz else None)
        return super().pre_save(model_instance, add)

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
                    value = datetime.strptime(value, "%Y-%m-%d").date()
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
                return datetime.strptime(value, "%Y-%m-%d").date()
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
                    formats = ["%H:%M:%S", "%H:%M:%S.%f", "%H:%M"]
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
            elif not isinstance(value, datetime) and hasattr(value, "hour"):
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
    Email field with validation and database-specific optimizations
    """
    __slots__ = ("email_regex",)

    def __init__(self, **kwargs):
        # استخراج max_length از kwargs اگر وجود داشت
        max_length = kwargs.pop('max_length', 254)
        super().__init__(max_length=max_length, **kwargs)
        self.email_regex = re.compile(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
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

    __slots__ = ("url_regex",)

    def __init__(self, **kwargs):
        super().__init__(max_length=2048, **kwargs)
        self.url_regex = re.compile(
            r"^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$"
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

    def __init__(self, protocol: str = "both", **kwargs):
        super().__init__(field_type="VARCHAR(39)", **kwargs)  # 39 chars for IPv6
        self.protocol = protocol.lower()

    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if isinstance(value, (IPv4Address, IPv6Address)):
                if self.protocol == "ipv4" and not isinstance(value, IPv4Address):
                    raise ValidationError("Only IPv4 addresses are allowed")
                elif self.protocol == "ipv6" and not isinstance(value, IPv6Address):
                    raise ValidationError("Only IPv6 addresses are allowed")
                return value
            elif isinstance(value, str):
                try:
                    ip = IPv4Address(value)
                    if self.protocol == "ipv6":
                        raise ValidationError("Only IPv6 addresses are allowed")
                    return ip
                except ValueError:
                    try:
                        ip = IPv6Address(value)
                        if self.protocol == "ipv4":
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

    __slots__ = ("slug_regex",)

    def __init__(self, **kwargs):
        # اصلاح اینجا: ابتدا max_length را استخراج کنیم یا مقدار پیش‌فرض بدهیم
        max_length = kwargs.pop("max_length", 50)
        super().__init__(max_length=max_length, **kwargs)
        self.slug_regex = re.compile(r"^[-a-zA-Z0-9_]+$")

    def validate(self, value: Any) -> Any:
        value = super().validate(value)
        if value is not None:
            if not isinstance(value, str):
                value = str(value)
            if not self.slug_regex.match(value):
                raise ValidationError(
                    "Slug can only contain letters, numbers, hyphens, and underscores"
                )
        return value


class PositiveIntegerField(IntegerField):
    """
    Integer field that only accepts positive values
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("min_value", 0)
        super().__init__(**kwargs)


class PositiveSmallIntegerField(SmallIntegerField):
    """
    Small integer field that only accepts positive values
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("min_value", 0)
        super().__init__(**kwargs)


class AutoField(IntegerField):
    def __init__(self, **kwargs):
        kwargs["primary_key"] = True
        kwargs["auto_increment"] = True
        kwargs["nullable"] = False
        super().__init__(**kwargs)


class BigAutoField(BigIntegerField):
    """
    Auto-incrementing big integer field
    """

    def __init__(self, **kwargs):
        kwargs["primary_key"] = True
        kwargs["auto_increment"] = True
        kwargs["nullable"] = False
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
                raise ValidationError(
                    f"Value exceeds maximum length of {self.max_length}"
                )
        return value


# Convenience aliases
SmallAutoField = AutoField  # For consistency


# New Field for Relationships
class ForeignKeyField(Field):
    """
    Foreign Key field for relationships
    """

    __slots__ = ("to", "related_name", "on_delete")

    def __init__(
        self,
        to: Union[str, Type["ormax.Model"]],
        related_name: Optional[str] = None,
        on_delete: str = "CASCADE",  # CASCADE, SET_NULL, RESTRICT, etc.
        **kwargs,
    ):
        # Set nullable to True by default for ForeignKey unless explicitly set
        kwargs.setdefault("nullable", True)
        super().__init__(field_type="INTEGER", **kwargs)  # Default to INTEGER for FK
        self.to = to  # Can be a string (model name) or a Model class
        self.related_name = related_name
        self.on_delete = on_delete

    def get_sql_type(self, database=None) -> str:
        """Get SQL type for this field with database-specific optimizations"""
        # Determine appropriate type based on database connection
        if database:
            if "mysql" in database.connection_string:
                return "BIGINT" if "BIGINT" in self.field_type else "INT"
            elif "sqlite" in database.connection_string:
                return "INTEGER"
            elif "postgresql" in database.connection_string:
                return "BIGINT" if "BIGINT" in self.field_type else "INTEGER"
            elif "mssql" in database.connection_string:
                return "BIGINT" if "BIGINT" in self.field_type else "INT"
            elif "oracle" in database.connection_string:
                return "NUMBER(19)" if "BIGINT" in self.field_type else "NUMBER(10)"
        # Default fallback
        return "BIGINT" if "BIGINT" in self.field_type else "INTEGER"