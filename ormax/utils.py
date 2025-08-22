# ormax/utils.py
"""
Utility functions and helpers for Ormax ORM
"""
import hashlib
import secrets
import json
import re
from typing import Any, Dict, Iterator, List, Tuple, Optional, Callable, Union, TypeVar
from datetime import datetime, date, time
from decimal import Decimal
from functools import wraps
import asyncio
import logging
from collections import OrderedDict, defaultdict
import inspect
import urllib

import ormax

logger = logging.getLogger(__name__)
# Type variables for generic functions
T = TypeVar("T")
ModelType = TypeVar("ModelType")


class OrmaxJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Ormax ORM types"""

    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, time):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)


def json_dumps(obj: Any, **kwargs) -> str:
    """JSON dumps with Ormax custom encoder"""
    return json.dumps(obj, cls=OrmaxJSONEncoder, **kwargs)


def json_loads(s: str, **kwargs) -> Any:
    """JSON loads with standard decoder"""
    return json.loads(s, **kwargs)


def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token"""
    return secrets.token_hex(length)


def generate_secure_password(length: int = 12) -> str:
    """Generate a secure random password"""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hash a password with salt using PBKDF2"""
    if salt is None:
        salt = generate_secure_token(16)
    pwdhash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
    )
    return pwdhash.hex(), salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verify a password against hash"""
    pwdhash, _ = hash_password(password, salt)
    return pwdhash == hashed


def sanitize_input(value: Any) -> Any:
    """Sanitize input to prevent injection attacks"""
    if isinstance(value, str):
        # Remove null bytes and other dangerous characters
        value = value.replace("\x00", "").replace("\x1a", "")
        # HTML escape
        import html

        value = html.escape(value)
    return value


def sanitize_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize all values in a dictionary"""
    return {key: sanitize_input(value) for key, value in data.items()}


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case"""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase"""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def pluralize(word: str) -> str:
    """Simple pluralization of English words"""
    if word.endswith("y"):
        return word[:-1] + "ies"
    elif word.endswith(("s", "sh", "ch", "x", "z")):
        return word + "es"
    else:
        return word + "s"


def singularize(word: str) -> str:
    """Simple singularization of English words"""
    if word.endswith("ies"):
        return word[:-3] + "y"
    elif word.endswith("ses"):
        return word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    else:
        return word


def get_class_name(obj: Any) -> str:
    """Get the class name of an object"""
    if inspect.isclass(obj):
        return obj.__name__
    return obj.__class__.__name__


def get_module_name(obj: Any) -> str:
    """Get the module name of an object"""
    if inspect.isclass(obj):
        return obj.__module__
    return obj.__class__.__module__


def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    """Flatten a nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict, sep: str = "_") -> Dict:
    """Unflatten a flattened dictionary"""
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d_ptr = result
        for part in parts[:-1]:
            if part not in d_ptr:
                d_ptr[part] = {}
            d_ptr = d_ptr[part]
        d_ptr[parts[-1]] = value
    return result


def chunk_list(lst: List[T], chunk_size: int) -> Iterator[List[T]]:
    """Split a list into chunks"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def unique_list(lst: List[T]) -> List[T]:
    """Remove duplicates while preserving order"""
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def retry_async(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying async functions"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise e
                    logger.warning(
                        f"Attempt {attempts} failed for {func.__name__}: {e}"
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator


def timeout_async(seconds: int):
    """Decorator to add timeout to async functions"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Function {func.__name__} timed out after {seconds} seconds"
                )

        return wrapper

    return decorator


def cached_property(func):
    """Cached property decorator"""
    attr_name = f"_cached_{func.__name__}"

    @wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return property(wrapper)


def memoize_async(maxsize: int = 128):
    """Async memoization decorator"""

    def decorator(func):
        cache = {}
        cache_order = []

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            if key in cache:
                # Move to end (most recently used)
                cache_order.remove(key)
                cache_order.append(key)
                return cache[key]
            # Compute result
            result = await func(*args, **kwargs)
            # Add to cache
            cache[key] = result
            cache_order.append(key)
            # Remove oldest if cache is full
            if len(cache) > maxsize:
                oldest_key = cache_order.pop(0)
                del cache[oldest_key]
            return result

        return wrapper

    return decorator


def async_lru_cache(maxsize: int = 128):
    """LRU cache for async functions (alternative implementation)"""
    return memoize_async(maxsize)


def measure_time_async(func):
    """Decorator to measure execution time of async functions"""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")

    return wrapper


def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def validate_url(url: str) -> bool:
    """Validate URL format"""
    pattern = r"^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$"
    return re.match(pattern, url) is not None


def validate_phone(phone: str) -> bool:
    """Validate phone number format (international)"""
    pattern = r"^\+?[1-9]\d{1,14}$"
    return re.match(pattern, phone) is not None


def format_currency(amount: Union[int, float, Decimal], currency: str = "USD") -> str:
    """Format currency amount"""
    if isinstance(amount, Decimal):
        amount = float(amount)
    return f"{amount:,.2f} {currency}"


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime object"""
    return dt.strftime(format_str)


def parse_datetime(dt_str: str) -> datetime:
    """Parse datetime string"""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse datetime string: {dt_str}")


def calculate_age(birth_date: date) -> int:
    """Calculate age from birth date"""
    today = date.today()
    return (
        today.year
        - birth_date.year
        - ((today.month, today.day) < (birth_date.month, birth_date.day))
    )


def generate_slug(text: str) -> str:
    """Generate URL-friendly slug from text"""
    # Convert to lowercase
    slug = text.lower()
    # Replace spaces and special characters with hyphens
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    # Remove leading and trailing hyphens
    slug = slug.strip("-")
    # Remove multiple consecutive hyphens
    slug = re.sub(r"-+", "-", slug)
    return slug


def mask_string(s: str, visible_chars: int = 4) -> str:
    """Mask string for privacy (e.g., credit card numbers)"""
    if len(s) <= visible_chars:
        return "*" * len(s)
    return "*" * (len(s) - visible_chars) + s[-visible_chars:]


def truncate_string(s: str, max_length: int, suffix: str = "...") -> str:
    """Truncate string to maximum length"""
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


def humanize_bytes(bytes_value: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def humanize_number(num: Union[int, float]) -> str:
    """Convert number to human readable format"""
    for unit in ["", "K", "M", "B", "T"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}".strip()
        num /= 1000.0
    return f"{num:.1f}P"


def get_percentage(part: Union[int, float], whole: Union[int, float]) -> float:
    """Calculate percentage"""
    if whole == 0:
        return 0.0
    return (part / whole) * 100


def paginate_list(items: List[T], page: int, per_page: int) -> Tuple[List[T], int, int]:
    """Paginate a list of items"""
    total_items = len(items)
    total_pages = (total_items + per_page - 1) // per_page
    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_items = items[start_idx:end_idx]
    return paginated_items, total_pages, total_items


def diff_dicts(old_dict: Dict, new_dict: Dict) -> Dict[str, Tuple[Any, Any]]:
    """Compare two dictionaries and return differences"""
    differences = {}
    all_keys = set(old_dict.keys()).union(set(new_dict.keys()))
    for key in all_keys:
        old_value = old_dict.get(key)
        new_value = new_dict.get(key)
        if old_value != new_value:
            differences[key] = (old_value, new_value)
    return differences


def merge_model_instances(base_instance, update_instance) -> None:
    """Merge attributes from update_instance into base_instance"""
    for field_name, field_value in update_instance._data.items():
        if field_name in base_instance._fields:
            setattr(base_instance, field_name, field_value)


def get_model_fields(model_class) -> List[str]:
    """Get all field names of a model class"""
    return list(model_class._fields.keys())


def get_model_primary_key(model_class) -> Optional[str]:
    """Get primary key field name of a model class"""
    for field_name, field in model_class._fields.items():
        if field.primary_key:
            return field_name
    return None


def is_model_instance(obj) -> bool:
    """Check if object is a model instance"""
    return hasattr(obj, "_data") and hasattr(obj, "_fields")


def model_to_dict(model_instance, exclude_fields: List[str] = None) -> Dict[str, Any]:
    """Convert model instance to dictionary"""
    if not is_model_instance(model_instance):
        raise ValueError("Object is not a model instance")
    exclude_fields = exclude_fields or []
    result = {}
    for field_name, field_value in model_instance._data.items():
        if field_name not in exclude_fields:
            result[field_name] = field_value
    return result


def dict_to_model(model_class, data: Dict) -> "ormax.Model":
    """Convert dictionary to model instance"""
    return model_class(**data)


def batch_process_async(
    items: List[T],
    process_func: Callable[[T], Any],
    batch_size: int = 100,
    max_concurrent: int = 10,
) -> List[Any]:
    """Process items in batches with concurrency control"""

    async def process_batch(batch):
        tasks = [process_func(item) for item in batch]
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_task(task):
            async with semaphore:
                return await task

        return await asyncio.gather(*[limited_task(task) for task in tasks])

    async def run_batches():
        results = []
        for batch in chunk_list(items, batch_size):
            batch_results = await process_batch(batch)
            results.extend(batch_results)
        return results

    return asyncio.run(run_batches())


def create_database_url(
    scheme: str,
    username: str,
    password: str,
    host: str,
    port: Optional[int] = None,
    database: Optional[str] = None,
    **kwargs,
) -> str:
    """Create database connection URL"""
    if scheme == "sqlite":
        # برای SQLite فرمت متفاوتی نیاز است
        return f"{scheme}:///{database}" if database else f"{scheme}://"
    url = f"{scheme}://{username}:{password}@{host}"
    if port:
        url += f":{port}"
    if database:
        url += f"/{database}"
    if kwargs:
        query_params = "&".join([f"{k}={v}" for k, v in kwargs.items()])
        url += f"?{query_params}"
    return url


def parse_database_url(url: str) -> Dict[str, Any]:
    """Parse database connection URL"""
    import urllib.parse

    parsed = urllib.parse.urlparse(url)
    result = {
        "scheme": parsed.scheme,
        "username": parsed.username,
        "password": parsed.password,
        "host": parsed.hostname,
        "port": parsed.port,
        "database": parsed.path.lstrip("/"),
        "query_params": dict(urllib.parse.parse_qsl(parsed.query)),
    }
    return result


def get_database_type_from_url(url: str) -> str:
    """Get database type from connection URL"""
    scheme = urllib.parse.urlparse(url).scheme.lower()
    scheme_mapping = {
        "sqlite": "sqlite",
        "mysql": "mysql",
        "mariadb": "mariadb",
        "postgresql": "postgresql",
        "postgres": "postgresql",
        "mssql": "mssql",
        "oracle": "oracle",
        "aurora": "aurora",
    }
    return scheme_mapping.get(scheme, scheme)


def escape_sql_identifier(identifier: str) -> str:
    """Escape SQL identifier to prevent injection"""
    # Remove dangerous characters
    identifier = re.sub(r"[^\w]", "_", identifier)
    # Wrap in quotes (this varies by database)
    return f'"{identifier}"'


def format_sql_query(query: str, params: Tuple) -> str:
    """Format SQL query with parameters for logging"""
    try:
        # Simple parameter substitution for logging purposes
        formatted_query = query
        for param in params:
            if isinstance(param, str):
                formatted_query = formatted_query.replace("?", f"'{param}'", 1)
            else:
                formatted_query = formatted_query.replace("?", str(param), 1)
        return formatted_query
    except Exception:
        return query  # Return original query if formatting fails


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    import platform
    import sys

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }


def setup_logging(
    level: str = "INFO",
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[logging.StreamHandler(), logging.FileHandler("ormax.log")],
    )
    return logging.getLogger("ormax")


def deprecated(reason: str):
    """Decorator to mark functions as deprecated"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.warning(f"Function {func.__name__} is deprecated: {reason}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def requires_package(package_name: str):
    """Decorator to check if required package is installed"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                __import__(package_name)
            except ImportError:
                raise ImportError(
                    f"Package '{package_name}' is required for {func.__name__}"
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# Context managers
class Timer:
    """Context manager for timing operations"""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {duration:.4f} seconds")


class DatabaseTransaction:
    """Context manager for database transactions"""

    def __init__(self, database):
        self.database = database
        self.transaction_active = False

    async def __aenter__(self):
        await self.database.connection.execute("BEGIN")
        self.transaction_active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.transaction_active:
            if exc_type is None:
                await self.database.connection.execute("COMMIT")
            else:
                await self.database.connection.execute("ROLLBACK")


# Data structures
class LRUCache:
    """LRU (Least Recently Used) Cache implementation"""

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self.cache = OrderedDict()

    def get(self, key: str) -> Any:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.maxsize:
            # Remove least recently used
            self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self) -> None:
        self.cache.clear()

    def size(self) -> int:
        return len(self.cache)


class BoundedSemaphore:
    """Async bounded semaphore with dynamic resizing"""

    def __init__(self, initial_value: int = 10):
        self._semaphore = asyncio.BoundedSemaphore(initial_value)
        self._value = initial_value

    async def acquire(self):
        await self._semaphore.acquire()

    def release(self):
        self._semaphore.release()

    def resize(self, new_value: int):
        """Resize semaphore (this is a simplified implementation)"""
        if new_value != self._value:
            logger.warning(
                "Semaphore resizing is not fully supported in this implementation"
            )
            self._value = new_value


# Performance monitoring
class PerformanceMonitor:
    """Simple performance monitoring utility"""

    def __init__(self):
        self.metrics = defaultdict(list)

    def record(self, operation: str, duration: float):
        """Record operation duration"""
        self.metrics[operation].append(duration)

    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        if operation not in self.metrics:
            return {}
        durations = self.metrics[operation]
        return {
            "count": len(durations),
            "total": sum(durations),
            "average": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
        }

    def clear(self):
        """Clear all metrics"""
        self.metrics.clear()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
__all__ = [
    "OrmaxJSONEncoder",
    "json_dumps",
    "json_loads",
    "generate_secure_token",
    "generate_secure_password",
    "hash_password",
    "verify_password",
    "sanitize_input",
    "sanitize_dict",
    "camel_to_snake",
    "snake_to_camel",
    "pluralize",
    "singularize",
    "get_class_name",
    "get_module_name",
    "deep_merge_dicts",
    "flatten_dict",
    "unflatten_dict",
    "chunk_list",
    "unique_list",
    "retry_async",
    "timeout_async",
    "cached_property",
    "memoize_async",
    "measure_time_async",
    "validate_email",
    "validate_url",
    "validate_phone",
    "format_currency",
    "format_datetime",
    "parse_datetime",
    "calculate_age",
    "generate_slug",
    "mask_string",
    "truncate_string",
    "humanize_bytes",
    "humanize_number",
    "get_percentage",
    "paginate_list",
    "diff_dicts",
    "merge_model_instances",
    "get_model_fields",
    "get_model_primary_key",
    "is_model_instance",
    "model_to_dict",
    "dict_to_model",
    "batch_process_async",
    "create_database_url",
    "parse_database_url",
    "get_database_type_from_url",
    "escape_sql_identifier",
    "format_sql_query",
    "get_system_info",
    "setup_logging",
    "deprecated",
    "requires_package",
    "Timer",
    "DatabaseTransaction",
    "LRUCache",
    "BoundedSemaphore",
    "PerformanceMonitor",
    "performance_monitor",
]
