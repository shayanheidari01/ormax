# ormax/query.py
from typing import List, Optional, Dict, Any, Tuple, Callable, AsyncIterator, Union
import logging
import asyncio
import re

import ormax
from .exceptions import DatabaseError, MultipleObjectsReturned
from .utils import format_placeholders

logger = logging.getLogger(__name__)


class PrefetchedRelatedList(list):
    """List-like reverse relation that preserves the async ``all()`` API."""

    async def all(self):
        return list(self)


class QuerySet:
    """
    QuerySet class for database queries with advanced optimization features
    """

    __slots__ = (
        "model_class",
        "_database",
        "_where_clauses",
        "_where_params",
        "_order_by",
        "_limit",
        "_offset",
        "_select_fields",
        "_query_cache",
        "_distinct",
        "_group_by",
        "_having",
        "_having_params",
        "_joins",
        "_annotations",
        "_prefetch_related",
        "_select_related",
        "_deferred_fields",
        "_only_fields",
        "_using_index",
        "_for_update",
        "_timeout",
    )

    def __init__(self, model_class):
        self.model_class = model_class
        self._database = model_class._database
        self._where_clauses: List[str] = []
        self._where_params: List[Any] = []
        self._order_by: List[str] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._select_fields: List[str] = ["*"]
        self._query_cache = {}
        self._distinct: bool = False
        self._group_by: List[str] = []
        self._having: List[str] = []
        self._having_params: List[Any] = []
        self._joins: List[str] = []
        self._annotations: Dict[str, str] = {}
        self._prefetch_related: List[str] = []
        self._select_related: List[str] = []
        self._deferred_fields: List[str] = []
        self._only_fields: List[str] = []
        self._using_index: Optional[str] = None
        self._for_update: bool = False
        self._timeout: Optional[int] = None

    def _resolve_target_model(self, target_model: Any) -> Any:
        """Resolve a string target model reference (e.g. 'self' or 'ModelName') to a Model class."""
        if not isinstance(target_model, str):
            return target_model

        if target_model == "self":
            return self.model_class

        if target_model == self.model_class.__name__:
            return self.model_class

        db = self._database or getattr(self.model_class, "_database", None)
        if db and hasattr(db, "models") and target_model in db.models:
            return db.models[target_model]

        return target_model

    def _prepare_field_value(self, field_name: str, lookup: str, value: Any) -> Any:
        """Normalize model instances and field values before binding them."""
        field = self.model_class._fields.get(field_name)
        if field is None or lookup == "isnull":
            return value

        from .fields import ForeignKeyField

        def prepare_one(item):
            if isinstance(field, ForeignKeyField) and hasattr(item, "pk"):
                item = item.pk
            validated = field.validate(item)
            if hasattr(validated, "resolve_expression"):
                return validated
            return field.to_database(validated)

        if lookup == "in":
            if not isinstance(value, (list, tuple, set)):
                value = [value]
            return [prepare_one(item) for item in value]
        if lookup == "range":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise DatabaseError("Range lookup requires exactly two values")
            return [prepare_one(value[0]), prepare_one(value[1])]
        return prepare_one(value)

    LOOKUP_OPERATORS = {
        "exact", "iexact", "contains", "icontains", "gt", "gte", "lt", "lte",
        "in", "startswith", "istartswith", "endswith", "iendswith", "isnull", "range"
    }

    def _parse_filter_clause(self, key: str, value: Any, is_exclude: bool = False) -> Tuple[str, List[Any]]:
        """
        Parse field lookups (e.g. age__gt=18, name__icontains="john", author__name="Rowling").
        """
        if key in self.model_class._fields or key in self._annotations:
            if value is None:
                clause = f"{key} IS NOT NULL" if is_exclude else f"{key} IS NULL"
                return clause, []
            prepared = (
                self._prepare_field_value(key, "exact", value)
                if key in self.model_class._fields
                else value
            )
            clause = f"{key} != ?" if is_exclude else f"{key} = ?"
            return clause, [prepared]

        if "__" in key:
            parts = key.split("__")
            if parts[-1] in self.LOOKUP_OPERATORS:
                lookup = parts[-1]
                field_parts = parts[:-1]
            else:
                lookup = "exact"
                field_parts = parts

            # Handle relationship traversal (e.g., author__name)
            if len(field_parts) > 1:
                first_part = field_parts[0]
                if first_part in self.model_class._fields:
                    from .fields import ForeignKeyField
                    field_obj = self.model_class._fields[first_part]
                    if isinstance(field_obj, ForeignKeyField):
                        target_model = self._resolve_target_model(field_obj.to)
                        if target_model and not isinstance(target_model, str) and hasattr(target_model, "_fields") and hasattr(target_model, "objects"):
                            target_pk = next((n for n, f in target_model._fields.items() if f.primary_key), "id")
                            rel_key = "__".join(field_parts[1:])
                            if lookup != "exact":
                                rel_key = f"{rel_key}__{lookup}"
                            sub_qs = target_model.objects().filter(**{rel_key: value})
                            sub_sql, sub_params = sub_qs._build_select_query()
                            neg_op = "NOT IN" if is_exclude else "IN"
                            clause = f"{first_part} {neg_op} (SELECT {target_pk} FROM {target_model.get_table_name()} WHERE {' AND '.join(sub_qs._where_clauses)})"
                            return clause, sub_params

            col = field_parts[0]
            if col not in self.model_class._fields and col not in self._annotations:
                raise DatabaseError(
                    f"Field '{col}' does not exist on model '{self.model_class.__name__}'"
                )

            if col in self.model_class._fields:
                value = self._prepare_field_value(col, lookup, value)
            return self._compile_lookup_clause(col, lookup, value, is_exclude)

        raise DatabaseError(
            f"Field '{key}' does not exist on model '{self.model_class.__name__}'"
        )

    def _compile_lookup_clause(self, col: str, lookup: str, value: Any, is_exclude: bool) -> Tuple[str, List[Any]]:
        """Compile SQL lookup clause for column."""
        if lookup == "exact":
            if value is None:
                return (f"{col} IS NOT NULL" if is_exclude else f"{col} IS NULL"), []
            return (f"{col} != ?" if is_exclude else f"{col} = ?"), [value]
        elif lookup == "iexact":
            return (f"LOWER({col}) != LOWER(?)" if is_exclude else f"LOWER({col}) = LOWER(?)"), [str(value)]
        elif lookup == "gt":
            return (f"{col} <= ?" if is_exclude else f"{col} > ?"), [value]
        elif lookup == "gte":
            return (f"{col} < ?" if is_exclude else f"{col} >= ?"), [value]
        elif lookup == "lt":
            return (f"{col} >= ?" if is_exclude else f"{col} < ?"), [value]
        elif lookup == "lte":
            return (f"{col} > ?" if is_exclude else f"{col} <= ?"), [value]
        elif lookup == "in":
            val_list = list(value) if isinstance(value, (list, tuple, set)) else [value]
            if not val_list:
                return ("1 = 1" if is_exclude else "1 = 0"), []
            placeholders = ", ".join(["?"] * len(val_list))
            clause = f"{col} NOT IN ({placeholders})" if is_exclude else f"{col} IN ({placeholders})"
            return clause, val_list
        elif lookup == "contains":
            return (f"{col} NOT LIKE ?" if is_exclude else f"{col} LIKE ?"), [f"%{value}%"]
        elif lookup == "icontains":
            return (f"LOWER({col}) NOT LIKE LOWER(?)" if is_exclude else f"LOWER({col}) LIKE LOWER(?)"), [f"%{value}%"]
        elif lookup == "startswith":
            return (f"{col} NOT LIKE ?" if is_exclude else f"{col} LIKE ?"), [f"{value}%"]
        elif lookup == "istartswith":
            return (f"LOWER({col}) NOT LIKE LOWER(?)" if is_exclude else f"LOWER({col}) LIKE LOWER(?)"), [f"{value}%"]
        elif lookup == "endswith":
            return (f"{col} NOT LIKE ?" if is_exclude else f"{col} LIKE ?"), [f"%{value}"]
        elif lookup == "iendswith":
            return (f"LOWER({col}) NOT LIKE LOWER(?)" if is_exclude else f"LOWER({col}) LIKE LOWER(?)"), [f"%{value}"]
        elif lookup == "isnull":
            is_null = bool(value)
            if is_exclude:
                is_null = not is_null
            return (f"{col} IS NULL" if is_null else f"{col} IS NOT NULL"), []
        elif lookup == "range":
            return (f"{col} NOT BETWEEN ? AND ?" if is_exclude else f"{col} BETWEEN ? AND ?"), [value[0], value[1]]

        raise DatabaseError(f"Unsupported lookup operator: '{lookup}'")

    def filter(self, **kwargs) -> "QuerySet":
        """Add filter conditions"""
        new_queryset = self._fast_clone()
        for field_name, value in kwargs.items():
            clause, params = new_queryset._parse_filter_clause(field_name, value, is_exclude=False)
            new_queryset._where_clauses.append(clause)
            new_queryset._where_params.extend(params)
        return new_queryset

    def exclude(self, **kwargs) -> "QuerySet":
        """Add exclude conditions"""
        new_queryset = self._fast_clone()
        for field_name, value in kwargs.items():
            clause, params = new_queryset._parse_filter_clause(field_name, value, is_exclude=True)
            new_queryset._where_clauses.append(clause)
            new_queryset._where_params.extend(params)
        return new_queryset


    def order_by(self, *fields) -> "QuerySet":
        """Add order by clause"""
        new_queryset = self._fast_clone()
        for field in fields:
            if field.startswith("-"):
                field_name = field[1:]
                direction = "DESC"
            else:
                field_name = field
                direction = "ASC"
            if (
                field_name not in self.model_class._fields
                and field_name not in self._annotations
            ):
                raise DatabaseError(
                    f"Field '{field_name}' does not exist on model '{self.model_class.__name__}'"
                )
            new_queryset._order_by.append(f"{field_name} {direction}")
        return new_queryset

    def limit(self, limit: int) -> "QuerySet":
        """Limit the number of returned rows."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        new_queryset = self._fast_clone()
        new_queryset._limit = limit
        return new_queryset

    def offset(self, offset: int) -> "QuerySet":
        """Skip a non-negative number of rows."""
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        new_queryset = self._fast_clone()
        new_queryset._offset = offset
        return new_queryset

    def distinct(self) -> "QuerySet":
        """Add DISTINCT clause"""
        new_queryset = self._fast_clone()
        new_queryset._distinct = True
        return new_queryset

    def group_by(self, *fields) -> "QuerySet":
        """Add GROUP BY clause"""
        new_queryset = self._fast_clone()
        for field in fields:
            if field not in self.model_class._fields and field not in self._annotations:
                raise DatabaseError(
                    f"Field '{field}' does not exist on model '{self.model_class.__name__}'"
                )
            new_queryset._group_by.append(field)
        return new_queryset

    def having(self, condition: str, *params) -> "QuerySet":
        """Add HAVING clause"""
        new_queryset = self._fast_clone()
        new_queryset._having.append(condition)
        new_queryset._having_params.extend(params)
        return new_queryset

    def annotate(self, **annotations) -> "QuerySet":
        """Add annotations (computed fields)"""
        new_queryset = self._fast_clone()
        new_queryset._annotations.update(annotations)
        return new_queryset

    def select_related(self, *relations) -> "QuerySet":
        """Eager-load forward foreign keys using batched queries."""
        from .fields import ForeignKeyField

        new_queryset = self._fast_clone()
        for relation in relations:
            field = self.model_class._fields.get(relation)
            if not isinstance(field, ForeignKeyField):
                raise DatabaseError(f"Unknown forward relation '{relation}'")
            new_queryset._select_related.append(relation)
        return new_queryset

    def prefetch_related(self, *relations) -> "QuerySet":
        """Prefetch related objects in separate queries"""
        new_queryset = self._fast_clone()
        new_queryset._prefetch_related.extend(relations)
        return new_queryset

    def defer(self, *fields) -> "QuerySet":
        """Exclude model fields from the SELECT list."""
        unknown = set(fields) - set(self.model_class._fields)
        if unknown:
            raise DatabaseError(f"Unknown field(s): {', '.join(sorted(unknown))}")
        new_queryset = self._fast_clone()
        new_queryset._deferred_fields.extend(fields)
        return new_queryset

    def only(self, *fields) -> "QuerySet":
        """Load only the selected model fields."""
        unknown = set(fields) - set(self.model_class._fields)
        if unknown:
            raise DatabaseError(f"Unknown field(s): {', '.join(sorted(unknown))}")
        new_queryset = self._fast_clone()
        new_queryset._only_fields = list(fields)
        return new_queryset

    def using_index(self, index_name: str) -> "QuerySet":
        """Apply a backend-specific index hint where supported."""
        if not isinstance(index_name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", index_name
        ):
            raise ValueError("index_name must be a valid SQL identifier")
        new_queryset = self._fast_clone()
        new_queryset._using_index = index_name
        return new_queryset

    def for_update(self) -> "QuerySet":
        """Add FOR UPDATE clause for locking"""
        new_queryset = self._fast_clone()
        new_queryset._for_update = True
        return new_queryset

    def timeout(self, seconds: float) -> "QuerySet":
        """Set a positive per-query timeout in seconds."""
        if (
            not isinstance(seconds, (int, float))
            or isinstance(seconds, bool)
            or seconds <= 0
        ):
            raise ValueError("timeout must be a positive number")
        new_queryset = self._fast_clone()
        new_queryset._timeout = float(seconds)
        return new_queryset

    async def all(self) -> List:
        """Get all records"""
        return await self._fast_execute_query()

    async def first(self) -> Union["ormax.Model", None]:
        """Get first record"""
        result = await self.limit(1).all()
        return result[0] if result else None

    async def last(self) -> Union["ormax.Model", None]:
        """Get last record (reverse order and get first)"""
        # Clone current queryset and reverse order
        new_queryset = self._fast_clone()
        if new_queryset._order_by:
            # Reverse the order
            reversed_order = []
            for order in new_queryset._order_by:
                if " DESC" in order:
                    reversed_order.append(order.replace(" DESC", " ASC"))
                else:
                    reversed_order.append(order.replace(" ASC", " DESC"))
            new_queryset._order_by = reversed_order
        else:
            # If no order, order by primary key descending
            pk_field = self._get_primary_key_field()
            if pk_field:
                new_queryset._order_by = [f"{pk_field} DESC"]
        result = await new_queryset.limit(1).all()
        return result[0] if result else None

    async def get(self, **kwargs) -> Any:
        """Get single record or raise exception"""
        if kwargs:
            qs = self.filter(**kwargs)
        else:
            qs = self
        result = await qs.limit(2).all()
        if len(result) == 0:
            raise self.model_class.DoesNotExist("Object not found")
        elif len(result) > 1:
            raise MultipleObjectsReturned("Multiple objects returned")
        return result[0]

    async def count(self) -> int:
        """Count rows represented by this queryset, including slices/groups."""
        clone = self._fast_clone()
        clone._for_update = False
        if clone._limit is None and clone._offset is None:
            clone._order_by = []
        query, params = clone._build_select_query()
        count_query = f"SELECT COUNT(*) AS count FROM ({query}) AS ormax_count"
        result = await clone._fetch_one(count_query, params)
        return int(result["count"]) if result else 0

    async def exists(self) -> bool:
        """Return whether the queryset contains at least one row."""
        clone = self._fast_clone()
        clone._select_fields = ["1 AS ormax_exists"]
        clone._only_fields = []
        clone._deferred_fields = []
        clone._order_by = []
        clone._limit = 1
        clone._offset = None
        query, params = clone._build_select_query()
        return await clone._fetch_one(query, params) is not None

    async def delete(self) -> int:
        """Bulk delete records"""
        table_name = self.model_class.get_table_name()
        # Build delete query
        query_parts = ["DELETE FROM", table_name]
        params = []
        # Add where clause
        if self._where_clauses:
            query_parts.append(f"WHERE {' AND '.join(self._where_clauses)}")
            params.extend(self._where_params)
        query = " ".join(query_parts)
        if hasattr(self._database, "connection_string"):
            query = format_placeholders(query, self._database.connection_string)
        result = await self._database.connection.execute(query, tuple(params))
        return result if isinstance(result, int) and result >= 0 else 0

    async def update(self, **kwargs) -> int:
        """Bulk update records"""
        table_name = self.model_class.get_table_name()
        # Validate update fields
        set_clauses = []
        set_params = []
        for field_name, value in kwargs.items():
            if field_name not in self.model_class._fields:
                raise DatabaseError(
                    f"Field '{field_name}' does not exist on model '{self.model_class.__name__}'"
                )
            field = self.model_class._fields[field_name]
            from .fields import ForeignKeyField
            if isinstance(field, ForeignKeyField) and hasattr(value, "pk"):
                value = value.pk
            validated_value = field.validate(value)
            if hasattr(validated_value, "resolve_expression"):
                expr_sql, expr_params = validated_value.resolve_expression(query=self, placeholder="?")
                set_clauses.append(f"{field_name} = {expr_sql}")
                set_params.extend(expr_params)
            else:
                set_clauses.append(f"{field_name} = ?")
                set_params.append(field.to_database(validated_value))
        if not set_clauses:
            return 0
        
        # Build where clauses
        where_clauses = []
        where_params = []
        
        # Add where clauses from self._where_clauses
        for clause in self._where_clauses:
            where_clauses.append(clause)
        where_params.extend(self._where_params)
        
        # Build update query
        query_parts = ["UPDATE", table_name, "SET", ", ".join(set_clauses)]
        params = set_params
        
        # Add where clause
        if where_clauses:
            query_parts.append(f"WHERE {' AND '.join(where_clauses)}")
            params.extend(where_params)
        
        query = " ".join(query_parts)
        if hasattr(self._database, "connection_string"):
            query = format_placeholders(query, self._database.connection_string)
        result = await self._database.connection.execute(query, tuple(params))
        return result if isinstance(result, int) and result >= 0 else 0

    async def bulk_create(
        self, objects: Union[List[Dict], List["ormax.Model"]], batch_size: int = 1000
    ) -> List["ormax.Model"]:
        """Bulk create multiple instances efficiently"""
        if not objects:
            return []
        return await self.model_class.bulk_create(objects, batch_size)

    async def values(self, *fields) -> List[Dict]:
        """Return selected values as dictionaries without mutating the queryset."""
        clone = self._fast_clone()
        if fields:
            unknown = set(fields) - set(self.model_class._fields) - set(self._annotations)
            if unknown:
                raise DatabaseError(f"Unknown field(s): {', '.join(sorted(unknown))}")
            clone._select_fields = list(fields)
        return await clone._fast_execute_values()

    async def values_list(self, *fields, flat: bool = False) -> List:
        """Return selected values as tuples, or a flat list for one field."""
        selected_fields = list(fields) if fields else list(self.model_class._fields)
        if flat and len(selected_fields) != 1:
            raise ValueError("flat=True requires exactly one selected field")
        results = await self.values(*selected_fields)
        if flat:
            return [row[selected_fields[0]] for row in results]
        return [tuple(row[field] for field in selected_fields) for row in results]

    async def aiterator(self) -> AsyncIterator:
        """Async iterator for large result sets"""
        results = await self._fast_execute_query()
        for result in results:
            yield result

    async def earliest(self, *fields) -> Union["ormax.Model", None]:
        """Get the earliest record based on field(s)"""
        if not fields:
            # Use primary key or created_at field
            pk_field = self._get_primary_key_field()
            if pk_field:
                fields = (pk_field,)
            else:
                # Look for common timestamp fields
                timestamp_fields = ["created_at", "created", "timestamp"]
                for field in timestamp_fields:
                    if field in self.model_class._fields:
                        fields = (field,)
                        break
        if fields:
            qs = self.order_by(*fields)
            return await qs.first()
        return await self.first()

    async def latest(self, *fields) -> Union["ormax.Model", None]:
        """Get the latest record based on field(s)"""
        if not fields:
            # Use primary key or created_at field
            pk_field = self._get_primary_key_field()
            if pk_field:
                fields = (f"-{pk_field}",)
            else:
                # Look for common timestamp fields
                timestamp_fields = ["created_at", "created", "timestamp"]
                for field in timestamp_fields:
                    if field in self.model_class._fields:
                        fields = (f"-{field}",)
                        break
        if fields:
            qs = self.order_by(*fields)
            return await qs.first()
        return await self.first()

    async def in_bulk(
        self, id_list: Optional[List[Any]] = None, *, field_name: str = "pk"
    ) -> Dict[Any, "ormax.Model"]:
        """Return a dictionary mapping each of the given IDs to the object with that ID"""
        if field_name == "pk":
            pk_field = self._get_primary_key_field()
            if pk_field is None:
                raise ValueError(
                    f"Model {self.model_class.__name__} has no primary key field defined"
                )
            field_name = pk_field
        if id_list is not None:
            if not id_list:
                return {}
            qs = self.filter(**{f"{field_name}__in": id_list})
        else:
            qs = self
        results = await qs.all()
        return {getattr(obj, field_name): obj for obj in results}

    async def create(self, **kwargs) -> "ormax.Model":
        """Create and save a new instance"""
        return await self.model_class.create(**kwargs)

    async def get_or_create(self, defaults: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Any, bool]:
        """
        Get an object matching kwargs, or create a new one with kwargs and defaults.
        Returns (instance, created_boolean).
        """
        try:
            instance = await self.get(**kwargs)
            return instance, False
        except self.model_class.DoesNotExist:
            params = dict(kwargs)
            if defaults:
                params.update(defaults)
            instance = await self.create(**params)
            return instance, True

    async def update_or_create(self, defaults: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Any, bool]:
        """
        Update an object matching kwargs with defaults, or create a new one.
        Returns (instance, created_boolean).
        """
        defaults = defaults or {}
        try:
            instance = await self.get(**kwargs)
            for k, v in defaults.items():
                setattr(instance, k, v)
            await instance.save()
            return instance, False
        except self.model_class.DoesNotExist:
            params = dict(kwargs)
            params.update(defaults)
            instance = await self.create(**params)
            return instance, True

    async def aggregate(self, **aggregations) -> Dict[str, Any]:
        """
        Perform SQL aggregations (e.g. total_price="SUM(price)", count="COUNT(*)").
        Returns dictionary of aggregated values.
        """
        if not aggregations:
            return {}
        table_name = self.model_class.get_table_name()
        select_parts = [f"{expr} AS {alias}" for alias, expr in aggregations.items()]
        query_parts = ["SELECT", ", ".join(select_parts), "FROM", table_name]
        params = []
        if self._where_clauses:
            query_parts.append(f"WHERE {' AND '.join(self._where_clauses)}")
            params.extend(self._where_params)
        query = " ".join(query_parts)
        if hasattr(self._database, "connection_string"):
            query = format_placeholders(query, self._database.connection_string)
        result = await self._database.connection.fetch_one(query, tuple(params))
        return result or {}

    def none(self) -> "QuerySet":
        """Create an empty QuerySet"""
        new_queryset = self._fast_clone()
        # Add impossible condition
        new_queryset._where_clauses.append("1 = 0")
        return new_queryset


    def union(self, *other_querysets) -> "QuerySet":
        """Combine QuerySets using UNION"""
        # This is a simplified implementation
        # In practice, would need to ensure compatible SELECT clauses
        raise NotImplementedError("UNION operation not yet implemented")

    def intersection(self, *other_querysets) -> "QuerySet":
        """Combine QuerySets using INTERSECT"""
        raise NotImplementedError("INTERSECT operation not yet implemented")

    def difference(self, *other_querysets) -> "QuerySet":
        """Combine QuerySets using EXCEPT"""
        raise NotImplementedError("EXCEPT operation not yet implemented")

    def extra(
        self,
        select: Optional[Dict[str, str]] = None,
        where: Optional[List[str]] = None,
        params: Optional[List[Any]] = None,
        tables: Optional[List[str]] = None,
        order_by: Optional[List[str]] = None,
        select_params: Optional[List[Any]] = None,
    ) -> "QuerySet":
        """Add extra SQL clauses"""
        new_queryset = self._fast_clone()
        if select:
            new_queryset._annotations.update(select)
        if where:
            new_queryset._where_clauses.extend(where)
            if params:
                new_queryset._where_params.extend(params)
        if order_by:
            new_queryset._order_by.extend(order_by)
        return new_queryset

    def raw(self, raw_query: str, params: Optional[Tuple] = None) -> "RawQuerySet":
        """Execute raw SQL query"""
        return RawQuerySet(raw_query, params or (), self._database, self.model_class)

    def _fast_clone(self) -> "QuerySet":
        """Ultra-fast cloning with minimal overhead"""
        new_queryset = QuerySet.__new__(QuerySet)
        new_queryset.model_class = self.model_class
        new_queryset._database = self._database
        new_queryset._where_clauses = self._where_clauses.copy()
        new_queryset._where_params = self._where_params.copy()
        new_queryset._order_by = self._order_by.copy()
        new_queryset._limit = self._limit
        new_queryset._offset = self._offset
        new_queryset._select_fields = self._select_fields.copy()
        new_queryset._query_cache = self._query_cache
        new_queryset._distinct = self._distinct
        new_queryset._group_by = self._group_by.copy()
        new_queryset._having = self._having.copy()
        new_queryset._having_params = self._having_params.copy()
        new_queryset._joins = self._joins.copy()
        new_queryset._annotations = self._annotations.copy()
        new_queryset._prefetch_related = self._prefetch_related.copy()
        new_queryset._select_related = self._select_related.copy()
        new_queryset._deferred_fields = self._deferred_fields.copy()
        new_queryset._only_fields = self._only_fields.copy()
        new_queryset._using_index = self._using_index
        new_queryset._for_update = self._for_update
        new_queryset._timeout = self._timeout
        return new_queryset

    def _get_primary_key_field(self) -> Optional[str]:
        """Get the primary key field name"""
        for field_name, field in self.model_class._fields.items():
            if field.primary_key:
                return field_name
        return None

    async def _fetch_all(self, query: str, params: Tuple) -> List[Dict]:
        operation = self._database.connection.fetch_all(query, params)
        if self._timeout is not None:
            return await asyncio.wait_for(operation, timeout=self._timeout)
        return await operation

    async def _fetch_one(self, query: str, params: Tuple) -> Optional[Dict]:
        operation = self._database.connection.fetch_one(query, params)
        if self._timeout is not None:
            return await asyncio.wait_for(operation, timeout=self._timeout)
        return await operation

    async def _fast_execute_query(self) -> List:
        """Ultra-fast query execution with result optimization"""
        if not self._database:
            raise DatabaseError("Model not registered with database")
        # Build query
        query, params = self._build_select_query()
        results = await self._fetch_all(query, params)
        # Convert results to model instances
        instances = self._convert_rows_to_instances(results)
        # Handle prefetch_related if needed
        if self._prefetch_related or self._select_related:
            await self._prefetch_related_objects(instances)
        return instances

    async def _fast_execute_values(self) -> List[Dict]:
        """Execute query and return raw values"""
        if not self._database:
            raise DatabaseError("Model not registered with database")
        query, params = self._build_select_query()
        return await self._fetch_all(query, params)

    def _build_select_query(self) -> Tuple[str, Tuple]:
        """Build SELECT query with all clauses"""
        table_name = self.model_class.get_table_name()
        # ``values()`` takes precedence over ``only()``/``defer()``.
        if self._select_fields != ["*"]:
            select_parts = list(self._select_fields)
        elif self._only_fields:
            select_parts = list(dict.fromkeys(self._only_fields))
        elif self._deferred_fields:
            deferred = set(self._deferred_fields)
            select_parts = [
                name for name in self.model_class._fields if name not in deferred
            ]
        else:
            select_parts = ["*"]

        for alias, expression in self._annotations.items():
            if alias not in select_parts:
                select_parts.append(f"{expression} AS {alias}")
        select_fields = ", ".join(select_parts)
        # Build query parts
        query_parts = ["SELECT"]
        # Add DISTINCT if needed
        if self._distinct:
            query_parts.append("DISTINCT")
        from_clause = table_name
        if self._using_index:
            scheme = self._database.connection_string.split("://", 1)[0].lower()
            if scheme in {"mysql", "mariadb", "aurora"}:
                from_clause = f"{table_name} USE INDEX ({self._using_index})"
            elif scheme == "sqlite":
                from_clause = f"{table_name} INDEXED BY {self._using_index}"
            elif scheme in {"mssql", "microsoft"}:
                from_clause = f"{table_name} WITH (INDEX({self._using_index}))"
            else:
                raise DatabaseError(
                    f"Index hints are not supported for backend '{scheme}'"
                )
        query_parts.extend([select_fields, "FROM", from_clause])
        # Add joins
        if self._joins:
            query_parts.extend(self._joins)
        # Add where clause
        params = []
        if self._where_clauses:
            query_parts.append(f"WHERE {' AND '.join(self._where_clauses)}")
            params.extend(self._where_params)
        # Add group by
        if self._group_by:
            query_parts.append(f"GROUP BY {', '.join(self._group_by)}")
        # Add having
        if self._having:
            query_parts.append(f"HAVING {' AND '.join(self._having)}")
            params.extend(self._having_params)
        # Add order by
        if self._order_by:
            query_parts.append(f"ORDER BY {', '.join(self._order_by)}")
        # Add limit and offset
        if self._limit is not None:
            query_parts.append(f"LIMIT {self._limit}")
        if self._offset is not None:
            query_parts.append(f"OFFSET {self._offset}")
        # Add FOR UPDATE if needed
        if self._for_update:
            query_parts.append("FOR UPDATE")
        query = " ".join(query_parts)
        if hasattr(self._database, "connection_string"):
            query = format_placeholders(query, self._database.connection_string)
        return query, tuple(params)

    def _convert_rows_to_instances(self, rows: List[Dict]) -> List:
        """Convert database rows to model instances using field converters."""
        instances = []
        model_class = self.model_class
        for row in rows:
            converted = dict(row)
            for field_name, field in model_class._fields.items():
                if field_name in converted:
                    converted[field_name] = field.to_python(converted[field_name])
            instance = model_class.__new__(model_class)
            object.__setattr__(instance, "_data", converted)
            object.__setattr__(instance, "_original_data", converted.copy())
            object.__setattr__(instance, "_is_new", False)
            instances.append(instance)
        return instances

    async def _prefetch_related_objects(self, instances: List) -> None:
        """Prefetch related objects in separate queries for high performance"""
        if not (self._prefetch_related or self._select_related) or not instances:
            return

        from collections import defaultdict
        from .fields import ForeignKeyField

        relations = list(dict.fromkeys(self._select_related + self._prefetch_related))
        for relation in relations:
            # Case A: Forward ForeignKey relation
            if relation in self.model_class._fields and isinstance(self.model_class._fields[relation], ForeignKeyField):
                fk_field = self.model_class._fields[relation]
                target_model = self._resolve_target_model(fk_field.to)
                if target_model and not isinstance(target_model, str) and hasattr(target_model, "_fields") and hasattr(target_model, "objects"):
                    target_pk = next((n for n, f in target_model._fields.items() if f.primary_key), "id")
                    fk_vals = {inst._data.get(relation) for inst in instances if inst._data.get(relation) is not None}
                    if fk_vals:
                        rel_objs = await target_model.objects().filter(**{f"{target_pk}__in": list(fk_vals)}).all()
                        rel_map = {getattr(obj, target_pk): obj for obj in rel_objs}
                        for inst in instances:
                            fk_val = inst._data.get(relation)
                            inst._data[f"_prefetched_{relation}"] = rel_map.get(fk_val)

            # Case B: Reverse relation manager
            elif hasattr(self.model_class, "_relationships") and self.model_class._relationships:
                rev_manager = self.model_class._relationships.get_reverse_manager(relation)
                if rev_manager:
                    related_model = rev_manager.related_model_class
                    fk_name = rev_manager.field_name
                    pk_field = next((n for n, f in self.model_class._fields.items() if f.primary_key), "id")
                    pk_vals = {inst._data.get(pk_field) for inst in instances if inst._data.get(pk_field) is not None}
                    if pk_vals:
                        rel_objs = await related_model.objects().filter(**{f"{fk_name}__in": list(pk_vals)}).all()
                        grouped = defaultdict(list)
                        for robj in rel_objs:
                            fk_val = robj._data.get(fk_name)
                            grouped[fk_val].append(robj)
                        for inst in instances:
                            pk_val = inst._data.get(pk_field)
                            inst._data[f"_prefetched_{relation}"] = PrefetchedRelatedList(
                                grouped[pk_val]
                            )



    def __await__(self):
        """Allow await on queryset to get all results"""
        return self.all().__await__()

    def __len__(self):
        """Get count of records (synchronous - use count() for async)"""
        raise DatabaseError("Use await queryset.count() for async count")

    def __aiter__(self):
        """Async iterator"""
        return self.aiterator()

    def __repr__(self):
        return f"<QuerySet for {self.model_class.__name__}>"


class RawQuerySet:
    """Raw SQL query execution"""

    __slots__ = ("raw_query", "params", "_database", "_model_class")

    def __init__(self, raw_query: str, params: Tuple, database, model_class):
        self.raw_query = raw_query
        self.params = params
        self._database = database
        self._model_class = model_class

    async def execute(self) -> List[Dict]:
        """Execute raw query and return results"""
        return await self._database.connection.fetch_all(self.raw_query, self.params)

    async def fetch_one(self) -> Optional[Dict]:
        """Execute raw query and return first result"""
        return await self._database.connection.fetch_one(self.raw_query, self.params)

    async def __aiter__(self):
        """Async iterator for raw query results"""
        results = await self.execute()
        for result in results:
            yield result


# Manager class for model-level query operations
class Manager:
    """Model manager for database operations"""

    def __init__(self, model_class):
        self.model_class = model_class

    def get_queryset(self) -> QuerySet:
        """Get base queryset for this model"""
        return QuerySet(self.model_class)

    def all(self) -> QuerySet:
        """Get all objects"""
        return self.get_queryset()

    def filter(self, **kwargs) -> QuerySet:
        """Filter objects"""
        return self.get_queryset().filter(**kwargs)

    def exclude(self, **kwargs) -> QuerySet:
        """Exclude objects"""
        return self.get_queryset().exclude(**kwargs)

    async def get(self, **kwargs) -> Any:
        """Get single object"""
        return await self.get_queryset().get(**kwargs)

    async def create(self, **kwargs) -> Any:
        """Create and save new object"""
        return await self.model_class.create(**kwargs)

    async def get_or_create(self, defaults: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Any, bool]:
        """Get object or create a new one"""
        return await self.get_queryset().get_or_create(defaults=defaults, **kwargs)

    async def update_or_create(self, defaults: Optional[Dict[str, Any]] = None, **kwargs) -> Tuple[Any, bool]:
        """Update object or create a new one"""
        return await self.get_queryset().update_or_create(defaults=defaults, **kwargs)


    async def bulk_create(self, objects: List, batch_size: int = 1000) -> List:
        """Bulk create objects"""
        return await self.get_queryset().bulk_create(objects, batch_size)

    async def count(self) -> int:
        """Count objects"""
        return await self.get_queryset().count()

    async def exists(self) -> bool:
        """Check if objects exist"""
        return await self.get_queryset().exists()

    def order_by(self, *fields) -> QuerySet:
        """Order objects"""
        return self.get_queryset().order_by(*fields)

    def distinct(self) -> QuerySet:
        """Get distinct objects"""
        return self.get_queryset().distinct()

    async def first(self) -> Any:
        """Get first object"""
        return await self.get_queryset().first()

    async def last(self) -> Any:
        """Get last object"""
        return await self.get_queryset().last()


# Async aggregation functions
class Aggregation:
    """Aggregation functions for QuerySet"""

    @staticmethod
    async def sum(queryset: QuerySet, field: str) -> Optional[float]:
        """Calculate sum of field"""
        table_name = queryset.model_class.get_table_name()
        query = f"SELECT SUM({field}) as sum_result FROM {table_name}"
        params = []
        if queryset._where_clauses:
            query += f" WHERE {' AND '.join(queryset._where_clauses)}"
            params.extend(queryset._where_params)
        if hasattr(queryset._database, "connection_string"):
            query = format_placeholders(query, queryset._database.connection_string)
        result = await queryset._database.connection.fetch_one(query, tuple(params))
        return (
            result["sum_result"] if result and result["sum_result"] is not None else 0
        )

    @staticmethod
    async def avg(queryset: QuerySet, field: str) -> Optional[float]:
        """Calculate average of field"""
        table_name = queryset.model_class.get_table_name()
        query = f"SELECT AVG({field}) as avg_result FROM {table_name}"
        params = []
        if queryset._where_clauses:
            query += f" WHERE {' AND '.join(queryset._where_clauses)}"
            params.extend(queryset._where_params)
        if hasattr(queryset._database, "connection_string"):
            query = format_placeholders(query, queryset._database.connection_string)
        result = await queryset._database.connection.fetch_one(query, tuple(params))
        return (
            result["avg_result"] if result and result["avg_result"] is not None else 0
        )

    @staticmethod
    async def min(queryset: QuerySet, field: str) -> Optional[Any]:
        """Find minimum value of field"""
        table_name = queryset.model_class.get_table_name()
        query = f"SELECT MIN({field}) as min_result FROM {table_name}"
        params = []
        if queryset._where_clauses:
            query += f" WHERE {' AND '.join(queryset._where_clauses)}"
            params.extend(queryset._where_params)
        if hasattr(queryset._database, "connection_string"):
            query = format_placeholders(query, queryset._database.connection_string)
        result = await queryset._database.connection.fetch_one(query, tuple(params))
        return (
            result["min_result"]
            if result and result["min_result"] is not None
            else None
        )

    @staticmethod
    async def max(queryset: QuerySet, field: str) -> Optional[Any]:
        """Find maximum value of field"""
        table_name = queryset.model_class.get_table_name()
        query = f"SELECT MAX({field}) as max_result FROM {table_name}"
        params = []
        if queryset._where_clauses:
            query += f" WHERE {' AND '.join(queryset._where_clauses)}"
            params.extend(queryset._where_params)
        if hasattr(queryset._database, "connection_string"):
            query = format_placeholders(query, queryset._database.connection_string)
        result = await queryset._database.connection.fetch_one(query, tuple(params))
        return (
            result["max_result"]
            if result and result["max_result"] is not None
            else None
        )


# Convenience functions
async def sync_to_async(func: Callable, *args, **kwargs):
    """Run sync function in async context"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
