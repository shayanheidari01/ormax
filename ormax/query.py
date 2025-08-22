# ormax/query.py
from typing import List, Optional, Dict, Any, Tuple, Callable, AsyncIterator, Union
import logging
import asyncio

import ormax
from .exceptions import DatabaseError

logger = logging.getLogger(__name__)


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

    def filter(self, **kwargs) -> "QuerySet":
        """Add filter conditions"""
        new_queryset = self._fast_clone()
        for field_name, value in kwargs.items():
            if field_name not in self.model_class._fields:
                # Check if it's a related field or annotation
                if field_name not in self._annotations:
                    raise DatabaseError(
                        f"Field '{field_name}' does not exist on model '{self.model_class.__name__}'"
                    )
            # همیشه از ? به عنوان پلاس‌هولدر استفاده کنید
            new_queryset._where_clauses.append(f"{field_name} = ?")
            new_queryset._where_params.append(value)
        return new_queryset

    def exclude(self, **kwargs) -> "QuerySet":
        """Add exclude conditions"""
        new_queryset = self._fast_clone()
        for field_name, value in kwargs.items():
            if field_name not in self.model_class._fields:
                if field_name not in self._annotations:
                    raise DatabaseError(
                        f"Field '{field_name}' does not exist on model '{self.model_class.__name__}'"
                    )
            
            # برای PostgreSQL، پلاس‌هولدرها را به شکل $1, $2 و ... می‌سازیم
            if "postgresql" in self._database.connection_string:
                placeholder = f"${len(new_queryset._where_params) + 1}"
            else:
                placeholder = self._database.connection.get_placeholder()
                
            new_queryset._where_clauses.append(f"{field_name} != {placeholder}")
            new_queryset._where_params.append(value)
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
        """Add limit clause"""
        new_queryset = self._fast_clone()
        new_queryset._limit = limit
        return new_queryset

    def offset(self, offset: int) -> "QuerySet":
        """Add offset clause"""
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
        """Select related objects in the same query (JOIN)"""
        new_queryset = self._fast_clone()
        new_queryset._select_related.extend(relations)
        return new_queryset

    def prefetch_related(self, *relations) -> "QuerySet":
        """Prefetch related objects in separate queries"""
        new_queryset = self._fast_clone()
        new_queryset._prefetch_related.extend(relations)
        return new_queryset

    def defer(self, *fields) -> "QuerySet":
        """Defer loading of specific fields"""
        new_queryset = self._fast_clone()
        new_queryset._deferred_fields.extend(fields)
        return new_queryset

    def only(self, *fields) -> "QuerySet":
        """Load only specific fields"""
        new_queryset = self._fast_clone()
        new_queryset._only_fields.extend(fields)
        return new_queryset

    def using_index(self, index_name: str) -> "QuerySet":
        """Force using specific index"""
        new_queryset = self._fast_clone()
        new_queryset._using_index = index_name
        return new_queryset

    def for_update(self) -> "QuerySet":
        """Add FOR UPDATE clause for locking"""
        new_queryset = self._fast_clone()
        new_queryset._for_update = True
        return new_queryset

    def timeout(self, seconds: int) -> "QuerySet":
        """Set query timeout"""
        new_queryset = self._fast_clone()
        new_queryset._timeout = seconds
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

    async def get(self, **kwargs) -> object:
        """Get single record or raise exception"""
        if kwargs:
            qs = self.filter(**kwargs)
        else:
            qs = self
        result = await qs.limit(2).all()
        if len(result) == 0:
            raise self.model_class.DoesNotExist("Object not found")
        elif len(result) > 1:
            raise DatabaseError("Multiple objects returned")
        return result[0]

    async def count(self) -> int:
        """Count records efficiently"""
        table_name = self.model_class.get_table_name()
        # Build query parts
        query_parts = ["SELECT COUNT(*) as count FROM", table_name]
        params = []
        # Add joins if needed
        if self._joins:
            query_parts.extend(self._joins)
        # Add where clause
        if self._where_clauses:
            query_parts.append(f"WHERE {' AND '.join(self._where_clauses)}")
            params.extend(self._where_params)
        # Add group by and having if present
        if self._group_by:
            query_parts.append(f"GROUP BY {', '.join(self._group_by)}")
            if self._having:
                query_parts.append(f"HAVING {' AND '.join(self._having)}")
                params.extend(self._having_params)
        query = " ".join(query_parts)
        result = await self._database.connection.fetch_one(query, tuple(params))
        return result["count"] if result else 0

    async def exists(self) -> bool:
        """Check if any records exist"""
        # Optimize by limiting to 1
        count = await self.limit(1).count()
        return count > 0

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
        await self._database.connection.execute(query, tuple(params))
        return 1  # Simplified return

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
            validated_value = field.validate(value)
            # همیشه از ? به عنوان پلاس‌هولدر استفاده کنید
            set_clauses.append(f"{field_name} = ?")
            set_params.append(validated_value)
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
        await self._database.connection.execute(query, tuple(params))
        return 1  # Simplified return

    async def bulk_create(
        self, objects: Union[List[Dict], List["ormax.Model"]], batch_size: int = 1000
    ) -> List["ormax.Model"]:
        """Bulk create multiple instances efficiently"""
        if not objects:
            return []
        return await self.model_class.bulk_create(objects, batch_size)

    async def values(self, *fields) -> List[Dict]:
        """Return values as dictionaries"""
        if fields:
            self._select_fields = list(fields)
        return await self._fast_execute_values()

    async def values_list(self, *fields, flat: bool = False) -> List:
        """Return values as lists or tuples"""
        if fields:
            self._select_fields = list(fields)
        results = await self._fast_execute_values()
        if flat and len(fields) == 1:
            # Return flat list
            return [row[fields[0]] for row in results]
        else:
            # Return list of tuples
            return [tuple(row[field] for field in fields) for row in results]

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
        self, id_list: List[Any] = None, *, field_name: str = "pk"
    ) -> Dict[Any, "ormax.Model"]:
        """Return a dictionary mapping each of the given IDs to the object with that ID"""
        if field_name == "pk":
            field_name = self._get_primary_key_field()
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
        select: Dict[str, str] = None,
        where: List[str] = None,
        params: List[Any] = None,
        tables: List[str] = None,
        order_by: List[str] = None,
        select_params: List[Any] = None,
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

    def raw(self, raw_query: str, params: Tuple = None) -> "RawQuerySet":
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

    async def _fast_execute_query(self) -> List:
        """Ultra-fast query execution with result optimization"""
        if not self._database:
            raise DatabaseError("Model not registered with database")
        # Build query
        query, params = self._build_select_query()
        results = await self._database.connection.fetch_all(query, params)
        # Convert results to model instances
        instances = self._convert_rows_to_instances(results)
        # Handle prefetch_related if needed
        if self._prefetch_related:
            await self._prefetch_related_objects(instances)
        return instances

    async def _fast_execute_values(self) -> List[Dict]:
        """Execute query and return raw values"""
        if not self._database:
            raise DatabaseError("Model not registered with database")
        query, params = self._build_select_query()
        return await self._database.connection.fetch_all(query, params)

    def _build_select_query(self) -> Tuple[str, Tuple]:
        """Build SELECT query with all clauses"""
        table_name = self.model_class.get_table_name()
        # Determine select fields
        if self._annotations:
            # Include annotations in select
            select_parts = []
            if "*" in self._select_fields or not self._select_fields:
                select_parts.append("*")
            else:
                select_parts.extend(self._select_fields)
            for alias, expression in self._annotations.items():
                select_parts.append(f"{expression} AS {alias}")
            select_fields = ", ".join(select_parts)
        else:
            select_fields = (
                ", ".join(self._select_fields)
                if "*" not in self._select_fields
                else "*"
            )
        # Build query parts
        query_parts = ["SELECT"]
        # Add DISTINCT if needed
        if self._distinct:
            query_parts.append("DISTINCT")
        query_parts.extend([select_fields, "FROM", table_name])
        # Add joins
        if self._joins:
            query_parts.extend(self._joins)
        # Add where clause
        params = []
        if self._where_clauses:
            # برای PostgreSQL، پلاس‌هولدرها را به درستی شماره‌گذاری می‌کنیم
            if "postgresql" in self._database.connection_string:
                # شماره‌گذاری پلاس‌هولدرها از 1 شروع می‌شود
                where_clauses = []
                for i, clause in enumerate(self._where_clauses):
                    # جایگزینی ? با $i+1
                    where_clauses.append(clause.replace("?", f"${i+1}"))
                query_parts.append(f"WHERE {' AND '.join(where_clauses)}")
            else:
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
        return query, tuple(params)

    def _convert_rows_to_instances(self, rows: List[Dict]) -> List:
        """Convert database rows to model instances efficiently"""
        instances = []
        model_class = self.model_class
        fields = model_class._fields
        for row in rows:
            instance = model_class.__new__(model_class)
            object.__setattr__(instance, "_data", dict(row))
            object.__setattr__(instance, "_original_data", instance._data.copy())
            object.__setattr__(instance, "_is_new", False)
            instances.append(instance)
        return instances

    async def _prefetch_related_objects(self, instances: List) -> None:
        """Prefetch related objects in separate queries for better performance"""
        if not self._prefetch_related or not instances:
            return
        # This is a simplified implementation
        # In practice, would need to handle foreign key relationships properly
        pass

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

    def get(self, **kwargs) -> object:
        """Get single object"""
        return self.get_queryset().get(**kwargs)

    async def create(self, **kwargs) -> object:
        """Create and save new object"""
        return await self.model_class.create(**kwargs)

    def bulk_create(self, objects: List, batch_size: int = 1000) -> QuerySet:
        """Bulk create objects"""
        return self.get_queryset().bulk_create(objects, batch_size)

    def count(self) -> QuerySet:
        """Count objects"""
        return self.get_queryset().count()

    def exists(self) -> QuerySet:
        """Check if objects exist"""
        return self.get_queryset().exists()

    def order_by(self, *fields) -> QuerySet:
        """Order objects"""
        return self.get_queryset().order_by(*fields)

    def distinct(self) -> QuerySet:
        """Get distinct objects"""
        return self.get_queryset().distinct()

    def first(self) -> QuerySet:
        """Get first object"""
        return self.get_queryset().first()

    def last(self) -> QuerySet:
        """Get last object"""
        return self.get_queryset().last()


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
        result = await queryset._database.connection.fetch_one(query, tuple(params))
        return (
            result["max_result"]
            if result and result["max_result"] is not None
            else None
        )


# Convenience functions
async def sync_to_async(func: Callable, *args, **kwargs):
    """Run sync function in async context"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
