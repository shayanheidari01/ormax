# database_orm/query.py
from typing import List, Optional, Dict, Any, Union
import logging
from .exceptions import DatabaseError

logger = logging.getLogger(__name__)


class QuerySet:
    def __init__(self, model_class):
        self.model_class = model_class
        self._database = model_class._database
        self._where_clauses = []
        self._where_params = []
        self._order_by = []
        self._limit = None
        self._offset = None
        self._select_fields = ['*']
    
    def filter(self, **kwargs) -> 'QuerySet':
        """Add filter conditions"""
        new_queryset = self._clone()
        
        for field_name, value in kwargs.items():
            if field_name not in self.model_class._fields:
                raise DatabaseError(f"Field '{field_name}' does not exist on model '{self.model_class.__name__}'")
            
            field = self.model_class._fields[field_name]
            placeholder = self._database.connection.get_placeholder()
            new_queryset._where_clauses.append(f"{field_name} = {placeholder}")
            new_queryset._where_params.append(value)
        
        return new_queryset
    
    def exclude(self, **kwargs) -> 'QuerySet':
        """Add exclude conditions"""
        new_queryset = self._clone()
        
        for field_name, value in kwargs.items():
            if field_name not in self.model_class._fields:
                raise DatabaseError(f"Field '{field_name}' does not exist on model '{self.model_class.__name__}'")
            
            field = self.model_class._fields[field_name]
            placeholder = self._database.connection.get_placeholder()
            new_queryset._where_clauses.append(f"{field_name} != {placeholder}")
            new_queryset._where_params.append(value)
        
        return new_queryset
    
    def order_by(self, *fields) -> 'QuerySet':
        """Add order by clause"""
        new_queryset = self._clone()
        
        for field in fields:
            if field.startswith('-'):
                field_name = field[1:]
                direction = 'DESC'
            else:
                field_name = field
                direction = 'ASC'
            
            if field_name not in self.model_class._fields:
                raise DatabaseError(f"Field '{field_name}' does not exist on model '{self.model_class.__name__}'")
            
            new_queryset._order_by.append(f"{field_name} {direction}")
        
        return new_queryset
    
    def limit(self, limit: int) -> 'QuerySet':
        """Add limit clause"""
        new_queryset = self._clone()
        new_queryset._limit = limit
        return new_queryset
    
    def offset(self, offset: int) -> 'QuerySet':
        """Add offset clause"""
        new_queryset = self._clone()
        new_queryset._offset = offset
        return new_queryset
    
    def select_related(self, *fields) -> 'QuerySet':
        """Select specific fields"""
        new_queryset = self._clone()
        new_queryset._select_fields = list(fields) if fields else ['*']
        return new_queryset
    
    async def all(self) -> List:
        """Get all records"""
        return await self._execute_query()
    
    async def first(self) -> Optional:
        """Get first record"""
        result = await self.limit(1).all()
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
        """Count records"""
        table_name = self.model_class.get_table_name()
        query_parts = [f"SELECT COUNT(*) as count FROM {table_name}"]
        
        if self._where_clauses:
            query_parts.append(f"WHERE {' AND '.join(self._where_clauses)}")
        
        query = " ".join(query_parts)
        result = await self._database.connection.fetch_one(query, tuple(self._where_params))
        return result['count'] if result else 0
    
    async def exists(self) -> bool:
        """Check if any records exist"""
        count = await self.limit(1).count()
        return count > 0
    
    async def delete(self) -> int:
        """Delete records matching the query"""
        table_name = self.model_class.get_table_name()
        query_parts = [f"DELETE FROM {table_name}"]
        
        if self._where_clauses:
            query_parts.append(f"WHERE {' AND '.join(self._where_clauses)}")
        
        query = " ".join(query_parts)
        await self._database.connection.execute(query, tuple(self._where_params))
        return 1  # Simplified return
    
    async def update(self, **kwargs) -> int:
        """Update records matching the query"""
        table_name = self.model_class.get_table_name()
        
        # Validate update fields
        set_clauses = []
        set_params = []
        for field_name, value in kwargs.items():
            if field_name not in self.model_class._fields:
                raise DatabaseError(f"Field '{field_name}' does not exist on model '{self.model_class.__name__}'")
            
            field = self.model_class._fields[field_name]
            validated_value = field.validate(value)
            placeholder = self._database.connection.get_placeholder()
            set_clauses.append(f"{field_name} = {placeholder}")
            set_params.append(validated_value)
        
        if not set_clauses:
            return 0
        
        query_parts = [f"UPDATE {table_name} SET {', '.join(set_clauses)}"]
        
        if self._where_clauses:
            query_parts.append(f"WHERE {' AND '.join(self._where_clauses)}")
            params = set_params + self._where_params
        else:
            params = set_params
        
        query = " ".join(query_parts)
        await self._database.connection.execute(query, tuple(params))
        return 1  # Simplified return
    
    def _clone(self) -> 'QuerySet':
        """Create a copy of the current queryset"""
        new_queryset = QuerySet(self.model_class)
        new_queryset._where_clauses = self._where_clauses.copy()
        new_queryset._where_params = self._where_params.copy()
        new_queryset._order_by = self._order_by.copy()
        new_queryset._limit = self._limit
        new_queryset._offset = self._offset
        new_queryset._select_fields = self._select_fields.copy()
        return new_queryset
    
    async def _execute_query(self) -> List:
        """Execute the query and return results"""
        if not self._database:
            raise DatabaseError("Model not registered with database")
        
        table_name = self.model_class.get_table_name()
        select_fields = ', '.join(self._select_fields) if '*' not in self._select_fields else '*'
        
        query_parts = [f"SELECT {select_fields} FROM {table_name}"]
        
        if self._where_clauses:
            query_parts.append(f"WHERE {' AND '.join(self._where_clauses)}")
        
        if self._order_by:
            query_parts.append(f"ORDER BY {', '.join(self._order_by)}")
        
        if self._limit is not None:
            query_parts.append(f"LIMIT {self._limit}")
        
        if self._offset is not None:
            query_parts.append(f"OFFSET {self._offset}")
        
        query = " ".join(query_parts)
        results = await self._database.connection.fetch_all(query, tuple(self._where_params))
        
        # Convert results to model instances
        instances = []
        for row in results:
            instance = self.model_class()
            instance._is_new = False
            instance._data = dict(row)  # row is already converted to dict
            instance._original_data = instance._data.copy()
            instances.append(instance)
        
        return instances
    
    def __await__(self):
        """Allow await on queryset to get all results"""
        return self.all().__await__()