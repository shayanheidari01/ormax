# database_orm/core.py
import asyncio
import logging
from typing import Dict, Any, Optional, List, Type, Union
from abc import ABC, abstractmethod
import json
from contextlib import asynccontextmanager

from .fields import Field
from .query import QuerySet
from .exceptions import DatabaseError, ValidationError

logger = logging.getLogger(__name__)


class DatabaseConnection(ABC):
    """Abstract base class for database connections"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
    
    @abstractmethod
    async def connect(self):
        pass
    
    @abstractmethod
    async def disconnect(self):
        pass
    
    @abstractmethod
    async def execute(self, query: str, params: tuple = None) -> Any:
        pass
    
    @abstractmethod
    async def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        pass
    
    @abstractmethod
    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        pass
    
    @abstractmethod
    def get_placeholder(self) -> str:
        pass


class MySQLConnection(DatabaseConnection):
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self.aiomysql = None
    
    async def connect(self):
        try:
            import aiomysql
            self.aiomysql = aiomysql
            # Parse connection string
            import urllib.parse
            parsed = urllib.parse.urlparse(self.connection_string)
            
            self.pool = await aiomysql.create_pool(
                host=parsed.hostname or 'localhost',
                port=parsed.port or 3306,
                user=parsed.username,
                password=parsed.password,
                db=parsed.path.lstrip('/'),
                charset='utf8mb4',
                autocommit=True,
                pool_recycle=3600
            )
        except ImportError:
            raise DatabaseError("aiomysql is not installed")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to MySQL: {e}")
    
    async def disconnect(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
    
    async def execute(self, query: str, params: tuple = None) -> Any:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            async with conn.cursor(self.aiomysql.DictCursor) as cur:
                await cur.execute(query, params or ())
                return cur.lastrowid
    
    async def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            async with conn.cursor(self.aiomysql.DictCursor) as cur:
                await cur.execute(query, params or ())
                result = await cur.fetchone()
                return dict(result) if result else None
    
    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            async with conn.cursor(self.aiomysql.DictCursor) as cur:
                await cur.execute(query, params or ())
                results = await cur.fetchall()
                return [dict(row) for row in results]
    
    def get_placeholder(self) -> str:
        return "%s"


class PostgreSQLConnection(DatabaseConnection):
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self.asyncpg = None
    
    async def connect(self):
        try:
            import asyncpg
            self.asyncpg = asyncpg
            self.pool = await asyncpg.create_pool(self.connection_string)
        except ImportError:
            raise DatabaseError("asyncpg is not installed")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to PostgreSQL: {e}")
    
    async def disconnect(self):
        if self.pool:
            await self.pool.close()
    
    async def execute(self, query: str, params: tuple = None) -> Any:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *(params or ()))
            # For INSERT, try to get the last inserted ID
            if query.strip().upper().startswith('INSERT'):
                try:
                    # Try to get last inserted ID
                    last_id_row = await conn.fetchval("SELECT LASTVAL()")
                    return last_id_row
                except:
                    return None
            return result
    
    async def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(query, *(params or ()))
            return dict(result) if result else None
    
    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            results = await conn.fetch(query, *(params or ()))
            return [dict(row) for row in results]
    
    def get_placeholder(self) -> str:
        return "$"


class SQLiteConnection(DatabaseConnection):
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self.aiosqlite = None
        self._connection = None
    
    async def connect(self):
        try:
            import aiosqlite
            self.aiosqlite = aiosqlite
            # Extract database path from connection string
            import urllib.parse
            parsed = urllib.parse.urlparse(self.connection_string)
            db_path = parsed.path.lstrip('/')
            
            # Create single connection instead of pool for SQLite
            self._connection = await aiosqlite.connect(db_path)
            await self._connection.execute("PRAGMA foreign_keys = ON")
        except ImportError:
            raise DatabaseError("aiosqlite is not installed")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to SQLite: {e}")
    
    async def disconnect(self):
        if self._connection:
            await self._connection.close()
            self._connection = None
    
    async def execute(self, query: str, params: tuple = None) -> Any:
        if not self._connection:
            raise DatabaseError("Database not connected")
        try:
            cursor = await self._connection.execute(query, params or ())
            await self._connection.commit()
            return cursor.lastrowid
        except Exception as e:
            await self._connection.rollback()
            raise
    
    async def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        if not self._connection:
            raise DatabaseError("Database not connected")
        cursor = await self._connection.execute(query, params or ())
        row = await cursor.fetchone()
        if row is None:
            return None
        # Get column names
        column_names = [description[0] for description in cursor.description]
        return dict(zip(column_names, row))
    
    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        if not self._connection:
            raise DatabaseError("Database not connected")
        cursor = await self._connection.execute(query, params or ())
        rows = await cursor.fetchall()
        if not rows:
            return []
        # Get column names
        column_names = [description[0] for description in cursor.description]
        return [dict(zip(column_names, row)) for row in rows]
    
    def get_placeholder(self) -> str:
        return "?"


class MariaDBConnection(MySQLConnection):
    """MariaDB connection is similar to MySQL"""
    pass


class Database:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection: DatabaseConnection = self._create_connection()
        self.models: Dict[str, Type['Model']] = {}
        self._connected = False
    
    def _create_connection(self) -> DatabaseConnection:
        if self.connection_string.startswith('mysql://'):
            return MySQLConnection(self.connection_string)
        elif self.connection_string.startswith('postgresql://') or self.connection_string.startswith('postgres://'):
            return PostgreSQLConnection(self.connection_string)
        elif self.connection_string.startswith('sqlite://'):
            return SQLiteConnection(self.connection_string)
        elif self.connection_string.startswith('mariadb://'):
            return MariaDBConnection(self.connection_string)
        else:
            raise DatabaseError(f"Unsupported database: {self.connection_string}")
    
    async def connect(self):
        await self.connection.connect()
        self._connected = True
        logger.info("Database connected successfully")
    
    async def disconnect(self):
        await self.connection.disconnect()
        self._connected = False
        logger.info("Database disconnected")
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions"""
        if not self._connected:
            raise DatabaseError("Database not connected")
        
        # Start transaction
        await self.connection.execute("BEGIN")
        transaction_active = True
        try:
            yield self
            if transaction_active:
                await self.connection.execute("COMMIT")
                transaction_active = False
        except Exception as e:
            if transaction_active:
                try:
                    await self.connection.execute("ROLLBACK")
                except Exception:
                    pass  # Ignore rollback errors in case transaction is already committed
            logger.error(f"Transaction rolled back due to: {e}")
            raise
    
    def register_model(self, model_class: Type['Model']):
        """Register a model with the database"""
        self.models[model_class.__name__] = model_class
        model_class._database = self
    
    async def create_tables(self):
        """Create all registered tables"""
        for model_name, model_class in self.models.items():
            await self.create_table(model_class)
    
    async def create_table(self, model_class: Type['Model']):
        """Create table for a specific model"""
        table_name = model_class.get_table_name()
        fields = []
        
        for field_name, field in model_class._fields.items():
            field_sql = f"{field_name} {field.get_sql_type()}"
            
            if field.primary_key:
                field_sql += " PRIMARY KEY"
                # Handle auto increment for different databases
                if field.auto_increment:
                    if 'sqlite' in self.connection_string:
                        field_sql += " AUTOINCREMENT"
                    elif 'mysql' in self.connection_string or 'mariadb' in self.connection_string:
                        field_sql += " AUTO_INCREMENT"
                    # PostgreSQL handles auto increment differently with SERIAL
            
            if not field.nullable:
                field_sql += " NOT NULL"
            
            if field.default is not None and not field.primary_key:
                field_sql += f" DEFAULT {field.get_default_sql()}"
            
            if field.unique:
                field_sql += " UNIQUE"
            
            fields.append(field_sql)
        
        # Add foreign key constraints
        for field_name, field in model_class._fields.items():
            if hasattr(field, 'foreign_key') and field.foreign_key:
                ref_table, ref_field = field.foreign_key.split('.')
                constraint_name = f"fk_{table_name}_{field_name}"
                fields.append(
                    f"CONSTRAINT {constraint_name} FOREIGN KEY ({field_name}) "
                    f"REFERENCES {ref_table}({ref_field})"
                )
        
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(fields)})"
        await self.connection.execute(query)
        logger.info(f"Table {table_name} created/verified")


class ModelMeta(type):
    def __new__(cls, name, bases, attrs):
        # Don't process base Model class
        if name == 'Model':
            return super().__new__(cls, name, bases, attrs)
        
        # Collect fields
        fields = {}
        meta = attrs.get('_meta', {})
        
        for key, value in list(attrs.items()):
            if isinstance(value, Field):
                fields[key] = value
                attrs.pop(key)
        
        attrs['_fields'] = fields
        attrs['_meta'] = meta
        
        return super().__new__(cls, name, bases, attrs)


class Model(metaclass=ModelMeta):
    _database: Database = None
    _fields: Dict[str, Field] = {}
    _meta: Dict[str, Any] = {}
    
    def __init__(self, **kwargs):
        self._data = {}
        self._original_data = {}
        self._is_new = True
        
        # Set default values and initialize fields
        for field_name, field in self._fields.items():
            if field_name in kwargs:
                setattr(self, field_name, kwargs[field_name])
            elif field.default is not None:
                setattr(self, field_name, field.default)
            else:
                setattr(self, field_name, None)
    
    def __setattr__(self, name, value):
        if name.startswith('_') or name not in self._fields:
            super().__setattr__(name, value)
        else:
            # Validate field value
            field = self._fields[name]
            validated_value = field.validate(value)
            self._data[name] = validated_value
    
    def __getattr__(self, name):
        if name in self._fields:
            return self._data.get(name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    @classmethod
    def get_table_name(cls) -> str:
        """Get table name for this model - simplified approach"""
        # Check _meta first
        if hasattr(cls, '_meta') and cls._meta.get('table_name'):
            return cls._meta.get('table_name')
        # Check for table_name class attribute
        if hasattr(cls, 'table_name'):
            return cls.table_name
        # Default to class name lowercase
        return cls.__name__.lower()
    
    @classmethod
    def objects(cls) -> QuerySet:
        """Return a QuerySet for this model"""
        return QuerySet(cls)
    
    async def save(self):
        """Save the model instance to database"""
        if not self._database:
            raise DatabaseError("Model not registered with database")
        
        # Validate all fields
        for field_name, field in self._fields.items():
            if not field.nullable and self._data.get(field_name) is None:
                if not (field.primary_key and self._is_new and field.auto_increment):
                    raise ValidationError(f"Field '{field_name}' cannot be null")
        
        if self._is_new:
            await self._insert()
        else:
            await self._update()
    
    async def _insert(self):
        """Insert new record"""
        table_name = self.get_table_name()
        columns = []
        values = []
        placeholders = []
        
        for field_name, field in self._fields.items():
            # Skip auto increment primary keys for insert
            if field.primary_key and field.auto_increment and self._data.get(field_name) is None:
                continue
            
            if self._data.get(field_name) is not None or not field.primary_key:
                columns.append(field_name)
                values.append(self._data.get(field_name))
                placeholders.append(self._database.connection.get_placeholder())
        
        if columns:
            query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            lastrowid = await self._database.connection.execute(query, tuple(values))
        else:
            # Insert empty row
            query = f"INSERT INTO {table_name} DEFAULT VALUES"
            lastrowid = await self._database.connection.execute(query)
        
        # Set primary key if it's auto-generated
        for field_name, field in self._fields.items():
            if field.primary_key and field.auto_increment and self._data.get(field_name) is None:
                self._data[field_name] = lastrowid
                break
        
        self._is_new = False
        self._original_data = self._data.copy()
    
    async def _update(self):
        """Update existing record"""
        table_name = self.get_table_name()
        set_clauses = []
        values = []
        
        # Only update changed fields
        for field_name, field in self._fields.items():
            if field.primary_key:
                continue
            
            if self._data.get(field_name) != self._original_data.get(field_name):
                set_clauses.append(f"{field_name} = {self._database.connection.get_placeholder()}")
                values.append(self._data.get(field_name))
        
        if not set_clauses:
            return  # No changes to update
        
        # Get primary key value for WHERE clause
        pk_field = next((name for name, field in self._fields.items() if field.primary_key), None)
        if not pk_field:
            raise DatabaseError("No primary key defined")
        
        pk_value = self._data.get(pk_field)
        values.append(pk_value)
        
        query = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {pk_field} = {self._database.connection.get_placeholder()}"
        await self._database.connection.execute(query, tuple(values))
        self._original_data = self._data.copy()
    
    async def delete(self):
        """Delete the model instance from database"""
        if not self._database:
            raise DatabaseError("Model not registered with database")
        
        pk_field = next((name for name, field in self._fields.items() if field.primary_key), None)
        if not pk_field:
            raise DatabaseError("No primary key defined")
        
        pk_value = self._data.get(pk_field)
        if pk_value is None:
            raise DatabaseError("Cannot delete unsaved instance")
        
        table_name = self.get_table_name()
        query = f"DELETE FROM {table_name} WHERE {pk_field} = {self._database.connection.get_placeholder()}"
        await self._database.connection.execute(query, (pk_value,))
        
        self._is_new = True
    
    @classmethod
    async def create(cls, **kwargs) -> 'Model':
        """Create and save a new instance"""
        instance = cls(**kwargs)
        await instance.save()
        return instance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to dictionary"""
        return self._data.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Model':
        """Create model instance from dictionary"""
        return cls(**data)
    
    def __repr__(self):
        pk_field = next((name for name, field in self._fields.items() if field.primary_key), 'id')
        pk_value = self._data.get(pk_field, 'None')
        return f"<{self.__class__.__name__}: {pk_field}={pk_value}>"


# Add DoesNotExist exception to Model class
Model.DoesNotExist = type('DoesNotExist', (Exception,), {})