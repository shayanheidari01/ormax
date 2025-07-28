# ormax/core.py
import asyncio
import logging
from typing import Dict, Any, Optional, List, Type, Union, Tuple
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from collections import defaultdict
import weakref

from .fields import Field
from .query import QuerySet
from .exceptions import DatabaseError, ValidationError

logger = logging.getLogger(__name__)


class DatabaseConnection(ABC):
    """Abstract base class for database connections with connection pooling"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
        self._placeholder = None
    
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
        self._placeholder = "%s"
    
    async def connect(self):
        try:
            import aiomysql
            self.aiomysql = aiomysql
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
                pool_recycle=3600,
                minsize=5,
                maxsize=20
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
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, params or ())
                return cur.lastrowid
    
    async def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, params or ())
                result = await cur.fetchone()
                return dict(result) if result else None
    
    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, params or ())
                results = await cur.fetchall()
                return [dict(row) for row in results]
    
    def get_placeholder(self) -> str:
        return self._placeholder


class PostgreSQLConnection(DatabaseConnection):
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self.asyncpg = None
        self._placeholder = "$"
    
    async def connect(self):
        try:
            import asyncpg
            self.asyncpg = asyncpg
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
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
            if query.strip().upper().startswith('INSERT'):
                try:
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
        return self._placeholder


class SQLiteConnection(DatabaseConnection):
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self.aiosqlite = None
        self._connection = None
        self._placeholder = "?"
        self._prepared_statements = {}
    
    async def connect(self):
        try:
            import aiosqlite
            self.aiosqlite = aiosqlite
            import urllib.parse
            parsed = urllib.parse.urlparse(self.connection_string)
            db_path = parsed.path.lstrip('/')
            
            self._connection = await aiosqlite.connect(db_path)
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._connection.execute("PRAGMA journal_mode = WAL")
            await self._connection.execute("PRAGMA synchronous = NORMAL")
        except ImportError:
            raise DatabaseError("aiosqlite is not installed")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to SQLite: {e}")
    
    async def disconnect(self):
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._prepared_statements.clear()
    
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
        column_names = [description[0] for description in cursor.description]
        return dict(zip(column_names, row))
    
    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        if not self._connection:
            raise DatabaseError("Database not connected")
        cursor = await self._connection.execute(query, params or ())
        rows = await cursor.fetchall()
        if not rows:
            return []
        column_names = [description[0] for description in cursor.description]
        return [dict(zip(column_names, row)) for row in rows]
    
    def get_placeholder(self) -> str:
        return self._placeholder


class MSSQLConnection(DatabaseConnection):
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self.aioodbc = None
        self._placeholder = "?"
    
    async def connect(self):
        try:
            import aioodbc
            self.aioodbc = aioodbc
            self.pool = await aioodbc.create_pool(
                dsn=self.connection_string,
                minsize=5,
                maxsize=20
            )
        except ImportError:
            raise DatabaseError("aioodbc is not installed")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to Microsoft SQL Server: {e}")
    
    async def disconnect(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
    
    async def execute(self, query: str, params: tuple = None) -> Any:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params or ())
                if query.strip().upper().startswith('INSERT'):
                    # Try to get identity value
                    try:
                        await cur.execute("SELECT SCOPE_IDENTITY()")
                        row = await cur.fetchone()
                        return row[0] if row else None
                    except:
                        return None
                return None
    
    async def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params or ())
                row = await cur.fetchone()
                if row is None:
                    return None
                # Get column names
                columns = [column[0] for column in cur.description]
                return dict(zip(columns, row))
    
    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params or ())
                rows = await cur.fetchall()
                if not rows:
                    return []
                # Get column names
                columns = [column[0] for column in cur.description]
                return [dict(zip(columns, row)) for row in rows]
    
    def get_placeholder(self) -> str:
        return self._placeholder


class OracleConnection(DatabaseConnection):
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self.async_oracledb = None
        self._placeholder = ":{}"
        self._param_counter = 0
    
    async def connect(self):
        try:
            import async_oracledb
            self.async_oracledb = async_oracledb
            # Parse Oracle connection string
            import urllib.parse
            parsed = urllib.parse.urlparse(self.connection_string)
            
            # Oracle connection format: oracle://user:password@host:port/service_name
            self.pool = await async_oracledb.create_pool(
                user=parsed.username,
                password=parsed.password,
                dsn=f"{parsed.hostname}:{parsed.port or 1521}/{parsed.path.lstrip('/')}",
                min=5,
                max=20,
                increment=1
            )
        except ImportError:
            raise DatabaseError("async_oracledb is not installed")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to Oracle: {e}")
    
    async def disconnect(self):
        if self.pool:
            await self.pool.close()
    
    async def execute(self, query: str, params: tuple = None) -> Any:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params or ())
                if query.strip().upper().startswith('INSERT'):
                    # Oracle doesn't have simple lastrowid, return None
                    return None
                return None
    
    async def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params or ())
                row = await cur.fetchone()
                if row is None:
                    return None
                # Get column names
                columns = [column[0] for column in cur.description]
                return dict(zip(columns, row))
    
    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params or ())
                rows = await cur.fetchall()
                if not rows:
                    return []
                # Get column names
                columns = [column[0] for column in cur.description]
                return [dict(zip(columns, row)) for row in rows]
    
    def get_placeholder(self) -> str:
        self._param_counter += 1
        return self._placeholder.format(self._param_counter)


class AuroraConnection(MySQLConnection):
    """Amazon Aurora connection - inherits from MySQL with some optimizations"""
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
    
    async def connect(self):
        # Aurora is compatible with MySQL, so use MySQL connection
        # But with Aurora-specific optimizations
        try:
            import aiomysql
            self.aiomysql = aiomysql
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
                pool_recycle=300,  # Shorter recycle for Aurora
                minsize=5,
                maxsize=30,  # Larger pool for Aurora
                connect_timeout=10
            )
        except ImportError:
            raise DatabaseError("aiomysql is not installed")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to Amazon Aurora: {e}")


class Database:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection: DatabaseConnection = self._create_connection()
        self.models: Dict[str, Type['Model']] = {}
        self._connected = False
        self._table_cache = {}
        self._query_cache = {}
    
    def _create_connection(self) -> DatabaseConnection:
        if self.connection_string.startswith('mysql://'):
            return MySQLConnection(self.connection_string)
        elif self.connection_string.startswith('postgresql://') or self.connection_string.startswith('postgres://'):
            return PostgreSQLConnection(self.connection_string)
        elif self.connection_string.startswith('sqlite://'):
            return SQLiteConnection(self.connection_string)
        elif self.connection_string.startswith('mssql://') or self.connection_string.startswith('microsoft://'):
            return MSSQLConnection(self.connection_string)
        elif self.connection_string.startswith('oracle://'):
            return OracleConnection(self.connection_string)
        elif self.connection_string.startswith('aurora://'):
            return AuroraConnection(self.connection_string)
        else:
            raise DatabaseError(f"Unsupported database: {self.connection_string}")
    
    async def connect(self):
        await self.connection.connect()
        self._connected = True
        logger.info("Database connected successfully")
    
    async def disconnect(self):
        await self.connection.disconnect()
        self._connected = False
        self._table_cache.clear()
        self._query_cache.clear()
        logger.info("Database disconnected")
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager for database transactions"""
        if not self._connected:
            raise DatabaseError("Database not connected")

        transaction_active = False
        try:
            await self.connection.execute("BEGIN")
            transaction_active = True
            yield self
            if transaction_active:
                await self.connection.execute("COMMIT")
                transaction_active = False
        except Exception as e:
            if transaction_active:
                try:
                    await self.connection.execute("ROLLBACK")
                except Exception:
                    pass  # Ignore rollback errors
            logger.error(f"Transaction rolled back due to: {e}")
            raise
    
    def register_model(self, model_class: Type['Model']):
        """Register a model with the database"""
        self.models[model_class.__name__] = model_class
        model_class._database = self
    
    async def create_tables(self):
        """Create all registered tables with bulk operations"""
        if not self.models:
            return
            
        tasks = []
        for model_name, model_class in self.models.items():
            tasks.append(self.create_table(model_class))
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def drop_tables(self, cascade: bool = True, if_exists: bool = True):
        """Drop all registered tables"""
        if not self.models:
            return
            
        table_names = []
        for model_name, model_class in self.models.items():
            table_names.append(self._get_cached_table_name(model_class))
        
        for table_name in reversed(table_names):
            await self.drop_table_by_name(table_name, cascade=cascade, if_exists=if_exists)
    
    async def drop_table(self, model_class: Type['Model'], cascade: bool = True, if_exists: bool = True):
        """Drop table for a specific model"""
        table_name = self._get_cached_table_name(model_class)
        await self.drop_table_by_name(table_name, cascade=cascade, if_exists=if_exists)
    
    async def drop_table_by_name(self, table_name: str, cascade: bool = True, if_exists: bool = True):
        """Drop a table by name"""
        query_parts = ["DROP TABLE"]
        
        if if_exists:
            query_parts.append("IF EXISTS")
        
        query_parts.append(table_name)
        
        # Handle database-specific CASCADE
        if cascade and 'sqlite' not in self.connection_string and 'oracle' not in self.connection_string:
            query_parts.append("CASCADE")
        elif cascade and 'oracle' in self.connection_string:
            # Oracle uses different syntax
            query_parts[-1] = f"{table_name} CASCADE CONSTRAINTS"
        
        query = " ".join(query_parts)
        
        try:
            await self.connection.execute(query)
            logger.info(f"Table {table_name} dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop table {table_name}: {e}")
            raise DatabaseError(f"Failed to drop table {table_name}: {e}")
    
    async def create_table(self, model_class: Type['Model']):
        """Create table for a specific model with optimizations"""
        table_name = self._get_cached_table_name(model_class)
        fields = []
        
        for field_name, field in model_class._fields.items():
            field_sql = f"{field_name} {field.get_sql_type()}"
            
            if field.primary_key:
                field_sql += " PRIMARY KEY"
                if field.auto_increment:
                    if 'sqlite' in self.connection_string:
                        field_sql += " AUTOINCREMENT"
                    elif 'mysql' in self.connection_string or 'mariadb' in self.connection_string or 'aurora' in self.connection_string:
                        field_sql += " AUTO_INCREMENT"
                    elif 'mssql' in self.connection_string:
                        field_sql += " IDENTITY(1,1)"
                    elif 'oracle' in self.connection_string:
                        # Oracle uses sequences and triggers for auto increment
                        pass  # Handle separately
            
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
    
    def _get_cached_table_name(self, model_class: Type['Model']) -> str:
        """Get cached table name for better performance"""
        model_name = model_class.__name__
        if model_name not in self._table_cache:
            self._table_cache[model_name] = model_class.get_table_name()
        return self._table_cache[model_name]


class ModelMeta(type):
    """Optimized metaclass for models"""
    
    _instances = weakref.WeakKeyDictionary()
    
    def __new__(cls, name, bases, attrs):
        if name == 'Model':
            return super().__new__(cls, name, bases, attrs)
        
        fields = {}
        meta = attrs.get('_meta', {})
        
        field_items = [(k, v) for k, v in attrs.items() if isinstance(v, Field)]
        for key, value in field_items:
            fields[key] = value
            attrs.pop(key)
        
        attrs['_fields'] = fields
        attrs['_meta'] = meta
        attrs['_field_names'] = tuple(fields.keys())
        
        return super().__new__(cls, name, bases, attrs)


class Model(metaclass=ModelMeta):
    _database: Database = None
    _fields: Dict[str, Field] = {}
    _meta: Dict[str, Any] = {}
    _field_names: Tuple[str, ...] = ()
    
    __slots__ = ('_data', '_original_data', '_is_new', '__dict__')
    
    def __init__(self, **kwargs):
        object.__setattr__(self, '_data', {})
        object.__setattr__(self, '_original_data', {})
        object.__setattr__(self, '_is_new', True)
        
        for field_name, field in self._fields.items():
            if field_name in kwargs:
                self.__setattr__(field_name, kwargs[field_name])
            elif field.default is not None:
                self.__setattr__(field_name, field.default)
            else:
                self.__setattr__(field_name, None)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        elif name in self._fields:
            field = self._fields[name]
            validated_value = field.validate(value)
            self._data[name] = validated_value
        else:
            object.__setattr__(self, name, value)
    
    def __getattr__(self, name):
        if name in self._fields:
            return self._data.get(name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    @classmethod
    def get_table_name(cls) -> str:
        """Optimized table name retrieval"""
        if hasattr(cls, '_cached_table_name'):
            return cls._cached_table_name
            
        if hasattr(cls, '_meta') and cls._meta.get('table_name'):
            table_name = cls._meta.get('table_name')
        elif hasattr(cls, 'table_name'):
            table_name = cls.table_name
        else:
            table_name = cls.__name__.lower()
            
        cls._cached_table_name = table_name
        return table_name
    
    @classmethod
    def objects(cls) -> QuerySet:
        """Return a QuerySet for this model"""
        return QuerySet(cls)
    
    async def save(self):
        """Optimized save method with bulk operations support"""
        if not self._database:
            raise DatabaseError("Model not registered with database")
        
        if self._is_new:
            await self._fast_insert()
        else:
            await self._fast_update()
    
    async def _fast_insert(self):
        """Fast insert with minimal overhead"""
        table_name = self.get_table_name()
        columns = []
        values = []
        placeholders = []
        
        for field_name, field in self._fields.items():
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
            query = f"INSERT INTO {table_name} DEFAULT VALUES"
            lastrowid = await self._database.connection.execute(query)
        
        for field_name, field in self._fields.items():
            if field.primary_key and field.auto_increment and self._data.get(field_name) is None:
                self._data[field_name] = lastrowid
                break
        
        object.__setattr__(self, '_is_new', False)
        object.__setattr__(self, '_original_data', self._data.copy())
    
    async def _fast_update(self):
        """Fast update with change tracking"""
        table_name = self.get_table_name()
        set_clauses = []
        values = []
        
        for field_name, field in self._fields.items():
            if field.primary_key:
                continue
            
            if self._data.get(field_name) != self._original_data.get(field_name):
                set_clauses.append(f"{field_name} = {self._database.connection.get_placeholder()}")
                values.append(self._data.get(field_name))
        
        if not set_clauses:
            return
        
        pk_field = next((name for name, field in self._fields.items() if field.primary_key), None)
        if not pk_field:
            raise DatabaseError("No primary key defined")
        
        pk_value = self._data.get(pk_field)
        values.append(pk_value)
        
        query = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {pk_field} = {self._database.connection.get_placeholder()}"
        await self._database.connection.execute(query, tuple(values))
        object.__setattr__(self, '_original_data', self._data.copy())
    
    async def delete(self):
        """Fast delete operation"""
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
        
        object.__setattr__(self, '_is_new', True)
    
    @classmethod
    async def create(cls, **kwargs) -> 'Model':
        """Fast create and save"""
        instance = cls(**kwargs)
        await instance.save()
        return instance
    
    @classmethod
    async def bulk_create(cls, objects: List[Dict] | List['Model'], batch_size: int = 1000) -> List['Model']:
        """Bulk create multiple instances efficiently"""
        if not objects:
            return []
        
        if not cls._database:
            raise DatabaseError("Model not registered with database")
        
        all_instances = []
        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
            batch_instances = await cls._bulk_create_batch(batch)
            all_instances.extend(batch_instances)
        
        return all_instances
    
    @classmethod
    async def _bulk_create_batch(cls, objects: List[Dict] | List['Model']) -> List['Model']:
        """Create a batch of objects efficiently"""
        if not objects:
            return []
        
        if isinstance(objects[0], cls):
            data_list = [obj.to_dict() for obj in objects]
        else:
            data_list = objects
        
        return await cls._bulk_insert_batch(data_list)
    
    @classmethod
    async def _bulk_insert_batch(cls, data_list: List[Dict]) -> List['Model']:
        """Internal bulk insert batch implementation"""
        if not data_list:
            return []
        
        table_name = cls.get_table_name()
        connection = cls._database.connection
        
        fields_to_insert = []
        auto_increment_field = None
        
        for field_name, field in cls._fields.items():
            if field.primary_key and field.auto_increment:
                auto_increment_field = field_name
            else:
                fields_to_insert.append(field_name)
        
        if not fields_to_insert and not auto_increment_field:
            return []
        
        if fields_to_insert:
            placeholders = [connection.get_placeholder() for _ in fields_to_insert]
            query = f"INSERT INTO {table_name} ({', '.join(fields_to_insert)}) VALUES ({', '.join(placeholders)})"
        else:
            query = f"INSERT INTO {table_name} DEFAULT VALUES"
        
        instances = []
        
        for data in data_list:
            values = [data.get(field_name) for field_name in fields_to_insert]
            lastrowid = await connection.execute(query, tuple(values) if values else ())
            
            instance = cls(**data)
            if auto_increment_field and lastrowid:
                instance._data[auto_increment_field] = lastrowid
            instance._is_new = False
            instance._original_data = instance._data.copy()
            instances.append(instance)
        
        return instances
    
    def to_dict(self) -> Dict[str, Any]:
        """Fast dictionary conversion"""
        return self._data.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Model':
        """Fast creation from dictionary"""
        return cls(**data)
    
    def __repr__(self):
        pk_field = next((name for name, field in self._fields.items() if field.primary_key), 'id')
        pk_value = self._data.get(pk_field, 'None')
        return f"<{self.__class__.__name__}: {pk_field}={pk_value}>"


Model.DoesNotExist = type('DoesNotExist', (Exception,), {})