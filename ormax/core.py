# ormax/core.py
"""
Core module for the Ormax ORM.

This module contains the main classes and logic for interacting with databases,
defining models, managing relationships, and executing queries.
"""
import asyncio
import copy
import inspect
import logging
import re
from typing import Dict, Any, Optional, List, Type, Tuple, Union, Set, cast
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import uuid
import weakref

from .fields import DateTimeField, Field, ForeignKeyField
from .query import QuerySet
from .exceptions import (
    DatabaseError,
    ValidationError,
    DoesNotExist,
    MultipleObjectsReturned,
)
from .utils import format_placeholders

logger = logging.getLogger(__name__)


def _get_running_task():
    """Return the current asyncio task, or None outside a running loop."""
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None

# --- Relationship Management ---


class RelationshipManager:
    """
    Manages relationships between model classes.

    This class holds related and reverse related managers for a given model,
    facilitating navigation of ForeignKey relationships.
    """

    def __init__(self, model_class: Type["Model"]):
        """
        Initialize the relationship manager for a model class.

        Args:
            model_class: The model class this manager belongs to.
        """
        self.model_class = model_class
        self._related_managers: Dict[str, "RelatedManager"] = {}
        self._reverse_managers: Dict[str, "ReverseRelatedManager"] = {}

    def add_related_manager(self, name: str, manager: "RelatedManager"):
        """
        Add a forward relationship manager.

        Args:
            name: The name of the relationship field on the model.
            manager: The RelatedManager instance.
        """
        self._related_managers[name] = manager

    def add_reverse_manager(self, name: str, manager: "ReverseRelatedManager"):
        """
        Add a reverse relationship manager.

        Args:
            name: The name of the reverse relationship (usually related_name).
            manager: The ReverseRelatedManager instance.
        """
        self._reverse_managers[name] = manager

    def get_related_manager(self, name: str) -> Optional["RelatedManager"]:
        """
        Get a forward relationship manager by name.

        Args:
            name: The name of the relationship field.

        Returns:
            The RelatedManager instance or None if not found.
        """
        return self._related_managers.get(name)

    def get_reverse_manager(self, name: str) -> Optional["ReverseRelatedManager"]:
        """
        Get a reverse relationship manager by name.

        Args:
            name: The name of the reverse relationship.

        Returns:
            The ReverseRelatedManager instance or None if not found.
        """
        return self._reverse_managers.get(name)

    def get_all_related_managers(self) -> Dict[str, "RelatedManager"]:
        """
        Get all forward relationship managers.

        Returns:
            A dictionary of all related managers.
        """
        return self._related_managers.copy()

    def get_all_reverse_managers(self) -> Dict[str, "ReverseRelatedManager"]:
        """
        Get all reverse relationship managers.

        Returns:
            A dictionary of all reverse related managers.
        """
        return self._reverse_managers.copy()


class BaseRelatedManager:
    """
    Base class for related managers.

    Provides common functionality for both forward and reverse relationship managers.
    """

    def __init__(self, model_class: Type["Model"], related_model_class: Type["Model"]):
        """
        Initialize the base related manager.

        Args:
            model_class: The model class that owns this relationship.
            related_model_class: The model class on the other side of the relationship.
        """
        self.model_class = model_class
        self.related_model_class = related_model_class


class RelatedManager(BaseRelatedManager):
    """
    Manager for forward relationships (ForeignKey).

    Handles accessing and setting related objects pointed to by a ForeignKey field.
    """

    def __init__(
        self,
        model_class: Type["Model"],
        related_model_class: Type["Model"],
        field_name: str,
        foreign_key_field: "ForeignKeyField",
    ):
        """
        Initialize the forward relationship manager.

        Args:
            model_class: The model class that owns the ForeignKey field.
            related_model_class: The model class the ForeignKey points to.
            field_name: The name of the ForeignKey field on the model.
            foreign_key_field: The ForeignKeyField instance.
        """
        super().__init__(model_class, related_model_class)
        self.field_name = field_name
        self.foreign_key_field = foreign_key_field
        self.related_field_name = (
            foreign_key_field.related_name or f"{model_class.__name__.lower()}_set"
        )

    async def get(self, instance: "Model") -> Optional["Model"]:
        """
        Get the related object for a model instance.

        Args:
            instance: The model instance to get the related object for.

        Returns:
            The related model instance or None if the FK is NULL.

        Raises:
            DatabaseError: If the model is not registered with a database.
        """
        if not instance._database:
            raise DatabaseError("Model not registered with database")

        fk_name = self.foreign_key_field.name or self.field_name
        fk_value = instance._data.get(fk_name)
        if fk_value is None:
            return None

        return (
            await self.related_model_class.objects()
            .filter(**{self.related_model_class._meta.get("pk_field", "id"): fk_value})
            .first()
        )

    async def set(self, instance: "Model", related_instance: Optional["Model"]):
        """
        Set the related object for a model instance.

        Args:
            instance: The model instance to set the relationship on.
            related_instance: The related model instance to set, or None to clear.

        Raises:
            ValueError: If the related_instance is not of the correct type.
            DatabaseError: If the model is not registered with a database.
        """
        fk_name = self.foreign_key_field.name or self.field_name
        if related_instance is None:
            setattr(instance, fk_name, None)
        else:
            if not isinstance(related_instance, self.related_model_class):
                raise ValueError(
                    f"Related instance must be of type {self.related_model_class.__name__}"
                )

            # Ensure related instance is saved
            if (
                getattr(related_instance, related_instance._meta.get("pk_field", "id"))
                is None
            ):
                await related_instance.save()

            pk_value = getattr(
                related_instance, related_instance._meta.get("pk_field", "id")
            )
            setattr(instance, fk_name, pk_value)

        # Save the instance if it's not new
        if not instance._is_new:
            await instance.save()


class ReverseRelatedManager(BaseRelatedManager):
    """
    Manager for reverse relationships (backward ForeignKey).

    Handles accessing objects that point to a specific instance via a ForeignKey.
    """

    def __init__(
        self,
        model_class: Type["Model"],
        related_model_class: Type["Model"],
        field_name: str,
        foreign_key_field: "ForeignKeyField",
    ):
        """
        Initialize the reverse relationship manager.

        Args:
            model_class: The model class that is pointed to by the ForeignKey.
            related_model_class: The model class that owns the ForeignKey field.
            field_name: The name of the ForeignKey field on the related model.
            foreign_key_field: The ForeignKeyField instance.
        """
        super().__init__(model_class, related_model_class)
        self.field_name = field_name
        self.foreign_key_field = foreign_key_field

    def get_queryset(self, instance: "Model") -> QuerySet:
        """
        Get a QuerySet for related objects pointing to this instance.

        Args:
            instance: The model instance to find related objects for.

        Returns:
            A QuerySet of related objects.

        Raises:
            DatabaseError: If the model is not registered with a database.
        """
        if not instance._database:
            raise DatabaseError("Model not registered with database")

        pk_value = getattr(instance, instance._meta.get("pk_field", "id"))
        return self.related_model_class.objects().filter(
            **{self.field_name: pk_value}
        )

    async def all(self, instance: "Model") -> List["Model"]:
        """
        Get all related objects pointing to this instance.

        Args:
            instance: The model instance to find related objects for.

        Returns:
            A list of related model instances.
        """
        qs = self.get_queryset(instance)
        return await qs.all()

    async def create(self, instance: "Model", **kwargs) -> "Model":
        """
        Create a new related object pointing to this instance.

        Args:
            instance: The model instance to create a related object for.
            **kwargs: Keyword arguments for the new related object.

        Returns:
            The newly created related model instance.
        """
        fk_name = self.foreign_key_field.name or self.field_name
        pk_value = getattr(instance, instance._meta.get("pk_field", "id"))
        kwargs[fk_name] = pk_value
        return await self.related_model_class.create(**kwargs)

    async def add(self, instance: "Model", *related_instances: "Model"):
        """
        Add existing objects to point to this instance.

        Args:
            instance: The model instance to add relationships to.
            *related_instances: The related model instances to add.

        Raises:
            ValueError: If any related_instance is not of the correct type.
        """
        fk_name = self.foreign_key_field.name or self.field_name
        pk_value = getattr(instance, instance._meta.get("pk_field", "id"))
        for related_instance in related_instances:
            if not isinstance(related_instance, self.related_model_class):
                raise ValueError(
                    f"Related instance must be of type {self.related_model_class.__name__}"
                )
            setattr(related_instance, fk_name, pk_value)
            await related_instance.save()

    async def remove(self, instance: "Model", *related_instances: "Model"):
        """
        Remove the relationship from related objects (set FK to NULL).

        Args:
            instance: The model instance to remove relationships from.
            *related_instances: The related model instances to remove.

        Raises:
            ValueError: If any related_instance is not of the correct type.
        """
        fk_name = self.foreign_key_field.name or self.field_name
        for related_instance in related_instances:
            if not isinstance(related_instance, self.related_model_class):
                raise ValueError(
                    f"Related instance must be of type {self.related_model_class.__name__}"
                )

            fk_value = getattr(related_instance, fk_name)
            pk_value = getattr(instance, instance._meta.get("pk_field", "id"))

            if fk_value == pk_value:
                setattr(related_instance, fk_name, None)
                await related_instance.save()

    async def clear(self, instance: "Model"):
        """
        Clear all relationships to this instance (set FK to NULL on all related objects).

        Args:
            instance: The model instance to clear relationships for.
        """
        fk_name = self.foreign_key_field.name or self.field_name
        qs = self.get_queryset(instance)
        await qs.update(**{fk_name: None})


# --- Model Definition and Metaclass ---


class ModelMeta(type):
    """Build model metadata, including inherited fields and ``Meta`` options."""

    _instances = weakref.WeakKeyDictionary()
    _relationships: RelationshipManager
    _fields: Dict[str, Field]
    _meta: Dict[str, Any]
    _field_names: Tuple[str, ...]

    def __new__(cls, name, bases, attrs):
        if name == "Model":
            return super().__new__(cls, name, bases, attrs)

        fields: Dict[str, Field] = {}
        meta: Dict[str, Any] = {}

        for base in bases:
            for field_name, field in getattr(base, "_fields", {}).items():
                fields[field_name] = copy.copy(field)
            meta.update(getattr(base, "_meta", {}))

        own_meta = dict(attrs.get("_meta", {}))
        meta.update(own_meta)
        meta_class = attrs.get("Meta")
        meta_class_values = {}
        if meta_class is not None:
            meta_class_values = {
                key: value
                for key, value in vars(meta_class).items()
                if not key.startswith("_")
            }
            meta.update(meta_class_values)

        # Abstract status applies only to the class that declares it.
        if "abstract" not in own_meta and "abstract" not in meta_class_values:
            meta["abstract"] = False

        for key, value in list(attrs.items()):
            if isinstance(value, Field):
                value.name = key
                fields[key] = value
                attrs.pop(key)

        has_pk = any(field.primary_key for field in fields.values())
        if not has_pk and "id" not in fields:
            from .fields import AutoField

            fields["id"] = AutoField(
                primary_key=True, auto_increment=True, name="id"
            )

        attrs["_fields"] = fields
        attrs["_meta"] = meta
        attrs["_field_names"] = tuple(fields.keys())
        attrs.pop("_cached_table_name", None)

        new_class = cast(Type["Model"], super().__new__(cls, name, bases, attrs))
        for field_name, field in new_class._fields.items():
            field.name = field_name
        new_class._relationships = RelationshipManager(new_class)
        return new_class


class Model(metaclass=ModelMeta):
    """
    Base class for all Ormax models.

    Provides the core functionality for database interaction, including
    saving, deleting, and querying model instances.
    """

    _database: Optional["Database"] = None
    _fields: Dict[str, Field] = {}
    _meta: Dict[str, Any] = {}
    _field_names: Tuple[str, ...] = ()
    _relationships: Optional["RelationshipManager"] = None
    __slots__ = ("_data", "_original_data", "_is_new", "__dict__")

    @property
    def pk(self):
        """Get the primary key value for this model instance."""
        for field_name, field in self._fields.items():
            if field.primary_key:
                return self._data.get(field_name)
        return self._data.get("id")

    @pk.setter
    def pk(self, value):
        """Set the primary key value for this model instance."""
        for field_name, field in self._fields.items():
            if field.primary_key:
                self._data[field_name] = value
                return
        self._data["id"] = value

    def __init_subclass__(cls, **kwargs):
        """
        Initialize subclass. Sets the name attribute on fields.
        """
        super().__init_subclass__(**kwargs)

        # Set the name attribute on fields
        if hasattr(cls, "_fields"):
            for attr_name, attr_value in cls._fields.items():
                attr_value.name = attr_name

    def __init__(self, **kwargs):
        """
        Initialize a new model instance.

        Args:
            **kwargs: Initial field values.
        """
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "_original_data", {})
        object.__setattr__(self, "_is_new", True)

        unknown_fields = set(kwargs) - set(self._fields)
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise TypeError(f"Unknown field(s) for {self.__class__.__name__}: {names}")

        for field_name, field in self._fields.items():
            if field_name in kwargs:
                self.__setattr__(field_name, kwargs[field_name])
            elif field.default is not None:
                self.__setattr__(field_name, field.get_default())
            else:
                self.__setattr__(field_name, None)

    def __setattr__(self, name, value):
        # Check if it's a relationship manager
        if hasattr(self, "_relationships") and self._relationships:
            manager = self._relationships.get_related_manager(name)
            if manager:
                # This is a related object assignment
                if hasattr(self, "_database") and self._database:
                    # Check if value is a model instance or a simple value (like an ID)
                    if value is not None and hasattr(value, "_meta"):
                        # It's a model instance, get its primary key
                        fk_value = getattr(value, value._meta.get("pk_field", "id"))
                    else:
                        # It's a simple value (e.g., an ID), use it directly
                        # Ensure it's validated by the foreign key field
                        fk_field = self._fields.get(manager.field_name)
                        if fk_field:
                            fk_value = fk_field.validate(value)
                        else:
                            fk_value = value
                    # Set the actual foreign key field's value in _data
                    self._data[manager.field_name] = fk_value
                    return
                # If not registered with database, fall through to default behavior
        
        # بقیه کد __setattr__ اصلی بدون تغییر
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        elif name in self._fields:
            field = self._fields[name]
            # Check for ForeignKeyField assignment
            if isinstance(field, ForeignKeyField) and value is not None:
                if hasattr(value, "_meta"):
                    try:
                        pk_field_name = value._meta.get("pk_field", "id")
                        pk_value = getattr(value, pk_field_name)
                    except AttributeError:
                        raise ValidationError(
                            f"Provided object '{value}' does not have a primary key field"
                        )
                else:
                    pk_value = value
                validated_value = field.validate(pk_value)
                self._data[name] = validated_value
                return
            # Default field validation and setting
            validated_value = field.validate(value)
            self._data[name] = validated_value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        """
        Get an attribute from the model instance.
        """
        # Check if attribute was prefetched
        prefetch_key = f"_prefetched_{name}"
        if prefetch_key in self._data:
            return self._data[prefetch_key]

        # Check for related managers
        if hasattr(self, "_relationships") and self._relationships:
            # Check for forward relationship
            related_manager = self._relationships.get_related_manager(name)
            if related_manager:
                raise AttributeError(
                    f"Accessing related field '{name}' requires async method call. Use `await get_{name}()` or similar."
                )

            # Check for reverse relationship
            reverse_manager = self._relationships.get_reverse_manager(name)
            if reverse_manager:
                return reverse_manager.get_queryset(self)

        if name in self._fields:
            return self._data.get(name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )


    async def get_related(self, name: str) -> Optional["Model"]:
        """Fetch a forward foreign-key relation by field name."""
        prefetched_key = f"_prefetched_{name}"
        if prefetched_key in self._data:
            return self._data[prefetched_key]
        manager = self._relationships.get_related_manager(name) if self._relationships else None
        if manager is None:
            raise AttributeError(f"'{self.__class__.__name__}' has no relation '{name}'")
        return await manager.get(self)

    async def set_related(self, name: str, value: Optional["Model"]) -> None:
        """Assign and persist a forward foreign-key relation by field name."""
        manager = self._relationships.get_related_manager(name) if self._relationships else None
        if manager is None:
            raise AttributeError(f"'{self.__class__.__name__}' has no relation '{name}'")
        await manager.set(self, value)

    @classmethod
    def get_table_name(cls) -> str:
        """
        Get the database table name for this model.

        Returns:
            The table name.
        """
        cached = cls.__dict__.get("_cached_table_name")
        if cached is not None:
            return cached
        if hasattr(cls, "_meta") and cls._meta.get("table_name"):
            table_name = cls._meta.get("table_name")
        elif hasattr(cls, "table_name"):
            table_name = cls.table_name
        else:
            table_name = cls.__name__.lower()
        cls._cached_table_name = table_name
        return table_name

    @classmethod
    def objects(cls) -> QuerySet:
        """
        Get a QuerySet for this model.

        Returns:
            A QuerySet instance for this model.
        """
        return QuerySet(cls)

    @classmethod
    async def create_table(cls):
        """
        Create table for this model in its registered database.
        """
        if cls._database:
            await cls._database.create_table(cls)
        else:
            raise DatabaseError(f"Model '{cls.__name__}' is not registered with any database.")

    def clean(self):
        """Hook for custom model validation."""
        pass

    def full_clean(self):
        """Run field-level and model-level validation."""
        for field_name, field in self._fields.items():
            if field_name in self._data:
                self._data[field_name] = field.validate(self._data[field_name])
        self.clean()

    async def save(self):
        """Save the model instance to the database."""
        if hasattr(self, "pre_save"):
            res = self.pre_save()
            if inspect.isawaitable(res):
                await res

        # Process pre_save for all fields
        for field_name, field in self._fields.items():
            if hasattr(field, 'pre_save'):
                value = field.pre_save(self, self._is_new)
                if value is not None:
                    self._data[field_name] = value

        self.full_clean()
        is_created = self._is_new
        if is_created:
            await self._fast_insert()
        else:
            await self._fast_update()
    
        # Update instance state
        object.__setattr__(self, "_is_new", False)
        object.__setattr__(self, "_original_data", self._data.copy())

        if hasattr(self, "post_save"):
            res = self.post_save(created=is_created)
            if inspect.isawaitable(res):
                await res

    async def _fast_insert(self):
        """Fast insert with correct placeholder handling."""
        if not self._database:
            raise DatabaseError("Model not registered with database")
        table_name = self.get_table_name()
        connection = self._database.connection
        
        # Get fields to insert
        columns = []
        values = []
        for field_name, field in self._fields.items():
            if field.primary_key and field.auto_increment:
                continue
                
            # Skip if value is None and field has default
            if self._data.get(field_name) is None:
                if field.default is not None:
                    continue
                # Handle auto_now_add for DateTimeField
                if isinstance(field, DateTimeField) and field.auto_now_add:
                    continue
                # Handle auto_now for DateTimeField (shouldn't happen in insert, but just in case)
                if isinstance(field, DateTimeField) and field.auto_now:
                    continue
                    
            columns.append(field_name)
            values.append(field.to_database(self._data.get(field_name)))
        
        # Generate query with placeholders
        if columns:
            placeholders = ["?" for _ in columns]
            query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        else:
            query = f"INSERT INTO {table_name} DEFAULT VALUES"
        
        if hasattr(self._database, "connection_string"):
            query = format_placeholders(query, self._database.connection_string)
            
        lastrowid = await connection.execute(query, tuple(values))
        
        # Set primary key if needed
        pk_field = next((f for f in self._fields if self._fields[f].primary_key), "id")
        if self._fields[pk_field].auto_increment and lastrowid:
            self._data[pk_field] = lastrowid
        
        # Update instance state
        object.__setattr__(self, "_is_new", False)
        object.__setattr__(self, "_original_data", self._data.copy())

    async def _fast_update(self):
        """Fast update with change tracking and correct placeholder handling."""
        if not self._database:
            raise DatabaseError("Model not registered with database")
        table_name = self.get_table_name()
        set_clauses = []
        values = []
        connection = self._database.connection
        
        # Get primary key field
        pk_field = next((f for f in self._fields if self._fields[f].primary_key), "id")
        pk_value = self._data.get(pk_field)
        
        if pk_value is None:
            raise DatabaseError("Cannot update model without primary key value")
        
        # Build set clauses for fields that have changed
        for field_name, field in self._fields.items():
            if field.primary_key:
                continue
                
            # Skip auto-increment fields
            if field.auto_increment:
                continue
                
            # Skip if value hasn't changed
            if self._data.get(field_name) == self._original_data.get(field_name):
                continue
                
            values.append(field.to_database(self._data.get(field_name)))
            set_clauses.append(f"{field_name} = ?")
        
        # If no fields to update, return early
        if not values:
            return
        
        # Build WHERE clause
        where_clause = f"{pk_field} = ?"
        values.append(pk_value)
        
        # Build final query
        query = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {where_clause}"
        
        if hasattr(self._database, "connection_string"):
            query = format_placeholders(query, self._database.connection_string)
            
        await connection.execute(query, tuple(values))
        
        # Update instance state
        object.__setattr__(self, "_is_new", False)
        object.__setattr__(self, "_original_data", self._data.copy())

    async def delete(self):
        """
        Delete the model instance from the database.

        Raises:
            DatabaseError: If the model is not registered with a database or has no primary key.
        """
        if hasattr(self, "pre_delete"):
            res = self.pre_delete()
            if inspect.isawaitable(res):
                await res

        if not self._database:
            raise DatabaseError("Model not registered with database")
        pk_field = next(
            (name for name, field in self._fields.items() if field.primary_key), None
        )
        if not pk_field:
            raise DatabaseError("No primary key defined")
        pk_value = self._data.get(pk_field)
        if pk_value is None:
            raise DatabaseError("Cannot delete unsaved instance")

        table_name = self.get_table_name()
        connection = self._database.connection
        
        query = f"DELETE FROM {table_name} WHERE {pk_field} = ?"
        if hasattr(self._database, "connection_string"):
            query = format_placeholders(query, self._database.connection_string)
            
        await connection.execute(query, (pk_value,))
        object.__setattr__(self, "_is_new", True)

        if hasattr(self, "post_delete"):
            res = self.post_delete()
            if inspect.isawaitable(res):
                await res

    @classmethod
    async def create(cls, **kwargs) -> "Model":
        """
        Create and save a new model instance.

        Args:
            **kwargs: Initial field values for the new instance.

        Returns:
            The newly created and saved model instance.
        """
        instance = cls(**kwargs)
        await instance.save()
        return instance

    @classmethod
    async def get_or_create(
        cls, defaults: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Tuple["Model", bool]:
        """
        Get an instance matching kwargs, or create a new one with kwargs and defaults.
        Returns (instance, created_boolean).
        """
        return await cls.objects().get_or_create(defaults=defaults, **kwargs)

    @classmethod
    async def update_or_create(
        cls, defaults: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Tuple["Model", bool]:
        """
        Update an instance matching kwargs with defaults, or create a new one.
        Returns (instance, created_boolean).
        """
        return await cls.objects().update_or_create(defaults=defaults, **kwargs)


    @classmethod
    async def bulk_create(
        cls, objects: Union[List[Dict], List["Model"]], batch_size: int = 1000
    ) -> List["Model"]:
        """
        Bulk create multiple instances efficiently.

        Args:
            objects: A list of dictionaries or model instances to create.
            batch_size: The number of objects to create in each batch.

        Returns:
            A list of the created model instances.
        """
        if not objects:
            return []
        if not cls._database:
            raise DatabaseError("Model not registered with database")

        all_instances = []
        for i in range(0, len(objects), batch_size):
            batch = objects[i : i + batch_size]
            batch_instances = await cls._bulk_create_batch(batch)
            all_instances.extend(batch_instances)
        return all_instances

    @classmethod
    async def _bulk_create_batch(
        cls, objects: Union[List[Dict], List["Model"]]
    ) -> List["Model"]:
        """Create a batch of objects efficiently."""
        if not objects:
            return []
        data_list: List[Dict[str, Any]] = []
        for obj in objects:
            if isinstance(obj, Model):
                data_list.append(obj.to_dict())
            elif isinstance(obj, dict):
                data_list.append(obj)
            else:
                data_list.append(dict(obj))
        return await cls._bulk_insert_batch(data_list)

    @classmethod
    async def _bulk_insert_batch(cls, data_list: List[Dict]) -> List["Model"]:
        """
        Internal bulk insert batch implementation with correct placeholder handling.
        Properly handles auto-increment fields, auto_now/auto_now_add fields, and defaults.
        """
        if not data_list:
            return []
        
        # Process default values and auto fields
        current_time = datetime.now(timezone.utc)
        
        for data in data_list:
            for field_name, field in cls._fields.items():
                # Skip if already provided
                if field_name in data:
                    continue
                    
                # Handle default values
                if field.default is not None:
                    data[field_name] = field.get_default()
                    
                # Handle auto_now_add and auto_now for DateTimeField
                if isinstance(field, DateTimeField):
                    if field.auto_now_add or field.auto_now:
                        data[field_name] = current_time
        
        table_name = cls.get_table_name()
        if not cls._database:
            raise DatabaseError("Model not registered with database")
        connection = cls._database.connection
        # Determine fields to insert based on the model definition
        # We assume all instances in the batch have the same structure
        # Find fields that are not auto-increment PKs or have explicit values
        sample_data = data_list[0]
        fields_to_insert = []
        auto_increment_field = None
        for field_name, field in cls._fields.items():
            if field.primary_key and field.auto_increment:
                auto_increment_field = field_name
                # Always exclude auto-increment PK from bulk insert fields
            else:
                fields_to_insert.append(field_name)
        if not fields_to_insert and not auto_increment_field:
            # This is a very unusual case, but handle it
            return []
        # Generate query with correct placeholders
        if fields_to_insert:
            if hasattr(connection, "get_placeholder") and callable(
                connection.get_placeholder
            ):
                if connection.get_placeholder() == "$":
                    # PostgreSQL: $1, $2, ...
                    placeholders = [f"${i+1}" for i in range(len(fields_to_insert))]
                else:
                    # MySQL, SQLite, etc.: ?, %s, ...
                    placeholders = [
                        connection.get_placeholder() for _ in fields_to_insert
                    ]
            else:
                # Default fallback
                placeholders = ["?" for _ in fields_to_insert]
            query = f"INSERT INTO {table_name} ({', '.join(fields_to_insert)}) VALUES ({', '.join(placeholders)})"
        else:
            query = f"INSERT INTO {table_name} DEFAULT VALUES"
        instances: List["Model"] = []
        for data in data_list:
            # Prepare values for this row
            values = [
                cls._fields[field_name].to_database(data.get(field_name))
                for field_name in fields_to_insert
            ]
            # Execute the insert
            lastrowid = await connection.execute(query, tuple(values) if values else ())
            # Create instance and set its data
            instance = cls(**data)  # This will set _data based on **kwargs
            # Handle auto-increment field if needed
            if auto_increment_field and lastrowid:
                instance._data[auto_increment_field] = lastrowid
            instance._is_new = False
            instance._original_data = instance._data.copy()
            instances.append(instance)
        return instances

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model instance to a dictionary.

        Returns:
            A dictionary representation of the model instance.
        """
        return self._data.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Model":
        """
        Create a model instance from a dictionary.

        Args:
            data: A dictionary of field values.

        Returns:
            A new model instance.
        """
        return cls(**data)

    def __repr__(self):
        """String representation of the model instance."""
        pk_field = next(
            (name for name, field in self._fields.items() if field.primary_key), "id"
        )
        pk_value = self._data.get(pk_field, "None")
        return f"<{self.__class__.__name__}: {pk_field}={pk_value}>"


# Attach the public exception type to the base model.
Model.DoesNotExist = DoesNotExist


# --- Database Connection and Management ---


class Database:
    """
    Main database interface for the Ormax ORM.

    Handles connection management, model registration, table creation,
    and relationship setup.
    """

    def __init__(self, connection_string: str, models: Optional[Union[List, Tuple, Set, Any]] = None):
        """
        Initialize the database connection.

        Args:
            connection_string: The database connection string.
            models: Optional model(s), module(s), or directory path(s) to register automatically.
        """
        self.connection_string = connection_string
        self.connection: DatabaseConnection = self._create_connection()
        self.connection.database = self
        self.models: Dict[str, Type["Model"]] = {}
        self._connected = False
        self._current_transaction_var: ContextVar[Any] = ContextVar(
            f"ormax_current_transaction_{id(self)}", default=None
        )
        self._transaction_level_var: ContextVar[Any] = ContextVar(
            f"ormax_transaction_level_{id(self)}", default=0
        )
        self._sqlite_transaction_lock = asyncio.Lock()
        self._sqlite_transaction_owner = None
        self._table_cache = {}
        self._query_cache = {}
        if models is not None:
            if isinstance(models, (list, tuple, set)):
                self.register_model(*models)
            else:
                self.register_model(models)

    @property
    def _current_transaction(self):
        state = self._current_transaction_var.get()
        if state is None:
            return None
        owner, value = state
        return value if owner is _get_running_task() else None

    @_current_transaction.setter
    def _current_transaction(self, value):
        if value is None:
            self._current_transaction_var.set(None)
        else:
            self._current_transaction_var.set((_get_running_task(), value))

    @property
    def _transaction_level(self) -> int:
        state = self._transaction_level_var.get()
        if isinstance(state, tuple):
            owner, value = state
            return value if owner is _get_running_task() else 0
        return int(state or 0)

    @_transaction_level.setter
    def _transaction_level(self, value: int):
        if value <= 0:
            self._transaction_level_var.set(0)
        else:
            self._transaction_level_var.set((_get_running_task(), value))

    def _create_connection(self) -> "DatabaseConnection":
        """
        Create the appropriate database connection based on the connection string.

        Returns:
            A DatabaseConnection instance.

        Raises:
            DatabaseError: If the database type is not supported.
        """
        if self.connection_string.startswith(("mysql://", "mariadb://")):
            return MySQLConnection(self.connection_string)
        elif self.connection_string.startswith(
            "postgresql://"
        ) or self.connection_string.startswith("postgres://"):
            return PostgreSQLConnection(self.connection_string)
        elif self.connection_string.startswith("sqlite://"):
            return SQLiteConnection(self.connection_string)
        elif self.connection_string.startswith(
            "mssql://"
        ) or self.connection_string.startswith("microsoft://"):
            return MSSQLConnection(self.connection_string)
        elif self.connection_string.startswith("oracle://"):
            return OracleConnection(self.connection_string)
        elif self.connection_string.startswith("aurora://"):
            return AuroraConnection(self.connection_string)
        else:
            raise DatabaseError(f"Unsupported database: {self.connection_string}")

    @property
    def is_connected(self) -> bool:
        """Return True if database is connected."""
        return self._connected

    async def __aenter__(self) -> "Database":
        await self.connect()
        return self
    
    async def __aexit__(self, *args, **kwargs):
        await self.disconnect()

    async def connect(self):
        """
        Connect to the database.

        Raises:
            DatabaseError: If connection fails.
        """
        if self._connected:
            return
        await self.connection.connect()
        self._connected = True
        logger.info("Database connected successfully")

    async def disconnect(self):
        """
        Disconnect from the database.
        """
        if not self._connected:
            return
        await self.connection.disconnect()
        self._connected = False
        self._table_cache.clear()
        self._query_cache.clear()
        logger.info("Database disconnected")

    async def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute a raw SQL query using the active connection."""
        return await self.connection.execute(query, params)

    async def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict]:
        """Fetch a single row using the active connection."""
        return await self.connection.fetch_one(query, params)

    async def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Fetch all rows using the active connection."""
        return await self.connection.fetch_all(query, params)

    @asynccontextmanager
    async def transaction(self):
        """Run operations atomically, using savepoints for nested contexts."""
        if not self._connected:
            raise DatabaseError("Database not connected")

        inherited_state = self._current_transaction_var.get()
        if (
            isinstance(inherited_state, tuple)
            and inherited_state[0] is not _get_running_task()
        ):
            raise DatabaseError(
                "Transaction contexts cannot be shared across asyncio tasks"
            )

        scheme = self.connection_string.split("://", 1)[0].lower()
        if scheme in {"postgresql", "postgres"}:
            db_type = "postgresql"
        elif scheme in {"mysql", "mariadb", "aurora"}:
            db_type = "mysql"
        elif scheme == "sqlite":
            db_type = "sqlite"
        elif scheme in {"mssql", "microsoft"}:
            db_type = "mssql"
        elif scheme == "oracle":
            db_type = "oracle"
        else:
            db_type = "other"

        savepoint_name = f"sp_{uuid.uuid4().hex[:12]}"
        is_nested = self._current_transaction is not None
        previous_level = self._transaction_level
        outer_lock_acquired = False
        transaction_started = False
        savepoint_created = False

        if not is_nested and db_type == "sqlite":
            # SQLite uses one connection, so outer transactions must not overlap.
            await self._sqlite_transaction_lock.acquire()
            outer_lock_acquired = True
            self._sqlite_transaction_owner = _get_running_task()

        try:
            if is_nested:
                if db_type in {"postgresql", "mysql", "sqlite", "oracle"}:
                    await self.connection.execute(f"SAVEPOINT {savepoint_name}")
                elif db_type == "mssql":
                    await self.connection.execute(f"SAVE TRANSACTION {savepoint_name}")
                else:
                    raise DatabaseError(f"Savepoints not supported for {db_type}")
                savepoint_created = True
                logger.debug(
                    "Created savepoint %s at transaction level %s",
                    savepoint_name,
                    previous_level + 1,
                )
            else:
                await self.connection.start_transaction()
                transaction_started = True
                logger.info("Started new transaction")

            self._transaction_level = previous_level + 1
            self._current_transaction = self.connection

            yield self

            if is_nested and savepoint_created:
                # SQL Server and Oracle do not require/support explicit release.
                if db_type in {"postgresql", "mysql", "sqlite"}:
                    await self.connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    logger.debug("Released savepoint %s", savepoint_name)
            elif transaction_started:
                await self.connection.commit_transaction()
                logger.info("Committed transaction")

        except BaseException as error:
            try:
                if is_nested and savepoint_created:
                    if db_type in {"postgresql", "mysql", "sqlite", "oracle"}:
                        await self.connection.execute(f"ROLLBACK TO {savepoint_name}")
                    elif db_type == "mssql":
                        await self.connection.execute(
                            f"ROLLBACK TRANSACTION {savepoint_name}"
                        )
                    logger.warning(
                        "Rolled back to savepoint %s due to %r",
                        savepoint_name,
                        error,
                    )
                elif transaction_started:
                    await self.connection.rollback_transaction()
                    logger.error("Rolled back transaction due to %r", error)
            except Exception as rollback_error:
                # Preserve the original application error while exposing rollback failure.
                logger.exception("Transaction rollback also failed: %r", rollback_error)
            raise
        finally:
            self._transaction_level = previous_level
            if previous_level == 0:
                self._current_transaction = None
                logger.debug("Transaction context cleared")
            if outer_lock_acquired:
                self._sqlite_transaction_owner = None
                self._sqlite_transaction_lock.release()

    def register_model(self, *models):
        """
        Register one or multiple model classes, modules, package names, or directory paths with this database.

        Examples:
            db.register_model(Book)
            db.register_model(Book, Store, Department)
            db.register_model([Book, Store])
            db.register_model("models")
            db.register_model(my_models_module)
        """
        import inspect
        import importlib
        import os
        import sys
        import types

        for item in models:
            if item is None:
                continue

            # If it's a list, tuple, or set, recurse
            if isinstance(item, (list, tuple, set)):
                self.register_model(*item)
                continue

            # If item is a Model class
            if inspect.isclass(item) and issubclass(item, Model) and item is not Model:
                if item._meta.get("abstract", False):
                    continue
                self.models[item.__name__] = item
                item._database = self
                continue

            # If item is a module object
            if isinstance(item, types.ModuleType):
                for name, obj in inspect.getmembers(item, inspect.isclass):
                    if (
                        issubclass(obj, Model)
                        and obj is not Model
                        and not obj._meta.get("abstract", False)
                    ):
                        self.models[obj.__name__] = obj
                        obj._database = self
                continue

            # If item is a string (module name or directory name or path)
            if isinstance(item, str):
                # Try importing as a Python module/package first
                try:
                    mod = importlib.import_module(item)
                    self.register_model(mod)
                    continue
                except ImportError:
                    pass

                # Try as a directory path or python file
                abspath = os.path.abspath(item)
                if os.path.isdir(abspath):
                    if os.path.dirname(abspath) not in sys.path:
                        sys.path.insert(0, os.path.dirname(abspath))
                    dir_name = os.path.basename(abspath)
                    try:
                        mod = importlib.import_module(dir_name)
                        self.register_model(mod)
                    except Exception:
                        for filename in os.listdir(abspath):
                            if filename.endswith(".py") and not filename.startswith("_"):
                                mod_name = f"{dir_name}.{filename[:-3]}"
                                try:
                                    mod = importlib.import_module(mod_name)
                                    self.register_model(mod)
                                except Exception:
                                    pass
                elif os.path.isfile(abspath) and abspath.endswith(".py"):
                    dir_path = os.path.dirname(abspath)
                    if dir_path not in sys.path:
                        sys.path.insert(0, dir_path)
                    mod_name = os.path.basename(abspath)[:-3]
                    try:
                        mod = importlib.import_module(mod_name)
                        self.register_model(mod)
                    except Exception:
                        pass

        self.setup_relationships()

    def setup_relationships(self):
        """
        Set up relationships between registered models.

        This method creates RelatedManager and ReverseRelatedManager instances
        for all ForeignKey fields.
        """
        for model_name, model_class in self.models.items():
            # Process forward relationships (ForeignKey fields)
            for field_name, field in model_class._fields.items():
                if isinstance(field, ForeignKeyField):
                    # Get the related model class
                    related_model_class = None
                    if isinstance(field.to, str):
                        # String reference to model
                        if field.to == "self":
                            related_model_class = model_class
                        else:
                            related_model_class = self.models.get(field.to)
                        if not related_model_class:
                            logger.warning(
                                f"Related model '{field.to}' not found for field '{field_name}' in '{model_name}'"
                            )
                            continue
                    else:
                        # Direct class reference
                        related_model_class = field.to

                    if related_model_class:
                        # Create related manager
                        related_manager = RelatedManager(
                            model_class, related_model_class, field_name, field
                        )
                        if model_class._relationships is not None:
                            model_class._relationships.add_related_manager(
                                field_name, related_manager
                            )

                        # Create reverse manager
                        reverse_manager = ReverseRelatedManager(
                            related_model_class, model_class, field_name, field
                        )
                        reverse_name = field.related_name or f"{model_name.lower()}_set"
                        if related_model_class._relationships is not None:
                            related_model_class._relationships.add_reverse_manager(
                                reverse_name, reverse_manager
                            )

    async def create_tables(self):
        """
        Create tables for all registered models in the correct order.
        Handles foreign keys correctly for SQLite.
        """
        if not self.models:
            return
        
        # Setup relationships before creating tables
        self.setup_relationships()
        
        # First, create all tables without foreign keys
        tasks = []
        for model_name, model_class in self.models.items():
            # Create table without foreign keys
            tasks.append(self._create_table_without_foreign_keys(model_class))
        
        if tasks:
            await asyncio.gather(*tasks)
        
        # Then, add foreign keys to all tables
        # But only for databases that support ALTER TABLE ADD CONSTRAINT
        if "sqlite" not in self.connection_string:
            tasks = []
            for model_name, model_class in self.models.items():
                # Add foreign keys
                tasks.append(self._add_foreign_keys_to_table(model_class))
            
            if tasks:
                await asyncio.gather(*tasks)

        # Create explicitly requested non-unique indexes after tables exist.
        index_tasks = [self._create_indexes(model) for model in self.models.values()]
        if index_tasks:
            await asyncio.gather(*index_tasks)

    async def _create_indexes(self, model_class: Type["Model"]):
        """Create indexes declared with ``index=True`` or ``db_index=True``."""
        table_name = self._get_cached_table_name(model_class)
        scheme = self.connection_string.split("://", 1)[0].lower()

        for field_name, field in model_class._fields.items():
            if not field.index or field.primary_key or field.unique:
                continue

            safe_table = re.sub(r"[^A-Za-z0-9_]", "_", table_name)
            safe_field = re.sub(r"[^A-Za-z0-9_]", "_", field_name)
            index_name = f"idx_{safe_table}_{safe_field}"

            if scheme in {"sqlite", "postgresql", "postgres"}:
                query = (
                    f"CREATE INDEX IF NOT EXISTS {index_name} "
                    f"ON {table_name} ({field_name})"
                )
            elif scheme in {"mssql", "microsoft"}:
                query = (
                    "IF NOT EXISTS (SELECT 1 FROM sys.indexes "
                    f"WHERE name = '{index_name}' "
                    f"AND object_id = OBJECT_ID('{table_name}')) "
                    f"CREATE INDEX {index_name} ON {table_name} ({field_name})"
                )
            elif scheme == "oracle":
                query = (
                    "BEGIN EXECUTE IMMEDIATE 'CREATE INDEX "
                    f"{index_name} ON {table_name} ({field_name})'; "
                    "EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; "
                    "END IF; END;"
                )
            else:
                query = f"CREATE INDEX {index_name} ON {table_name} ({field_name})"

            try:
                await self.connection.execute(query)
            except Exception as error:
                # MySQL/MariaDB do not consistently support IF NOT EXISTS here.
                message = str(error).lower()
                if scheme in {"mysql", "mariadb", "aurora"} and any(
                    token in message
                    for token in ("duplicate key name", "already exists", "duplicate")
                ):
                    continue
                raise

    async def _create_table_without_foreign_keys(self, model_class: Type["Model"]):
        """
        Create table without foreign keys.
        For SQLite, foreign keys will be added in the CREATE TABLE statement.
        """
        table_name = self._get_cached_table_name(model_class)
        fields_sql_parts = []
        
        for field_name, field in model_class._fields.items():
            # Handle auto-increment primary key for PostgreSQL
            if (
                field.primary_key
                and field.auto_increment
                and (
                    self.connection_string.startswith("postgresql://")
                    or self.connection_string.startswith("postgres://")
                )
            ):
                # Use SERIAL for better PostgreSQL compatibility
                fields_sql_parts.append(f"{field_name} SERIAL PRIMARY KEY")
                continue
            # Fallback for other fields and databases
            field_sql = f"{field_name} {field.get_sql_type(database=self)}"
            if field.primary_key:
                field_sql += " PRIMARY KEY"
                if field.auto_increment:
                    if "sqlite" in self.connection_string:
                        field_sql += " AUTOINCREMENT"
                    elif any(
                        name in self.connection_string
                        for name in ("mysql", "mariadb", "aurora")
                    ):
                        field_sql += " AUTO_INCREMENT"
                    elif any(
                        name in self.connection_string
                        for name in ("mssql", "microsoft")
                    ):
                        field_sql += " IDENTITY(1,1)"
                    elif "oracle" in self.connection_string:
                        field_sql += " GENERATED BY DEFAULT AS IDENTITY"
            if not field.nullable:
                field_sql += " NOT NULL"
            if field.default is not None and not callable(field.default) and not field.primary_key:
                field_sql += f" DEFAULT {field.get_default_sql()}"
            if field.unique:
                field_sql += " UNIQUE"
            fields_sql_parts.append(field_sql)
        
        # Add foreign key constraints directly in CREATE TABLE for SQLite
        if "sqlite" in self.connection_string:
            for field_name, field in model_class._fields.items():
                if isinstance(field, ForeignKeyField):
                    ref_model = field.to
                    if isinstance(ref_model, str):
                        if ref_model == "self":
                            ref_model = model_class
                        elif ref_model in self.models:
                            ref_model = self.models[ref_model]
                    elif not isinstance(ref_model, str):
                        # If ref_model is a class, get it from models
                        ref_model_name = ref_model.__name__
                        if ref_model_name in self.models:
                            ref_model = self.models[ref_model_name]
                    
                    # Get the table name of the referenced model
                    ref_table = (
                        ref_model.get_table_name()
                        if hasattr(ref_model, "get_table_name")
                        else field.to
                    )
                    
                    # Get the primary key field name of the referenced model
                    ref_pk_field = "id"
                    if hasattr(ref_model, "_fields"):
                        for name, f in ref_model._fields.items():
                            if f.primary_key:
                                ref_pk_field = name
                                break
                    
                    constraint_name = f"fk_{table_name}_{field_name}"
                    
                    # Determine correct ON DELETE action
                    on_delete_clause = ""
                    if hasattr(field, 'on_delete') and field.on_delete:
                        on_delete_clause = f"ON DELETE {field.on_delete}"
                    
                    # Determine correct ON UPDATE action
                    on_update_clause = ""
                    if hasattr(field, 'on_update') and field.on_update:
                        on_update_clause = f"ON UPDATE {field.on_update}"
                    
                    # For SQLite, add foreign key constraint directly in CREATE TABLE
                    fk_constraint = (
                        f"CONSTRAINT {constraint_name} FOREIGN KEY ({field_name}) "
                        f"REFERENCES {ref_table}({ref_pk_field}) {on_delete_clause} {on_update_clause}"
                    )
                    fields_sql_parts.append(fk_constraint)
        
        # Create table without foreign keys (or with them for SQLite)
        columns_sql = ", ".join(fields_sql_parts)
        scheme = self.connection_string.split("://", 1)[0].lower()
        if scheme in {"mssql", "microsoft"}:
            create_query = (
                f"IF OBJECT_ID(N'{table_name}', N'U') IS NULL "
                f"CREATE TABLE {table_name} ({columns_sql})"
            )
        elif scheme == "oracle":
            escaped_ddl = f"CREATE TABLE {table_name} ({columns_sql})".replace(
                "'", "''"
            )
            create_query = (
                f"BEGIN EXECUTE IMMEDIATE '{escaped_ddl}'; "
                "EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; "
                "END IF; END;"
            )
        else:
            create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql})"
        await self.connection.execute(create_query)
        logger.info(f"Table {table_name} created without foreign keys (or with them for SQLite)")

    async def _add_foreign_keys_to_table(self, model_class: Type["Model"]):
        """
        Add foreign keys to an existing table.
        This is only used for databases that support ALTER TABLE ADD CONSTRAINT.
        """
        table_name = self._get_cached_table_name(model_class)
        
        # Add foreign key constraints
        for field_name, field in model_class._fields.items():
            if isinstance(field, ForeignKeyField):
                ref_model = field.to
                if isinstance(ref_model, str):
                    if ref_model == "self":
                        ref_model = model_class
                    elif ref_model in self.models:
                        ref_model = self.models[ref_model]
                elif not isinstance(ref_model, str):
                    # If ref_model is a class, get it from models
                    ref_model_name = ref_model.__name__
                    if ref_model_name in self.models:
                        ref_model = self.models[ref_model_name]
                
                # Get the table name of the referenced model
                ref_table = (
                    ref_model.get_table_name()
                    if hasattr(ref_model, "get_table_name")
                    else field.to
                )
                
                # Get the primary key field name of the referenced model
                ref_pk_field = "id"
                if hasattr(ref_model, "_fields"):
                    for name, f in ref_model._fields.items():
                        if f.primary_key:
                            ref_pk_field = name
                            break
                
                constraint_name = f"fk_{table_name}_{field_name}"
                
                # Determine correct ON DELETE action
                on_delete_clause = ""
                if hasattr(field, 'on_delete') and field.on_delete:
                    on_delete_clause = f"ON DELETE {field.on_delete}"
                
                # Determine correct ON UPDATE action
                on_update_clause = ""
                if (
                    hasattr(field, "on_update")
                    and field.on_update
                    and not self.connection_string.startswith("oracle://")
                ):
                    on_update_clause = f"ON UPDATE {field.on_update}"
                
                # Add foreign key constraint using ALTER TABLE
                alter_query = (
                    f"ALTER TABLE {table_name} ADD CONSTRAINT {constraint_name} "
                    f"FOREIGN KEY ({field_name}) REFERENCES {ref_table}({ref_pk_field}) "
                    f"{on_delete_clause} {on_update_clause}"
                )
                try:
                    await self.connection.execute(alter_query)
                    logger.info(
                        "Foreign key constraint %s added to %s",
                        constraint_name,
                        table_name,
                    )
                except Exception as error:
                    message = str(error).lower()
                    duplicate_markers = (
                        "already exists",
                        "duplicate",
                        "already an object named",
                        "ora-02264",
                        "name is already used",
                    )
                    if any(marker in message for marker in duplicate_markers):
                        logger.debug(
                            "Foreign key constraint %s already exists on %s",
                            constraint_name,
                            table_name,
                        )
                        continue
                    logger.error(
                        "Failed to add foreign key constraint %s: %s",
                        constraint_name,
                        error,
                    )
                    raise
        
        logger.info(f"Foreign keys added to table {table_name}")

    async def drop_tables(self, cascade: bool = True, if_exists: bool = True):
        """
        Drop tables for all registered models.

        Args:
            cascade: Whether to drop dependent objects.
            if_exists: Whether to ignore errors if table doesn't exist.
        """
        if not self.models:
            return
        table_names = []
        for model_name, model_class in self.models.items():
            table_names.append(self._get_cached_table_name(model_class))
        for table_name in reversed(table_names):
            await self.drop_table_by_name(
                table_name, cascade=cascade, if_exists=if_exists
            )

    async def drop_table(
        self, model_class: Type["Model"], cascade: bool = True, if_exists: bool = True
    ):
        """
        Drop table for a specific model.

        Args:
            model_class: The model class whose table to drop.
            cascade: Whether to drop dependent objects.
            if_exists: Whether to ignore errors if table doesn't exist.
        """
        table_name = self._get_cached_table_name(model_class)
        await self.drop_table_by_name(table_name, cascade=cascade, if_exists=if_exists)

    async def drop_table_by_name(
        self, table_name: str, cascade: bool = True, if_exists: bool = True
    ):
        """Drop a table using backend-compatible conditional syntax."""
        scheme = self.connection_string.split("://", 1)[0].lower()

        if scheme == "oracle":
            suffix = " CASCADE CONSTRAINTS" if cascade else ""
            ddl = f"DROP TABLE {table_name}{suffix}".replace("'", "''")
            if if_exists:
                query = (
                    f"BEGIN EXECUTE IMMEDIATE '{ddl}'; "
                    "EXCEPTION WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; "
                    "END IF; END;"
                )
            else:
                query = ddl
        elif scheme in {"mssql", "microsoft"}:
            ddl = f"DROP TABLE {table_name}"
            query = (
                f"IF OBJECT_ID(N'{table_name}', N'U') IS NOT NULL {ddl}"
                if if_exists
                else ddl
            )
        else:
            query_parts = ["DROP TABLE"]
            if if_exists:
                query_parts.append("IF EXISTS")
            query_parts.append(table_name)
            if cascade and scheme in {"postgresql", "postgres"}:
                query_parts.append("CASCADE")
            query = " ".join(query_parts)

        try:
            await self.connection.execute(query)
            logger.info("Table %s dropped successfully", table_name)
        except Exception as error:
            logger.error("Failed to drop table %s: %s", table_name, error)
            raise DatabaseError(f"Failed to drop table {table_name}: {error}") from error

    async def create_table(self, model_class: Type["Model"]):
        """Create one model table, its foreign keys, and declared indexes."""
        self.setup_relationships()
        await self._create_table_without_foreign_keys(model_class)
        if not self.connection_string.startswith("sqlite://"):
            await self._add_foreign_keys_to_table(model_class)
        await self._create_indexes(model_class)

    def _get_cached_table_name(self, model_class: Type["Model"]) -> str:
        """
        Get cached table name for better performance.

        Args:
            model_class: The model class to get the table name for.

        Returns:
            The table name.
        """
        model_name = model_class.__name__
        if model_name not in self._table_cache:
            self._table_cache[model_name] = model_class.get_table_name()
        return self._table_cache[model_name]


# --- Database Connection Implementations ---


class DatabaseConnection(ABC):
    """
    Abstract base class for database connections with connection pooling.
    """

    def __init__(self, connection_string: str):
        """
        Initialize the database connection.

        Args:
            connection_string: The database connection string.
        """
        self.connection_string = connection_string
        self.database: Optional["Database"] = None
        self.pool = None
        self._placeholder: str = "?"
        self._active_connection: ContextVar[Any] = ContextVar(
            f"ormax_active_connection_{id(self)}", default=None
        )

    def _get_active_connection(self):
        state = self._active_connection.get()
        if state is None:
            return None
        owner, connection = state
        return connection if owner is _get_running_task() else None

    def _set_active_connection(self, connection) -> None:
        if connection is None:
            self._active_connection.set(None)
        else:
            self._active_connection.set((_get_running_task(), connection))

    async def _acquire_connection(self):
        """Acquire a pooled connection across driver-specific pool APIs."""
        if self.pool is None:
            raise DatabaseError("Database not connected")
        acquired = self.pool.acquire()
        if inspect.isawaitable(acquired):
            return await acquired
        if hasattr(acquired, "__aenter__"):
            return await acquired.__aenter__()
        return acquired

    @asynccontextmanager
    async def connection_context(self):
        """Yield this task's pinned transaction connection or a pooled connection."""
        active = self._get_active_connection()
        if active is not None:
            yield active
            return
        if self.pool is None:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as connection:
            yield connection

    async def _release_active_connection(self) -> None:
        connection = self._get_active_connection()
        self._set_active_connection(None)
        if connection is None or self.pool is None:
            return
        result = self.pool.release(connection)
        if inspect.isawaitable(result):
            await result

    async def start_transaction(self) -> None:
        await self.execute("BEGIN")

    async def commit_transaction(self) -> None:
        await self.execute("COMMIT")

    async def rollback_transaction(self) -> None:
        await self.execute("ROLLBACK")

    @abstractmethod
    async def connect(self):
        """Connect to the database."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the database."""
        pass

    @abstractmethod
    async def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """
        Execute a query.

        Args:
            query: The SQL query to execute.
            params: Parameters for the query.

        Returns:
            The result of the query execution.
        """
        pass

    @abstractmethod
    async def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict]:
        """
        Fetch a single row.

        Args:
            query: The SQL query to execute.
            params: Parameters for the query.

        Returns:
            A dictionary representing the row, or None if no rows.
        """
        pass

    @abstractmethod
    async def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """
        Fetch all rows.

        Args:
            query: The SQL query to execute.
            params: Parameters for the query.

        Returns:
            A list of dictionaries representing the rows.
        """
        pass

    @abstractmethod
    def get_placeholder(self) -> str:
        """
        Get the placeholder for this database type.

        Returns:
            The placeholder string.
        """
        pass


class MySQLConnection(DatabaseConnection):
    """MySQL database connection implementation."""
    
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self.aiomysql: Any = None
        self.pool = None
        self._placeholder = "%s"
    
    async def connect(self):
        try:
            import aiomysql  # type: ignore
            self.aiomysql = aiomysql
            import urllib.parse
            parsed = urllib.parse.urlparse(self.connection_string)
            self.pool = await aiomysql.create_pool(
                host=parsed.hostname or "localhost",
                port=parsed.port or 3306,
                user=parsed.username,
                password=parsed.password,
                db=parsed.path.lstrip("/"),
                charset="utf8mb4",
                autocommit=True,
                pool_recycle=3600,
                minsize=5,
                maxsize=20,
            )
        except ImportError:
            raise DatabaseError("aiomysql is not installed")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to MySQL: {e}")
    
    async def disconnect(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()

    async def start_transaction(self) -> None:
        if not self.pool:
            raise DatabaseError("Database not connected")
        if self._get_active_connection() is not None:
            raise DatabaseError("Transaction already active")
        connection = await self._acquire_connection()
        self._set_active_connection(connection)
        try:
            await connection.begin()
        except Exception:
            await self._release_active_connection()
            raise

    async def commit_transaction(self) -> None:
        connection = self._get_active_connection()
        if connection is None:
            raise DatabaseError("No active transaction")
        try:
            await connection.commit()
        finally:
            await self._release_active_connection()

    async def rollback_transaction(self) -> None:
        connection = self._get_active_connection()
        if connection is None:
            raise DatabaseError("No active transaction")
        try:
            await connection.rollback()
        finally:
            await self._release_active_connection()
    
    async def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.connection_context() as conn:
            async with conn.cursor() as cur:
                # Convert ? placeholders to %s for MySQL
                if params and "?" in query:
                    param_count = query.count("?")
                    new_query = query
                    for i in range(param_count, 0, -1):
                        new_query = new_query.replace("?", "%s", 1)
                    await cur.execute(new_query, params or ())
                else:
                    await cur.execute(query, params or ())
                if query.strip().upper().startswith("INSERT"):
                    return cur.lastrowid
                return cur.rowcount
    
    async def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict]:
        if not self.pool or self.aiomysql is None:
            raise DatabaseError("Database not connected")
        async with self.connection_context() as conn:
            async with conn.cursor(self.aiomysql.DictCursor) as cur:
                # Convert ? placeholders to %s for MySQL
                if params and "?" in query:
                    param_count = query.count("?")
                    new_query = query
                    for i in range(param_count, 0, -1):
                        new_query = new_query.replace("?", "%s", 1)
                    await cur.execute(new_query, params or ())
                else:
                    await cur.execute(query, params or ())
                result = await cur.fetchone()
                return result
    
    async def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        if not self.pool or self.aiomysql is None:
            raise DatabaseError("Database not connected")
        async with self.connection_context() as conn:
            async with conn.cursor(self.aiomysql.DictCursor) as cur:
                # Convert ? placeholders to %s for MySQL
                if params and "?" in query:
                    param_count = query.count("?")
                    new_query = query
                    for i in range(param_count, 0, -1):
                        new_query = new_query.replace("?", "%s", 1)
                    await cur.execute(new_query, params or ())
                else:
                    await cur.execute(query, params or ())
                results = await cur.fetchall()
                return results
    
    def get_placeholder(self) -> str:
        return self._placeholder


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL database connection implementation."""
    
    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self.asyncpg: Any = None
        self._placeholder = "$"
    
    async def connect(self):
        try:
            import asyncpg  # type: ignore
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

    async def start_transaction(self) -> None:
        if not self.pool:
            raise DatabaseError("Database not connected")
        if self._get_active_connection() is not None:
            raise DatabaseError("Transaction already active")
        connection = await self._acquire_connection()
        self._set_active_connection(connection)
        try:
            await connection.execute("BEGIN")
        except Exception:
            await self._release_active_connection()
            raise

    async def commit_transaction(self) -> None:
        connection = self._get_active_connection()
        if connection is None:
            raise DatabaseError("No active transaction")
        try:
            await connection.execute("COMMIT")
        finally:
            await self._release_active_connection()

    async def rollback_transaction(self) -> None:
        connection = self._get_active_connection()
        if connection is None:
            raise DatabaseError("No active transaction")
        try:
            await connection.execute("ROLLBACK")
        finally:
            await self._release_active_connection()
    
    def _convert_placeholders(self, query: str, params: tuple) -> Tuple[str, tuple]:
        """Convert ? placeholders to $1, $2, ... for PostgreSQL."""
        if not params or "?" not in query:
            return query, params
            
        # Use the number of params instead of counting ? in query
        param_count = len(params)
        
        # Replace ? with $1, $2, ... from left to right
        new_query = query
        for i in range(1, param_count + 1):
            # Only replace the first occurrence each time
            new_query = new_query.replace("?", f"${i}", 1)
            
        return new_query, params
    
    async def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.connection_context() as conn:
            # Convert placeholders for PostgreSQL
            if params:
                query, params = self._convert_placeholders(query, params)
                # ارسال پارامترها به صورت جداگانه
                result = await conn.execute(query, *params)
            else:
                result = await conn.execute(query)
            
            # For INSERT queries, return the last inserted ID
            if query.strip().upper().startswith("INSERT"):
                try:
                    # Try to get the last inserted ID
                    if "RETURNING" in query.upper():
                        # If the query already has RETURNING, we need to fetch the result
                        if isinstance(result, list) and len(result) > 0:
                            return result[0].get("id")
                    else:
                        # Try to get the lastval() if available
                        last_id_row = await conn.fetchval("SELECT LASTVAL()")
                        return last_id_row
                except:
                    return None
            return result
    
    async def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.connection_context() as conn:
            # Convert placeholders for PostgreSQL
            if params:
                query, params = self._convert_placeholders(query, params)
                # ارسال پارامترها به صورت جداگانه
                result = await conn.fetchrow(query, *params)
            else:
                result = await conn.fetchrow(query)
            return dict(result) if result else None
    
    async def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.connection_context() as conn:
            # Convert placeholders for PostgreSQL
            if params:
                query, params = self._convert_placeholders(query, params)
                # ارسال پارامترها به صورت جداگانه
                results = await conn.fetch(query, *params)
            else:
                results = await conn.fetch(query)
            return [dict(row) for row in results]
    
    def get_placeholder(self) -> str:
        return "?"


class SQLiteConnection(DatabaseConnection):
    """SQLite database connection implementation."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self.aiosqlite = None
        self._connection = None
        self._placeholder = "?"
        self._prepared_statements = {}

    @asynccontextmanager
    async def _operation_context(self):
        """Serialize SQLite access and reject inherited cross-task transactions."""
        db_obj = self.database
        if db_obj is None:
            yield
            return

        current_task = _get_running_task()
        raw_state = db_obj._current_transaction_var.get()
        if (
            isinstance(raw_state, tuple)
            and raw_state[0] is not current_task
            and db_obj._sqlite_transaction_owner is raw_state[0]
        ):
            raise DatabaseError(
                "SQLite transaction contexts cannot be used from a child asyncio task"
            )

        if db_obj._sqlite_transaction_owner is current_task:
            yield
            return

        async with db_obj._sqlite_transaction_lock:
            yield

    async def connect(self):
        try:
            try:
                import aiosqlite  # type: ignore
            except ImportError:
                from . import _sqlite_fallback as aiosqlite

            self.aiosqlite = aiosqlite
            from .utils import parse_sqlite_path
            db_path = parse_sqlite_path(self.connection_string)
            self._connection = await aiosqlite.connect(db_path)
            # فعال‌سازی FOREIGN KEY در SQLite
            await self._connection.execute("PRAGMA foreign_keys = ON")
            await self._connection.execute("PRAGMA journal_mode = WAL")
            await self._connection.execute("PRAGMA synchronous = NORMAL")

        except Exception as e:
            raise DatabaseError(f"Failed to connect to SQLite: {e}") from e

    async def disconnect(self):
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._prepared_statements.clear()

    async def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        if not self._connection:
            raise DatabaseError("Database not connected")
        async with self._operation_context():
            db_obj = self.database
            in_tx = db_obj._transaction_level > 0 if db_obj else False
            try:
                cursor = await self._connection.execute(query, params or ())
                command = query.lstrip().split(None, 1)[0].upper() if query.strip() else ""
                transaction_command = command in {
                    "BEGIN",
                    "COMMIT",
                    "ROLLBACK",
                    "SAVEPOINT",
                    "RELEASE",
                }
                if not in_tx and not transaction_command:
                    await self._connection.commit()
                if command == "INSERT":
                    return cursor.lastrowid
                return getattr(cursor, "rowcount", 0)
            except Exception as error:
                if not in_tx:
                    try:
                        await self._connection.rollback()
                    except Exception:
                        pass
                if isinstance(error, DatabaseError):
                    raise
                raise DatabaseError(f"Database error: {error}") from error

    async def fetch_one(
        self, query: str, params: Optional[tuple] = None
    ) -> Optional[Dict]:
        if not self._connection:
            raise DatabaseError("Database not connected")
        async with self._operation_context():
            cursor = await self._connection.execute(query, params or ())
            row = await cursor.fetchone()
            if row is None:
                return None
            column_names = [description[0] for description in cursor.description]
            return dict(zip(column_names, row))

    async def fetch_all(
        self, query: str, params: Optional[tuple] = None
    ) -> List[Dict]:
        if not self._connection:
            raise DatabaseError("Database not connected")
        async with self._operation_context():
            cursor = await self._connection.execute(query, params or ())
            rows = await cursor.fetchall()
            if not rows:
                return []
            column_names = [description[0] for description in cursor.description]
            return [dict(zip(column_names, row)) for row in rows]

    def get_placeholder(self) -> str:
        return self._placeholder


class MSSQLConnection(DatabaseConnection):
    """Microsoft SQL Server database connection implementation."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self.aioodbc = None
        self._placeholder = "?"

    async def connect(self):
        try:
            import aioodbc  # type: ignore
            self.aioodbc = aioodbc
            
            if self.connection_string.startswith('mssql://'):
                self.connection_string = self.connection_string[len('mssql://'):].strip()
            
            elif self.connection_string.startswith('microsoft://'):
                self.connection_string = self.connection_string[len('microsoft://'):].strip()

            # ایجاد pool
            self.pool = await aioodbc.create_pool(
                dsn=self.connection_string,
                minsize=5,
                maxsize=20,
                autocommit=True
            )
        except ImportError:
            raise DatabaseError("aioodbc is not installed")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to Microsoft SQL Server: {e}")

    async def disconnect(self):
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()

    async def start_transaction(self) -> None:
        if not self.pool:
            raise DatabaseError("Database not connected")
        if self._get_active_connection() is not None:
            raise DatabaseError("Transaction already active")
        connection = await self._acquire_connection()
        self._set_active_connection(connection)
        try:
            async with connection.cursor() as cursor:
                await cursor.execute("BEGIN TRANSACTION")
        except Exception:
            await self._release_active_connection()
            raise

    async def commit_transaction(self) -> None:
        connection = self._get_active_connection()
        if connection is None:
            raise DatabaseError("No active transaction")
        try:
            await connection.commit()
        finally:
            await self._release_active_connection()

    async def rollback_transaction(self) -> None:
        connection = self._get_active_connection()
        if connection is None:
            raise DatabaseError("No active transaction")
        try:
            await connection.rollback()
        finally:
            await self._release_active_connection()

    async def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.connection_context() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params or ())
                if query.strip().upper().startswith("INSERT"):
                    # Try to get identity value
                    try:
                        await cur.execute("SELECT SCOPE_IDENTITY()")
                        row = await cur.fetchone()
                        return row[0] if row else None
                    except:
                        return None
                return cur.rowcount

    async def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.connection_context() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params or ())
                row = await cur.fetchone()
                if row is None:
                    return None
                # Get column names
                columns = [column[0] for column in cur.description]
                return dict(zip(columns, row))

    async def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.connection_context() as conn:
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
    """Oracle Database adapter using the official python-oracledb async API."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self.oracledb = None
        self._placeholder = ":{}"
        self._param_counter = 0

    async def connect(self):
        try:
            import oracledb  # type: ignore

            self.oracledb = oracledb
            import urllib.parse

            parsed = urllib.parse.urlparse(self.connection_string)
            dsn = f"{parsed.hostname}:{parsed.port or 1521}/{parsed.path.lstrip('/')}"
            # create_pool_async() is intentionally synchronous and returns an async pool.
            self.pool = oracledb.create_pool_async(
                user=parsed.username,
                password=parsed.password,
                dsn=dsn,
                min=1,
                max=20,
                increment=1,
            )
        except ImportError as error:
            raise DatabaseError(
                "oracledb is not installed; install ormax[oracle]"
            ) from error
        except Exception as error:
            raise DatabaseError(f"Failed to connect to Oracle: {error}") from error

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def start_transaction(self) -> None:
        if not self.pool:
            raise DatabaseError("Database not connected")
        if self._get_active_connection() is not None:
            raise DatabaseError("Transaction already active")
        connection = await self._acquire_connection()
        self._set_active_connection(connection)
        # Oracle begins a local transaction with the first DML statement.

    async def commit_transaction(self) -> None:
        connection = self._get_active_connection()
        if connection is None:
            raise DatabaseError("No active transaction")
        try:
            await connection.commit()
        finally:
            await self._release_active_connection()

    async def rollback_transaction(self) -> None:
        connection = self._get_active_connection()
        if connection is None:
            raise DatabaseError("No active transaction")
        try:
            await connection.rollback()
        finally:
            await self._release_active_connection()

    def _convert_placeholders(
        self, query: str, params: Optional[tuple]
    ) -> Tuple[str, tuple]:
        values = tuple(params or ())
        converted = query
        for index in range(1, len(values) + 1):
            converted = converted.replace("?", f":{index}", 1)
        return converted, values

    async def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        if not self.pool:
            raise DatabaseError("Database not connected")
        query, values = self._convert_placeholders(query, params)
        async with self.connection_context() as conn:
            with conn.cursor() as cur:
                await cur.execute(query, values)
                if self._get_active_connection() is None:
                    await conn.commit()
                if query.lstrip().upper().startswith("INSERT"):
                    return None
                return cur.rowcount

    async def fetch_one(
        self, query: str, params: Optional[tuple] = None
    ) -> Optional[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        query, values = self._convert_placeholders(query, params)
        async with self.connection_context() as conn:
            with conn.cursor() as cur:
                await cur.execute(query, values)
                row = await cur.fetchone()
                if row is None:
                    return None
                columns = [column[0] for column in cur.description]
                return dict(zip(columns, row))

    async def fetch_all(
        self, query: str, params: Optional[tuple] = None
    ) -> List[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        query, values = self._convert_placeholders(query, params)
        async with self.connection_context() as conn:
            with conn.cursor() as cur:
                await cur.execute(query, values)
                rows = await cur.fetchall()
                if not rows:
                    return []
                columns = [column[0] for column in cur.description]
                return [dict(zip(columns, row)) for row in rows]

    def get_placeholder(self) -> str:
        self._param_counter += 1
        return self._placeholder.format(self._param_counter)


class AuroraConnection(MySQLConnection):
    """Amazon Aurora connection - inherits from MySQL with some optimizations."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)

    async def connect(self):
        # Aurora is compatible with MySQL, so use MySQL connection
        # But with Aurora-specific optimizations
        try:
            import aiomysql  # type: ignore

            self.aiomysql = aiomysql
            import urllib.parse

            parsed = urllib.parse.urlparse(self.connection_string)
            self.pool = await aiomysql.create_pool(
                host=parsed.hostname or "localhost",
                port=parsed.port or 3306,
                user=parsed.username,
                password=parsed.password,
                db=parsed.path.lstrip("/"),
                charset="utf8mb4",
                autocommit=True,
                pool_recycle=300,  # Shorter recycle for Aurora
                minsize=5,
                maxsize=30,  # Larger pool for Aurora
                connect_timeout=10,
            )
        except ImportError:
            raise DatabaseError("aiomysql is not installed")
        except Exception as e:
            raise DatabaseError(f"Failed to connect to Amazon Aurora: {e}")

Model.DoesNotExist = DoesNotExist