# ormax/core.py
"""
Core module for the Ormax ORM.

This module contains the main classes and logic for interacting with databases,
defining models, managing relationships, and executing queries.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List, Type, Tuple, Union
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import uuid
import weakref

from .fields import DateTimeField, Field, ForeignKeyField
from .query import QuerySet
from .exceptions import DatabaseError, ValidationError, DoesNotExist

logger = logging.getLogger(__name__)

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

        fk_value = getattr(instance, self.foreign_key_field.name)
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
        if related_instance is None:
            setattr(instance, self.foreign_key_field.name, None)
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
            setattr(instance, self.foreign_key_field.name, pk_value)

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
        pk_value = getattr(instance, instance._meta.get("pk_field", "id"))
        kwargs[self.foreign_key_field.name] = pk_value
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
        pk_value = getattr(instance, instance._meta.get("pk_field", "id"))
        for related_instance in related_instances:
            if not isinstance(related_instance, self.related_model_class):
                raise ValueError(
                    f"Related instance must be of type {self.related_model_class.__name__}"
                )
            setattr(related_instance, self.foreign_key_field.name, pk_value)
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
        for related_instance in related_instances:
            if not isinstance(related_instance, self.related_model_class):
                raise ValueError(
                    f"Related instance must be of type {self.related_model_class.__name__}"
                )

            fk_value = getattr(related_instance, self.foreign_key_field.name)
            pk_value = getattr(instance, instance._meta.get("pk_field", "id"))

            if fk_value == pk_value:
                setattr(related_instance, self.foreign_key_field.name, None)
                await related_instance.save()

    async def clear(self, instance: "Model"):
        """
        Clear all relationships to this instance (set FK to NULL on all related objects).

        Args:
            instance: The model instance to clear relationships for.
        """
        qs = self.get_queryset(instance)
        await qs.update(**{self.foreign_key_field.name: None})


# --- Model Definition and Metaclass ---


class ModelMeta(type):
    """
    Metaclass for Ormax models.

    Handles the setup of fields, metadata, and relationships when a model class is defined.
    """

    _instances = weakref.WeakKeyDictionary()

    def __new__(cls, name, bases, attrs):
        """
        Create a new model class.

        Args:
            name: The name of the new class.
            bases: The base classes.
            attrs: The class attributes dictionary.

        Returns:
            The newly created model class.
        """
        if name == "Model":
            return super().__new__(cls, name, bases, attrs)

        fields = {}
        meta = attrs.get("_meta", {})
        field_items = [(k, v) for k, v in attrs.items() if isinstance(v, Field)]

        for key, value in field_items:
            fields[key] = value
            attrs.pop(key)

        attrs["_fields"] = fields
        attrs["_meta"] = meta
        attrs["_field_names"] = tuple(fields.keys())

        # Create the class
        new_class = super().__new__(cls, name, bases, attrs)

        # Initialize relationship manager
        new_class._relationships = RelationshipManager(new_class)

        return new_class


class Model(metaclass=ModelMeta):
    """
    Base class for all Ormax models.

    Provides the core functionality for database interaction, including
    saving, deleting, and querying model instances.
    """

    _database: "Database" = None
    _fields: Dict[str, Field] = {}
    _meta: Dict[str, Any] = {}
    _field_names: Tuple[str, ...] = ()
    _relationships: "RelationshipManager" = None
    __slots__ = ("_data", "_original_data", "_is_new", "__dict__")

    def __init_subclass__(cls, **kwargs):
        """
        Initialize subclass. Sets the name attribute on fields.
        """
        super().__init_subclass__(**kwargs)

        # Set the name attribute on fields
        for attr_name, attr_value in cls.__dict__.items():
            if isinstance(attr_value, Field):
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

        for field_name, field in self._fields.items():
            if field_name in kwargs:
                self.__setattr__(field_name, kwargs[field_name])
            elif field.default is not None:
                self.__setattr__(field_name, field.default)
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
            # Check for ForeignKeyField assignment (general case, not related manager)
            if isinstance(field, ForeignKeyField) and value is not None:
                # Get the related model class
                related_model = field.to
                if isinstance(related_model, str):
                    if self._database is None:
                        raise DatabaseError("Model not registered with database")
                    related_model = self._database.models.get(related_model)
                    if related_model is None:
                        raise DatabaseError(f"Related model '{field.to}' not found")
                # Try to get the primary key value from the related object
                # Check if value is a model instance first
                if hasattr(value, "_meta"):
                    try:
                        pk_field_name = value._meta.get("pk_field", "id")
                        pk_value = getattr(value, pk_field_name)
                    except AttributeError:
                        raise ValidationError(
                            f"Provided object '{value}' does not have a primary key field '{pk_field_name}'"
                        )
                else:
                    # If it's not a model instance, assume it's the pk value directly
                    pk_value = value
                # Validate and set the foreign key value
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

        Handles access to related managers and fields.

        Args:
            name: The name of the attribute.

        Returns:
            The attribute value.

        Raises:
            AttributeError: If the attribute does not exist.
        """
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
                # Return a queryset-like object for reverse relationships
                return reverse_manager.get_queryset(self)

        if name in self._fields:
            return self._data.get(name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @classmethod
    def get_table_name(cls) -> str:
        """
        Get the database table name for this model.

        Returns:
            The table name.
        """
        if hasattr(cls, "_cached_table_name"):
            return cls._cached_table_name
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

    async def save(self):
        """Save the model instance to the database."""
        # Process pre_save for all fields
        for field_name, field in self._fields.items():
            if hasattr(field, 'pre_save'):
                value = field.pre_save(self, self._is_new)
                if value is not None:
                    self._data[field_name] = value
        
        if self._is_new:
            await self._fast_insert()
        else:
            await self._fast_update()
    
        # Update instance state
        object.__setattr__(self, "_is_new", False)
        object.__setattr__(self, "_original_data", self._data.copy())

    async def _fast_insert(self):
        """Fast insert with correct placeholder handling."""
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
            values.append(self._data.get(field_name))
        
        # Generate query with correct placeholders
        if columns:
            if hasattr(connection, "get_placeholder") and callable(connection.get_placeholder):
                if connection.get_placeholder() == "$":
                    # PostgreSQL: $1, $2, ...
                    placeholders = [f"${i+1}" for i in range(len(columns))]
                else:
                    # MySQL, SQLite, etc.: ?, %s, ...
                    placeholders = [connection.get_placeholder() for _ in columns]
            else:
                # Default fallback
                placeholders = ["?" for _ in columns]
            query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        else:
            query = f"INSERT INTO {table_name} DEFAULT VALUES"
        
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
        table_name = self.get_table_name()
        set_clauses = []
        values = []
        connection = self._database.connection
        
        # Get primary key field
        pk_field = next((f for f in self._fields if self._fields[f].primary_key), "id")
        pk_value = self._data.get(pk_field)
        
        if pk_value is None:
            raise DatabaseError("Cannot update model without primary key value")
        
        # First, identify auto_now fields
        auto_now_fields = []
        for field_name, field in self._fields.items():
            if isinstance(field, DateTimeField) and field.auto_now:
                auto_now_fields.append(field_name)
        
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
                
            # Skip if it's an auto_now field (will be handled separately)
            if field_name in auto_now_fields:
                continue
                
            values.append(self._data.get(field_name))
        
        # Add auto_now fields
        for field_name in auto_now_fields:
            field = self._fields[field_name]
            # Get current time
            current_time = datetime.now(tz=timezone.utc if field.use_tz else None)
            values.append(current_time)
        
        # If no fields to update, return early
        if not values:
            return
        
        # Build set clauses for fields that have changed
        placeholder_index = 0
        for field_name, field in self._fields.items():
            if field.primary_key:
                continue
                
            # Skip auto-increment fields
            if field.auto_increment:
                continue
                
            # Skip if value hasn't changed
            if self._data.get(field_name) == self._original_data.get(field_name):
                continue
                
            # Skip if it's an auto_now field (will be handled separately)
            if field_name in auto_now_fields:
                continue
                
            set_clauses.append(f"{field_name} = ?")
            placeholder_index += 1
        
        # Add auto_now fields to set clauses
        for field_name in auto_now_fields:
            set_clauses.append(f"{field_name} = ?")
            placeholder_index += 1
        
        # Build WHERE clause
        where_clause = f"{pk_field} = ?"
        values.append(pk_value)
        
        # Build final query
        query = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {where_clause}"
        
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
        # Generate correct placeholder for WHERE clause
        connection = self._database.connection
        if hasattr(connection, "get_placeholder") and callable(
            connection.get_placeholder
        ):
            if connection.get_placeholder() == "$":
                placeholder = "$1"
            else:
                placeholder = connection.get_placeholder()
        else:
            placeholder = "?"
        query = f"DELETE FROM {table_name} WHERE {pk_field} = {placeholder}"
        await connection.execute(query, (pk_value,))
        object.__setattr__(self, "_is_new", True)

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
        if isinstance(objects[0], cls):
            data_list = [obj.to_dict() for obj in objects]
        else:
            data_list = objects
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
                    data[field_name] = field.default
                    
                # Handle auto_now_add and auto_now for DateTimeField
                if isinstance(field, DateTimeField):
                    if field.auto_now_add or field.auto_now:
                        data[field_name] = current_time
        
        table_name = cls.get_table_name()
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
        instances = []
        for data in data_list:
            # Prepare values for this row
            values = [data.get(field_name) for field_name in fields_to_insert]
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


# Attach a DoesNotExist exception to the Model class
Model.DoesNotExist = type("DoesNotExist", (Exception,), {})


# --- Database Connection and Management ---


class Database:
    """
    Main database interface for the Ormax ORM.

    Handles connection management, model registration, table creation,
    and relationship setup.
    """

    def __init__(self, connection_string: str):
        """
        Initialize the database connection.

        Args:
            connection_string: The database connection string.
        """
        self.connection_string = connection_string
        self.connection: DatabaseConnection = self._create_connection()
        self.models: Dict[str, Type["Model"]] = {}
        self._connected = False
        self._current_transaction = None
        self._table_cache = {}
        self._query_cache = {}

    def _create_connection(self) -> "DatabaseConnection":
        """
        Create the appropriate database connection based on the connection string.

        Returns:
            A DatabaseConnection instance.

        Raises:
            DatabaseError: If the database type is not supported.
        """
        if self.connection_string.startswith("mysql://"):
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
        await self.connection.connect()
        self._connected = True
        logger.info("Database connected successfully")

    async def disconnect(self):
        """
        Disconnect from the database.
        """
        await self.connection.disconnect()
        self._connected = False
        self._table_cache.clear()
        self._query_cache.clear()
        logger.info("Database disconnected")

    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for database transactions with savepoint support for nested transactions.
        Handles different database types correctly and safely with checkpoint support.
        """
        if not self._connected:
            raise DatabaseError("Database not connected")
        
        # تعیین نوع پایگاه‌داده
        if "postgresql" in self.connection_string or "postgres" in self.connection_string:
            db_type = "postgresql"
        elif "mysql" in self.connection_string or "mariadb" in self.connection_string:
            db_type = "mysql"
        elif "sqlite" in self.connection_string:
            db_type = "sqlite"
        elif "mssql" in self.connection_string:
            db_type = "mssql"
        else:
            db_type = "other"
        
        # تولید نام منحصر به فرد برای savepoint
        savepoint_name = f"sp_{uuid.uuid4().hex[:12]}"
        is_nested = self._current_transaction is not None
        transaction_level = 0
        
        try:
            # دریافت سطح فعلی تراکنش
            transaction_level = getattr(self, '_transaction_level', 0)
            
            if is_nested:
                # ایجاد savepoint برای تراکنش تو در تو
                if db_type in ["postgresql", "mysql", "sqlite"]:
                    await self.connection.execute(f"SAVEPOINT {savepoint_name}")
                    logger.debug(f"Created savepoint: {savepoint_name} (level: {transaction_level + 1})")
                elif db_type == "mssql":
                    await self.connection.execute(f"SAVE TRANSACTION {savepoint_name}")
                    logger.debug(f"Created save transaction: {savepoint_name} (level: {transaction_level + 1})")
                else:
                    raise DatabaseError(f"Savepoints not supported for {db_type}")
            else:
                # شروع یک تراکنش جدید
                await self.connection.execute("BEGIN")
                logger.info("Started new transaction")
            
            # ردیابی سطح تراکنش
            self._transaction_level = transaction_level + 1
            self._current_transaction = self.connection
            
            yield self  # اجازه اجرای کد داخل تراکنش
            
            # اگر تراکنش تو در تو بود، savepoint را آزاد می‌کنیم
            if is_nested and savepoint_name:
                if db_type in ["postgresql", "mysql", "sqlite"]:
                    await self.connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")
                    logger.debug(f"Released savepoint: {savepoint_name}")
                # برای MSSQL نیازی به آزاد کردن صریح نیست
            
            # فقط در سطح بالایی تراکنش، commit انجام می‌دهیم
            elif not is_nested:
                await self.connection.execute("COMMIT")
                logger.info("Committed transaction")
                
        except Exception as e:
            # برگرداندن به سطح مناسب در صورت خطا
            if is_nested and savepoint_name:
                if db_type in ["postgresql", "mysql", "sqlite"]:
                    await self.connection.execute(f"ROLLBACK TO {savepoint_name}")
                    logger.warning(f"Rolled back to savepoint: {savepoint_name} due to error: {str(e)}")
                elif db_type == "mssql":
                    await self.connection.execute(f"ROLLBACK TRANSACTION {savepoint_name}")
                    logger.warning(f"Rolled back to save transaction: {savepoint_name} due to error: {str(e)}")
            else:
                await self.connection.execute("ROLLBACK")
                logger.error(f"Rolled back entire transaction due to error: {str(e)}")
            raise
        finally:
            # به‌روزرسانی سطح تراکنش
            self._transaction_level = getattr(self, '_transaction_level', 1) - 1
            if self._transaction_level <= 0:
                self._transaction_level = 0
                self._current_transaction = None
                logger.debug("Transaction context cleared")

    def register_model(self, model_class: Type["Model"]):
        """
        Register a model class with this database.

        Args:
            model_class: The model class to register.
        """
        self.models[model_class.__name__] = model_class
        model_class._database = self

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
                        model_class._relationships.add_related_manager(
                            field_name, related_manager
                        )

                        # Create reverse manager
                        reverse_manager = ReverseRelatedManager(
                            related_model_class, model_class, field_name, field
                        )
                        reverse_name = field.related_name or f"{model_name.lower()}_set"
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
                    elif (
                        "mysql" in self.connection_string
                        or "mariadb" in self.connection_string
                    ):
                        field_sql += " AUTO_INCREMENT"
                    elif "mssql" in self.connection_string:
                        field_sql += " IDENTITY(1,1)"
            if not field.nullable:
                field_sql += " NOT NULL"
            if field.default is not None and not field.primary_key:
                field_sql += f" DEFAULT {field.get_default_sql()}"
            if field.unique:
                field_sql += " UNIQUE"
            fields_sql_parts.append(field_sql)
        
        # Add foreign key constraints directly in CREATE TABLE for SQLite
        if "sqlite" in self.connection_string:
            for field_name, field in model_class._fields.items():
                if isinstance(field, ForeignKeyField):
                    ref_model = field.to
                    if isinstance(ref_model, str) and ref_model in self.models:
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
        create_query = (
            f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(fields_sql_parts)})"
        )
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
                if isinstance(ref_model, str) and ref_model in self.models:
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
                
                # Add foreign key constraint using ALTER TABLE
                alter_query = (
                    f"ALTER TABLE {table_name} ADD CONSTRAINT {constraint_name} "
                    f"FOREIGN KEY ({field_name}) REFERENCES {ref_table}({ref_pk_field}) "
                    f"{on_delete_clause} {on_update_clause}"
                )
                try:
                    await self.connection.execute(alter_query)
                    logger.info(f"Foreign key constraint {constraint_name} added to {table_name}")
                except Exception as e:
                    logger.error(f"Failed to add foreign key constraint {constraint_name}: {e}")
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
        """
        Drop a table by name.

        Args:
            table_name: The name of the table to drop.
            cascade: Whether to drop dependent objects.
            if_exists: Whether to ignore errors if table doesn't exist.
        """
        query_parts = ["DROP TABLE"]
        if if_exists:
            query_parts.append("IF EXISTS")
        query_parts.append(table_name)
        # Handle database-specific CASCADE
        if (
            cascade
            and "sqlite" not in self.connection_string
            and "oracle" not in self.connection_string
        ):
            query_parts.append("CASCADE")
        elif cascade and "oracle" in self.connection_string:
            # Oracle uses different syntax
            query_parts[-1] = f"{table_name} CASCADE CONSTRAINTS"
        query = " ".join(query_parts)
        try:
            await self.connection.execute(query)
            logger.info(f"Table {table_name} dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop table {table_name}: {e}")
            raise DatabaseError(f"Failed to drop table {table_name}: {e}")

    async def create_table(self, model_class: Type["Model"]):
        """
        Create table for a specific model with optimizations.
        Handles foreign keys separately to avoid dependency issues in PostgreSQL.
        """
        table_name = self._get_cached_table_name(model_class)
        fields_sql_parts = []
        foreign_key_constraints = []
        
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
                    elif (
                        "mysql" in self.connection_string
                        or "mariadb" in self.connection_string
                    ):
                        field_sql += " AUTO_INCREMENT"
                    elif "mssql" in self.connection_string:
                        field_sql += " IDENTITY(1,1)"
                    # Oracle auto-increment handled separately
            if not field.nullable:
                field_sql += " NOT NULL"
            if field.default is not None and not field.primary_key:
                field_sql += f" DEFAULT {field.get_default_sql()}"
            if field.unique:
                field_sql += " UNIQUE"
            fields_sql_parts.append(field_sql)
        
        # First, create the table without foreign keys
        create_query = (
            f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(fields_sql_parts)})"
        )
        await self.connection.execute(create_query)
        logger.info(f"Table {table_name} created without foreign keys")
        
        # Then, add foreign key constraints
        for field_name, field in model_class._fields.items():
            if isinstance(field, ForeignKeyField):
                ref_model = field.to
                if isinstance(ref_model, str) and ref_model in self.models:
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
                
                # Add foreign key constraint using ALTER TABLE
                alter_query = (
                    f"ALTER TABLE {table_name} ADD CONSTRAINT {constraint_name} "
                    f"FOREIGN KEY ({field_name}) REFERENCES {ref_table}({ref_pk_field}) "
                    f"{on_delete_clause} {on_update_clause}"
                )
                try:
                    await self.connection.execute(alter_query)
                    logger.info(f"Foreign key constraint {constraint_name} added to {table_name}")
                except Exception as e:
                    logger.error(f"Failed to add foreign key constraint {constraint_name}: {e}")
                    raise
        
        logger.info(f"Table {table_name} created/verified with foreign keys")

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
        self.pool = None
        self._placeholder = None

    @abstractmethod
    async def connect(self):
        """Connect to the database."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the database."""
        pass

    @abstractmethod
    async def execute(self, query: str, params: tuple = None) -> Any:
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
    async def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
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
    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
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
        self.aiomysql = None
        self.pool = None
        self._placeholder = "%s"
    
    async def connect(self):
        try:
            import aiomysql
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
    
    async def execute(self, query: str, params: tuple = None) -> Any:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
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
    
    async def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
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
    
    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
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
    
    async def execute(self, query: str, params: tuple = None) -> Any:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
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
    
    async def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
            # Convert placeholders for PostgreSQL
            if params:
                query, params = self._convert_placeholders(query, params)
                # ارسال پارامترها به صورت جداگانه
                result = await conn.fetchrow(query, *params)
            else:
                result = await conn.fetchrow(query)
            return dict(result) if result else None
    
    async def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
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

    async def connect(self):
        try:
            import aiosqlite
            self.aiosqlite = aiosqlite
            import urllib.parse
            parsed = urllib.parse.urlparse(self.connection_string)
            db_path = parsed.path.lstrip("/")
            self._connection = await aiosqlite.connect(db_path)
            # فعال‌سازی FOREIGN KEY در SQLite
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
    """Microsoft SQL Server database connection implementation."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self.aioodbc = None
        self._placeholder = "?"

    async def connect(self):
        try:
            import aioodbc
            self.aioodbc = aioodbc
            
            if self.connection_string.startswith('mssql://'):
                self.connection_string = self.connection_string[len('mssql://'):].strip()
            
            elif self.connection_string.startswith('microsoft://'):
                self.connection_string = self.connection_string[len('mssql://'):].strip()

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

    async def execute(self, query: str, params: tuple = None) -> Any:
        if not self.pool:
            raise DatabaseError("Database not connected")
        async with self.pool.acquire() as conn:
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
    """Oracle database connection implementation."""

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
                increment=1,
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
                if query.strip().upper().startswith("INSERT"):
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
    """Amazon Aurora connection - inherits from MySQL with some optimizations."""

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