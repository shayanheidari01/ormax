# 🚀 Ormax ORM — The Fastest Async ORM for Python

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Async](https://img.shields.io/badge/async-await-brightgreen)](https://docs.python.org/3/library/asyncio.html)

> **Ormax ORM** is a **high-performance**, **secure**, and **feature-rich** asynchronous Object-Relational Mapping (ORM) library for Python. Built for modern web applications, APIs, and microservices, Ormax delivers **unmatched speed** and supports multiple databases, including **MariaDB**, **MySQL**, **PostgreSQL**, **SQLite3**, **Microsoft SQL Server**, **Oracle**, and **Amazon Aurora**.

---

## 🌟 Why Choose Ormax ORM?

- **🚀 Blazing Fast**: Up to **2× faster** than other popular ORMs like SQLAlchemy and Tortoise ORM (see [Benchmarks](#-benchmarks)).
- **🔌 Multi-Database Support**: Seamlessly works with MariaDB, MySQL, PostgreSQL, SQLite3, MSSQL, Oracle, and Aurora.
- **⚡ Fully Asynchronous**: Built on `asyncio` for optimal performance in async applications.
- **🛡️ Secure by Design**: Robust input validation and protection against SQL injection.
- **📦 Intuitive API**: Inspired by Django ORM, but optimized for async workflows with a simple, Pythonic syntax.
- **🔗 Advanced Relationships**: Supports `ForeignKey`, reverse relationships, `select_related`, and `prefetch_related`.
- **💾 Connection Pooling**: Efficient connection management for high-concurrency workloads.
- **📊 Powerful QuerySet**: Chaining filters, annotations, aggregations, and bulk operations.
- **🔄 Transaction Support**: ACID-compliant transactions for reliable data operations.
- **🛠️ Flexible Field Types**: Comprehensive field types like `CharField`, `JSONField`, `UUIDField`, and more.

---

## 📈 Benchmarks

Ormax ORM consistently outperforms other Python ORMs in async CRUD operations, making it ideal for high-performance applications.

| ORM           | Insert 10k Rows | Select 10k Rows | Update 10k Rows |
|---------------|-----------------|-----------------|-----------------|
| **Ormax ORM** | **0.82s**       | **0.65s**       | **0.78s**       |
| Tortoise ORM  | 1.45s           | 1.10s           | 1.50s           |
| SQLAlchemy    | 1.60s           | 1.25s           | 1.62s           |

> Full benchmark details available in [docs/benchmark.md](docs/benchmark.md).

---

## 📦 Installation

Install Ormax ORM using pip:

```bash
pip install -U ormax
```

Or install from source:

```bash
git clone https://github.com/shayanheidari01/ormax.git
cd ormax
pip install -e .
```

### Dependencies

Depending on your database, install the required async driver:

```bash
# For MySQL/MariaDB/Amazon Aurora
pip install aiomysql

# For PostgreSQL
pip install asyncpg

# For SQLite
pip install aiosqlite

# For Microsoft SQL Server
pip install aioodbc

# For Oracle Database
pip install async-oracledb
```

---

## 🚀 Quick Start

Get started with Ormax in just a few lines of code:

```python
import asyncio
from ormax import Database, Model
from ormax.fields import AutoField, CharField, ForeignKeyField

# Define models
class Author(Model):
    id = AutoField()
    name = CharField(max_length=100)

class Book(Model):
    id = AutoField()
    title = CharField(max_length=200)
    author = ForeignKeyField(Author, related_name='books')

# Initialize database
db = Database("sqlite:///example.db")

async def main():
    # Connect to database and register models
    await db.connect()
    db.register_model(Author)
    db.register_model(Book)
    await db.create_tables()

    # Create instances
    author = await Author.create(name="J.K. Rowling")
    book = await Book.create(title="Harry Potter", author=author)

    # Query data
    books = await Book.objects().filter(author=author).all()
    print(books)

# Run the async application
asyncio.run(main())
```

---

## 🛠️ Key Features

### 1. **Model Definition**
Define database models using a clean, class-based syntax. Ormax supports a wide range of field types for flexible data modeling.

```python
from ormax import Model
from ormax.fields import *

class User(Model):
    table_name = "users_table"  # Optional custom table name
    id = AutoField()
    username = CharField(max_length=50, unique=True)
    email = EmailField(unique=True)
    password_hash = CharField(max_length=128)
    is_active = BooleanField(default=True)
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)
```

### 2. **Supported Field Types**
Ormax provides a comprehensive set of field types, each with built-in validation:

- **Basic Types**: `CharField`, `TextField`, `IntegerField`, `BigIntegerField`, `SmallIntegerField`, `FloatField`, `DecimalField`, `BooleanField`
- **Date/Time**: `DateTimeField`, `DateField`, `TimeField`
- **Specialized**: `EmailField`, `URLField`, `UUIDField`, `IPAddressField`, `SlugField`, `JSONField`, `BinaryField`
- **Auto-Incrementing**: `AutoField`, `BigAutoField`, `SmallAutoField`
- **Relationships**: `ForeignKeyField` (with `related_name` and `on_delete` options)
- **Positive Variants**: `PositiveIntegerField`, `PositiveSmallIntegerField`

Example:
```python
class Post(Model):
    _meta = {'table_name': 'blog_posts'}  # Alternative way to set table name
    id = AutoField()
    title = CharField(max_length=200)
    content = TextField()
    # ForeignKey with on_delete option
    author = ForeignKeyField(User, related_name='posts', nullable=True, on_delete="SET NULL")
    published = BooleanField(default=False)
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)
```

### 3. **QuerySet API**
Ormax's `QuerySet` provides a powerful and chainable interface for querying data:

```python
# Get all users
all_users = await User.objects().all()

# Filter users
active_users = await User.objects().filter(is_active=True)

# Get user by username
user = await User.objects().get(username="john_doe")

# Update user
user.email = "newemail@example.com"
await user.save()

# Count posts
post_count = await Post.objects().count()

# Update multiple records
updated_count = await Post.objects().filter(published=False).update(published=True)

# Prefetch related objects
users = await User.objects().prefetch_related('posts').all()
for user in users:
    posts = await user.posts.all()
    print(f"{user.username} has {len(posts)} posts")
```

### 4. **Relationships**
Ormax supports `ForeignKeyField` for forward and reverse relationships:

```python
# Forward relationship
post = await Post.objects().get(id=1)
author = await post.author.get()  # Access related User

# Reverse relationship
user = await User.objects().get(id=1)
posts = await user.posts.all()  # Get all Posts by this User
```

### 5. **Bulk Operations**
Efficiently create, update, or delete multiple records:

```python
# Bulk create
users_data = [
    {"username": f"user{i}", "email": f"user{i}@example.com", "password_hash": f"hash{i}"} 
    for i in range(10)
]
created_users = await User.bulk_create(users_data)

# Bulk update
await Post.objects().filter(published=False).update(published=True)
```

### 6. **Transactions**
Use transactions for atomic operations:

```python
# Transaction context manager
async with db.transaction():
    new_user = await User.create(
        username="transaction_user",
        email="transaction@example.com",
        password_hash="transaction_hash"
    )
    new_post = await Post.create(
        title="Transaction Post",
        content="Created in transaction",
        author=new_user
    )
```

### 7. **Connection Pooling**
Ormax uses connection pooling for efficient database access, optimized for high-concurrency workloads. Each database type has specific connection settings:

```python
# MySQL/Aurora optimized connection
db = Database("mysql://root:password@localhost:3306/mydb")

# PostgreSQL connection
db = Database("postgresql://postgres:password@localhost:5432/mydb")

# SQLite in-memory database
db = Database("sqlite:///:memory:")
```

### 8. **Security Features**
- **Input Sanitization**: Prevents SQL injection with built-in validation
- **Validation**: Robust field validation ensures data integrity
- **Secure Password Handling**: Comprehensive validation for fields like `EmailField`

---

## 📚 Advanced Usage

### Custom QuerySet Methods
Extend `QuerySet` for custom query logic:

```python
class CustomQuerySet(QuerySet):
    async def active(self):
        return self.filter(is_active=True)
    
    async def by_email_domain(self, domain):
        return self.filter(email__endswith=f"@{domain}")

class User(Model):
    objects = CustomQuerySet.as_manager()
    
    # Fields here...

# Usage
active_users = await User.objects().active().all()
gmail_users = await User.objects().by_email_domain("gmail.com").all()
```

### Nested Transactions with Savepoints
Ormax supports nested transactions using savepoints:

```python
async with db.transaction():
    # Outer transaction
    user = await User.create(username="main_user", email="main@example.com")
    
    try:
        async with db.transaction():
            # Nested transaction (savepoint)
            post = await Post.create(title="Nested", content="Nested content", author=user)
            # This would roll back only the nested transaction
            raise Exception("Simulated error")
    except Exception:
        pass
    
    # This will still be committed
    await Post.create(title="After Nested", content="Content after nested", author=user)
```

### Raw SQL Queries
Execute raw SQL for complex queries:

```python
# Execute raw query
results = await db.connection.execute(
    "SELECT * FROM users_table WHERE is_active = %s",
    (True,)
)

# Fetch one result
result = await db.connection.fetch_one(
    "SELECT * FROM users_table WHERE username = %s",
    ("john_doe",)
)

# Fetch all results
results = await db.connection.fetch_all(
    "SELECT * FROM blog_posts WHERE published = %s ORDER BY created_at DESC",
    (True,)
)
```

### Performance Optimization
Ormax includes features for optimizing database performance:

```python
# Prefetch related objects to avoid N+1 queries
users = await User.objects().prefetch_related('posts').all()

# Select only specific fields
users = await User.objects().values('id', 'username').all()

# Limit and offset for pagination
page_2 = await Post.objects().order_by('-created_at').limit(10).offset(10).all()
```

---

## 🔧 Configuration

### Database Connection
Create a `Database` instance with a connection string:

```python
# SQLite
db = Database("sqlite:///example.db")

# PostgreSQL
db = Database("postgresql://user:password@localhost:5432/dbname")

# MySQL/MariaDB
db = Database("mysql://user:password@localhost:3306/dbname")

# Microsoft SQL Server
db = Database("mssql://user:password@localhost:1433/dbname")

# Oracle
db = Database("oracle://user:password@localhost:1521/orcl")
```

### Model Registration
Register models before use:

```python
# Register models
db.register_model(User)
db.register_model(Post)
db.register_model(Category)

# Create tables
await db.create_tables()

# Drop tables (with cascade option)
await db.drop_tables(cascade=True, if_exists=True)
```

---

## 📜 API Reference

### Core Classes
- **Database**: Manages connections, model registration, and table creation.
  - `connect()`: Establish database connection
  - `disconnect()`: Close database connection
  - `register_model(model_class)`: Register a model class
  - `create_tables()`: Create tables for all registered models
  - `drop_tables(cascade=False, if_exists=False)`: Drop tables for all registered models
  - `transaction()`: Context manager for database transactions

- **Model**: Base class for defining database models.
  - `create(**kwargs)`: Create and save a new instance
  - `save()`: Save the model instance
  - `delete()`: Delete the model instance
  - `bulk_create(objects, batch_size=1000)`: Bulk create multiple instances
  - `objects()`: Get a QuerySet for this model

- **QuerySet**: Chainable query interface for filtering, ordering, and aggregating.
  - `filter(**kwargs)`: Add filter conditions
  - `exclude(**kwargs)`: Add exclude conditions
  - `get(**kwargs)`: Get a single record
  - `all()`: Get all records
  - `first()`: Get the first record
  - `last()`: Get the last record
  - `count()`: Count records
  - `exists()`: Check if records exist
  - `update(**kwargs)`: Bulk update records
  - `delete()`: Bulk delete records
  - `order_by(*fields)`: Order results
  - `limit(n)`: Limit results
  - `offset(n)`: Offset results
  - `prefetch_related(*relations)`: Prefetch related objects
  - `select_related(*relations)`: Select related objects in the same query

- **Field**: Base class for all field types, with validation and SQL generation.
  - `primary_key`: Whether this field is a primary key
  - `auto_increment`: Whether this field auto-increments
  - `nullable`: Whether this field can be NULL
  - `default`: Default value for the field
  - `unique`: Whether this field must be unique
  - `index`: Whether this field should be indexed

- **RelationshipManager**: Handles forward and reverse relationships.

### Field Types
Ormax provides a comprehensive set of field types:

- **AutoField**: Auto-incrementing primary key field
- **CharField**: Character field with max_length
- **TextField**: Large text field
- **IntegerField**: Integer field
- **BigIntegerField**: Large integer field
- **SmallIntegerField**: Small integer field
- **FloatField**: Floating point field
- **DecimalField**: Decimal field for precise calculations
- **BooleanField**: Boolean field
- **DateTimeField**: Date and time field
- **DateField**: Date field
- **TimeField**: Time field
- **EmailField**: Email address field with validation
- **URLField**: URL field with validation
- **UUIDField**: UUID field
- **IPAddressField**: IP address field
- **SlugField**: URL-friendly slug field
- **JSONField**: JSON data field
- **BinaryField**: Binary data field
- **ForeignKeyField**: Foreign key relationship field

### Error Types
Ormax provides specific exceptions for different error scenarios:

- **DatabaseError**: Base database error
- **ValidationError**: Validation error
- **DoesNotExist**: Record does not exist
- **MultipleObjectsReturned**: Multiple records returned when one expected

---

## 🔍 SEO Keywords
`Fastest Python ORM`, `Async Python ORM`, `Best Python ORM 2025`, `High Performance ORM`, `Python asyncio ORM`, `PostgreSQL Async ORM`, `MySQL Async ORM`, `Secure Python ORM`, `ORM for Microservices`, `Python Database Library`

---

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

---

## 📄 License
Ormax ORM is licensed under the [MIT License](LICENSE).

---

**Made with ❤️ for Python developers who value speed, simplicity, and reliability.**