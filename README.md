
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
pip install ormax
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
    author = ForeignKeyField('Author', related_name='books')

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
    id = AutoField()
    username = CharField(max_length=50, unique=True)
    email = EmailField()
    created_at = DateTimeField(auto_now_add=True)
    settings = JSONField(default={})
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
    id = AutoField()
    title = CharField(max_length=200)
    content = TextField(max_length=5000)
    slug = SlugField(unique=True)
    views = PositiveIntegerField(default=0)
    metadata = JSONField()
```

### 3. **QuerySet API**
Ormax's `QuerySet` provides a powerful and chainable interface for querying data:

```python
# Filter and order
posts = await Post.objects().filter(views__gt=100).order_by("-created_at").all()

# Select specific fields
titles = await Post.objects().values_list("title", flat=True)

# Aggregations
total_views = await Aggregation.sum(Post.objects(), "views")
avg_views = await Aggregation.avg(Post.objects(), "views")

# Relationships
author = await Author.objects().get(id=1)
books = await author.books.all()  # Reverse relationship
```

### 4. **Relationships**
Ormax supports `ForeignKeyField` for forward and reverse relationships:

```python
# Forward relationship
book = await Book.objects().get(id=1)
author = await book.author.get()  # Access related Author

# Reverse relationship
author = await Author.objects().get(id=1)
books = await author.books.all()  # Get all Books by this Author
```

### 5. **Bulk Operations**
Efficiently create, update, or delete multiple records:

```python
# Bulk create
await Post.bulk_create([
    {"title": "Post 1", "content": "Content 1"},
    {"title": "Post 2", "content": "Content 2"}
], batch_size=100)

# Bulk update
await Post.objects().filter(views__lt=10).update(views=0)
```

### 6. **Transactions**
Use transactions for atomic operations:

```python
async with db.transaction():
    author = await Author.create(name="New Author")
    await Book.create(title="New Book", author=author)
```

### 7. **Connection Pooling**
Ormax uses connection pooling for efficient database access, optimized for high-concurrency workloads.

### 8. **Security Features**
- **Input Sanitization**: Prevents SQL injection with `sanitize_input` and `sanitize_dict`.
- **Validation**: Robust field validation ensures data integrity.
- **Secure Password Handling**: Functions like `hash_password` and `verify_password` for secure authentication.

---

## 📚 Advanced Usage

### Custom QuerySet Methods
Extend `QuerySet` for custom query logic:

```python
class CustomQuerySet(QuerySet):
    async def by_category(self, category: str):
        return self.filter(category=category)

class Post(Model):
    objects = CustomQuerySet.as_manager()
    category = CharField(max_length=50)

# Usage
posts = await Post.objects().by_category("news").all()
```

### Raw SQL Queries
Execute raw SQL for complex queries:

```python
results = await Post.objects().raw("SELECT * FROM post WHERE views > %s", (100,)).execute()
```

### Caching
Use `memoize_async` or `cached_property` for performance optimization:

```python
from ormax.utils import memoize_async

@memoize_async(maxsize=100)
async def get_user_stats(user_id: int):
    return await User.objects().filter(id=user_id).values("stats")
```

### Logging and Performance Monitoring
Ormax includes built-in logging and performance monitoring:

```python
from ormax.utils import setup_logging, PerformanceMonitor

setup_logging(level="DEBUG")
monitor = PerformanceMonitor()

async def some_operation():
    with monitor.record("operation"):
        await Post.objects().all()
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
```

### Model Registration
Register models before use:

```python
db.register_model(Author)
db.register_model(Book)
await db.create_tables()
```

---

## 📜 API Reference

### Core Classes
- **Database**: Manages connections, model registration, and table creation.
- **Model**: Base class for defining database models.
- **QuerySet**: Chainable query interface for filtering, ordering, and aggregating.
- **Field**: Base class for all field types, with validation and SQL generation.
- **RelationshipManager**: Handles forward and reverse relationships.

### Utility Functions
- **sanitize_input**: Prevents SQL injection by sanitizing input.
- **hash_password** / **verify_password**: Secure password handling.
- **generate_slug**: Creates URL-friendly slugs.
- **json_dumps** / **json_loads**: Custom JSON serialization for ORM types.
- **retry_async** / **timeout_async**: Decorators for reliable async operations.

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
