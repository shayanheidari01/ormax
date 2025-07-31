
# Ormax ORM
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Async](https://img.shields.io/badge/async-await-brightgreen)](https://docs.python.org/3/library/asyncio.html)

**Ormax** is a high-performance, secure, and advanced asynchronous ORM for Python supporting MariaDB, MySQL, PostgreSQL, SQLite3, Microsoft SQL Server, Oracle Database, and Amazon Aurora.

## 🚀 Features
- **Multi-Database Support**: MariaDB, MySQL, PostgreSQL, SQLite3, MSSQL, Oracle, Aurora
- **Relationship Support**: Define and navigate ForeignKey relationships efficiently.
- **Fully Async**: Built with asyncio for maximum performance
- **Security First**: SQL injection protection and input validation
- **Connection Pooling**: Optimized database connections
- **Transaction Support**: ACID compliant transactions
- **Advanced QuerySet**: Powerful query capabilities including `select_related` and `prefetch_related`
- **Rich Field Types**: Comprehensive field validation
- **Bulk Operations**: Efficient bulk create, update, delete
- **Easy Configuration**: Simple setup and usage

## 📦 Installation
```bash
pip install ormax
```
Or install from source:
```bash
git clone https://github.com/shayanheidari01/ormax.git
cd ormax
pip install -e .
```

## 🛠️ Dependencies
```bash
# For MySQL/MariaDB
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

## 🚀 Quick Start

### 1. Define Models
```python
from ormax import Database, Model
from ormax.fields import *

class Author(Model):
    id = AutoField()
    name = CharField(max_length=100)

class Book(Model):
    id = AutoField()
    title = CharField(max_length=200)
    author = ForeignKeyField('Author', related_name='books') # Define relationship
```

### 2. Connect to Database
```python
# Initialize database connection
db = Database("sqlite:///example.db") # SQLite
# db = Database("mysql://user:password@localhost/dbname") # MySQL
# db = Database("postgresql://user:password@localhost/dbname") # PostgreSQL
# db = Database("mariadb://user:password@localhost/dbname") # MariaDB
# db = Database("mssql://user:password@localhost/dbname") # Microsoft SQL Server
# db = Database("oracle://user:password@localhost:1521/XE") # Oracle
# db = Database("aurora://user:password@cluster-endpoint/dbname") # Amazon Aurora

await db.connect()

# Register models
db.register_model(Author)
db.register_model(Book)

# Create tables (including relationship constraints)
await db.create_tables()
```

### 3. CRUD Operations

#### Create
```python
# Create single instance
author = await Author.create(name="J.K. Rowling")

# Create with relationship
book = await Book.create(title="Harry Potter", author=author) # Assign related object

# Bulk create
users_data = [
    {'name': 'Author 1'},
    {'name': 'Author 2'},
]
authors = await Author.objects().bulk_create(users_data)
```

#### Read
```python
# Fetch all
all_authors = await Author.objects().all()

# Filter
specific_author = await Author.objects().filter(name="J.K. Rowling").first()

# Navigate relationships (Forward)
book = await Book.objects().first()
# Access related author (requires async call in current simplified implementation)
# author_of_book = await book.get_author()

# Navigate relationships (Reverse)
author = await Author.objects().first()
# Access related books using the 'related_name'
# books_by_author = await author.get_books() # Gets a QuerySet

# Efficiently fetch related objects
# Select related (JOIN - forward relationships)
books_with_authors = await Book.objects().select_related('author').all()

# Prefetch related (separate query - forward & reverse relationships)
authors_with_books = await Author.objects().prefetch_related('books').all()
```

#### Update
```python
# Update single instance
author.name = "Updated Name"
await author.save()

# Bulk update
updated_count = await Book.objects().filter(title__startswith="Draft").update(title="Untitled")
```

#### Delete
```python
# Delete single instance
await book.delete()

# Bulk delete
deleted_count = await Book.objects().filter(title="Unwanted").delete()
```

## 🔧 Advanced Features

### Transactions
```python
async with db.transaction():
    author = await Author.create(name="New Author")
    post = await Book.create(title="New Book", author_id=author.id)
```

### Complex Queries
```python
# Chaining filters
active_authors = await Author.objects().filter(name__icontains="John")

# Ordering
recent_books = await Book.objects().order_by('-id', 'title')

# Pagination
page_1_books = await Book.objects().limit(10).offset(0)

# Distinct values
distinct_author_ids = await Book.objects().distinct().values_list('author_id', flat=True)
```

### Field Types
```python
class Product(Model):
    id = AutoField()
    name = CharField(max_length=100)
    description = TextField()
    price = DecimalField(max_digits=10, decimal_places=2)
    in_stock = BooleanField(default=True)
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)
    # category = ForeignKeyField('Category') # Example of another relationship
```

### Relationship Management
- **Forward Relationship (ForeignKey)**: Access the related object using the field name (e.g., `book.author`).
- **Reverse Relationship**: Access related objects using the `related_name` (e.g., `author.books` gives a QuerySet of related books).
- **`select_related(*relations)`**: Fetches the main object and specified related objects in a single database query (JOIN).
- **`prefetch_related(*relations)`**: Fetches the main objects and then fetches related objects in separate queries, efficiently handling multiple relationships.

## 📚 API Reference

### Database Class
- `Database(connection_string)` - Initialize database
- `connect()` - Connect to database
- `disconnect()` - Disconnect from database
- `transaction()` - Context manager for transactions
- `register_model(model)` - Register model with database
- `create_tables()` - Create all registered tables (including relationship constraints)
- `drop_tables()` - Drop all registered tables

### Model Class
- `objects()` - Get QuerySet for the model
- `create(**kwargs)` - Create and save a new instance
- `save()` - Save the instance
- `delete()` - Delete the instance
- `to_dict()` - Convert instance to dictionary
- `bulk_create(objects, batch_size)` - Efficiently create multiple instances

### QuerySet Class
- `filter(**kwargs)` - Filter objects
- `exclude(**kwargs)` - Exclude objects
- `get(**kwargs)` - Get a single object
- `all()` - Get all objects
- `first()` - Get the first object
- `last()` - Get the last object
- `count()` - Count objects
- `exists()` - Check if any objects exist
- `order_by(*fields)` - Order objects
- `limit(limit)` - Limit results
- `offset(offset)` - Offset results
- `distinct()` - Get distinct objects
- `select_related(*relations)` - Fetch related objects in the same query (JOIN)
- `prefetch_related(*relations)` - Fetch related objects in separate queries
- `update(**kwargs)` - Bulk update objects
- `delete()` - Bulk delete objects
- `values(*fields)` - Return values as dictionaries
- `values_list(*fields, flat=False)` - Return values as lists/tuples

For support, please open an issue on GitHub or contact the maintainers.

**Made with ❤️ using Python asyncio**
