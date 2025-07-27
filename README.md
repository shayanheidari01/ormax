
# Ormax ORM

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Async](https://img.shields.io/badge/async-await-brightgreen)](https://docs.python.org/3/library/asyncio.html)

**Ormax** is a high-performance, secure, and advanced asynchronous ORM for Python supporting MariaDB, MySQL, PostgreSQL, and SQLite3.

## 🚀 Features

- **Multi-Database Support**: MariaDB, MySQL, PostgreSQL, SQLite3
- **Fully Async**: Built with asyncio for maximum performance
- **Security First**: SQL injection protection and input validation
- **Connection Pooling**: Optimized database connections
- **Transaction Support**: ACID compliant transactions
- **Advanced QuerySet**: Powerful query capabilities
- **Rich Field Types**: Comprehensive field validation
- **Easy Configuration**: Simple setup and usage

## 📦 Installation

```bash
pip install ormax
```

Or install from source:
```bash
git clone https://github.com/yourusername/ormax.git
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
```

## 🚀 Quick Start

### 1. Define Models

```python
from ormax import Database, Model
from ormax.fields import *

class User(Model):
    table_name = "users"  # Simple table name setup
    
    id = IntegerField(primary_key=True, auto_increment=True)
    username = CharField(max_length=50, unique=True)
    email = EmailField(unique=True)
    is_active = BooleanField(default=True)
    created_at = DateTimeField(auto_now_add=True)

class Post(Model):
    _meta = {'table_name': 'posts'}  # Alternative setup
    
    id = IntegerField(primary_key=True, auto_increment=True)
    title = CharField(max_length=200)
    content = TextField()
    author_id = IntegerField(foreign_key='users.id')
    published = BooleanField(default=False)
```

### 2. Database Setup

```python
import asyncio

async def main():
    # Initialize database
    db = Database("sqlite:///example.db")  # SQLite
    # db = Database("mysql://user:password@localhost/dbname")  # MySQL
    # db = Database("postgresql://user:password@localhost/dbname")  # PostgreSQL
    # db = Database("mariadb://user:password@localhost/dbname")  # MariaDB
    
    await db.connect()
    
    # Register models
    db.register_model(User)
    db.register_model(Post)
    
    # Create tables
    await db.create_tables()
```

### 3. CRUD Operations

#### Create
```python
# Create single instance
user = await User.create(
    username="john_doe",
    email="john@example.com"
)

# Or create and save manually
user = User(username="jane_smith", email="jane@example.com")
await user.save()
```

#### Read
```python
# Get all records
all_users = await User.objects().all()

# Filter records
active_users = await User.objects().filter(is_active=True)

# Get single record
user = await User.objects().get(username="john_doe")

# Complex queries
published_posts = await Post.objects().filter(published=True).order_by('-created_at').limit(10)
```

#### Update
```python
# Update single instance
user.email = "newemail@example.com"
await user.save()

# Bulk update
updated_count = await Post.objects().filter(published=False).update(published=True)
```

#### Delete
```python
# Delete single instance
await user.delete()

# Bulk delete
deleted_count = await Post.objects().filter(published=False).delete()
```

## 🔧 Advanced Features

### Transactions
```python
async with db.transaction():
    user = await User.create(username="test", email="test@example.com")
    post = await Post.create(title="Test Post", content="Content", author_id=user.id)
```

### Complex Queries
```python
# Chaining filters
users = await User.objects().filter(is_active=True).exclude(username="admin")

# Ordering
posts = await Post.objects().order_by('-created_at', 'title')

# Pagination
page_1_posts = await Post.objects().limit(10).offset(0)
```

### Field Types
```python
class Product(Model):
    id = IntegerField(primary_key=True, auto_increment=True)
    name = CharField(max_length=100)
    description = TextField()
    price = FloatField()
    in_stock = BooleanField(default=True)
    created_at = DateTimeField(auto_now_add=True)
    updated_at = DateTimeField(auto_now=True)
    category_id = IntegerField(foreign_key='categories.id')
    tags = JSONField()  # Store JSON data
    website = URLField()
```

## 🏗️ Model Configuration

### Simple Table Name
```python
class User(Model):
    table_name = "app_users"  # Simple setup
    # ... fields
```

### Meta Configuration
```python
class Post(Model):
    _meta = {'table_name': 'blog_posts'}  # Traditional setup
    # ... fields
```

### Automatic Table Name
```python
class Category(Model):
    # Table name automatically becomes 'category'
    # ... fields
```

## 🔒 Security Features

- **SQL Injection Protection**: All queries use parameterized statements
- **Input Validation**: Built-in field validation
- **Data Sanitization**: Automatic data cleaning
- **Connection Security**: Secure connection handling

## ⚡ Performance Features

- **Connection Pooling**: Reuse database connections efficiently
- **Lazy Loading**: Queries execute only when needed
- **Batch Operations**: Efficient bulk operations
- **Memory Management**: Optimized memory usage

## 📊 Supported Databases

| Database | Connection String | Package Required |
|----------|-------------------|------------------|
| SQLite | `sqlite:///path/to/db.sqlite` | `aiosqlite` |
| MySQL | `mysql://user:pass@host:port/db` | `aiomysql` |
| PostgreSQL | `postgresql://user:pass@host:port/db` | `asyncpg` |
| MariaDB | `mariadb://user:pass@host:port/db` | `aiomysql` |

## 🧪 Example Usage

```python
import asyncio
from ormax import Database, Model
from ormax.fields import *

class User(Model):
    table_name = "users"
    id = IntegerField(primary_key=True, auto_increment=True)
    username = CharField(max_length=50, unique=True)
    email = EmailField(unique=True)
    is_active = BooleanField(default=True)

async def example():
    # Setup
    db = Database("sqlite:///example.db")
    await db.connect()
    db.register_model(User)
    await db.create_tables()
    
    # Create users
    user1 = await User.create(username="alice", email="alice@example.com")
    user2 = await User.create(username="bob", email="bob@example.com")
    
    # Query users
    all_users = await User.objects().all()
    active_users = await User.objects().filter(is_active=True)
    
    # Update user
    user1.email = "alice.new@example.com"
    await user1.save()
    
    # Close connection
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(example())
```

## 📚 API Reference

### Database Class
- `Database(connection_string)` - Initialize database
- `connect()` - Connect to database
- `disconnect()` - Disconnect from database
- `transaction()` - Context manager for transactions
- `register_model(model)` - Register model with database
- `create_tables()` - Create all registered tables

### Model Class
- `objects()` - Get QuerySet for model
- `create(**kwargs)` - Create and save instance
- `save()` - Save instance to database
- `delete()` - Delete instance from database
- `to_dict()` - Convert to dictionary

### QuerySet Class
- `filter(**kwargs)` - Filter records
- `exclude(**kwargs)` - Exclude records
- `order_by(*fields)` - Order records
- `limit(count)` - Limit results
- `offset(count)` - Offset results
- `all()` - Get all records
- `first()` - Get first record
- `get(**kwargs)` - Get single record
- `count()` - Count records
- `exists()` - Check if records exist
- `delete()` - Delete matching records
- `update(**kwargs)` - Update matching records

## 🛡️ Field Types

| Field | Description | Example |
|-------|-------------|---------|
| `CharField` | String with max length | `CharField(max_length=100)` |
| `TextField` | Long text | `TextField()` |
| `IntegerField` | Integer | `IntegerField()` |
| `BigIntegerField` | Big integer | `BigIntegerField()` |
| `FloatField` | Floating point | `FloatField()` |
| `BooleanField` | Boolean | `BooleanField()` |
| `DateTimeField` | DateTime | `DateTimeField(auto_now_add=True)` |
| `DateField` | Date | `DateField()` |
| `EmailField` | Validated email | `EmailField()` |
| `URLField` | Validated URL | `URLField()` |
| `JSONField` | JSON data | `JSONField()` |

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ for the Python community
- Inspired by Django ORM and SQLAlchemy
- Thanks to all contributors and users

## 🆘 Support

For support, please open an issue on GitHub or contact the maintainers.

---

**Made with ❤️ using Python asyncio**
```
