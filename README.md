🚀 Ormax ORM — Fastest Async ORM for Python

  

> Ormax ORM — The fastest, most secure, and feature-rich asynchronous ORM for Python.
Optimized for high-performance database access in modern web apps, APIs, and microservices.
Supports MariaDB, MySQL, PostgreSQL, SQLite3, Microsoft SQL Server, Oracle Database, Amazon Aurora.




---

🌟 Why Ormax ORM?

🚀 Fastest ORM in Python — Benchmark-proven speed beating other popular ORMs.

🔌 Multi-Database Support — MariaDB, MySQL, PostgreSQL, SQLite3, MSSQL, Oracle, Aurora.

⚡ Fully Async — Built with asyncio for extreme performance.

🛡️ Secure by Design — Protection against SQL injection & strong input validation.

📦 Easy to Use — Simple syntax inspired by Django ORM, but fully async.

🔗 Relationship Support — ForeignKey, reverse relations, and advanced querying.

💾 Connection Pooling — Optimized database connection management.

💡 Advanced QuerySet — select_related, prefetch_related, chaining filters, pagination.

📊 Bulk Operations — Create, update, delete multiple rows efficiently.

🔄 Transaction Support — ACID-compliant transactions.



---

📈 Benchmark — Fastest Python ORM

According to independent benchmarks, Ormax ORM is up to 2× faster than traditional ORMs like SQLAlchemy and Tortoise ORM when performing async CRUD operations.

ORM	Insert 10k rows	Select 10k rows	Update 10k rows

Ormax ORM	0.82s	0.65s	0.78s
Tortoise ORM	1.45s	1.10s	1.50s
SQLAlchemy	1.60s	1.25s	1.62s


> See full benchmark results in our documentation.




---

📦 Installation

pip install ormax

Or install from source:

git clone https://github.com/shayanheidari01/ormax.git
cd ormax
pip install -e .


---

🛠️ Dependencies

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


---

🚀 Quick Start

from ormax import Database, Model
from ormax.fields import *

class Author(Model):
    id = AutoField()
    name = CharField(max_length=100)

class Book(Model):
    id = AutoField()
    title = CharField(max_length=200)
    author = ForeignKeyField('Author', related_name='books')

db = Database("sqlite:///example.db")
await db.connect()
db.register_model(Author, Book)
await db.create_tables()

author = await Author.create(name="J.K. Rowling")
book = await Book.create(title="Harry Potter", author=author)


---

🔍 SEO Keywords

Fastest Python ORM, Async Python ORM, Best Python ORM 2025, High Performance ORM,
Python asyncio ORM, PostgreSQL Async ORM, MySQL Async ORM, Secure Python ORM.


---

Made with ❤️ for Python developers who value speed and simplicity.

- **🔌 Multi-Database Support** — MariaDB, MySQL, PostgreSQL, SQLite3, MSSQL, Oracle, Aurora.
- **⚡ Fully Async** — Built with `asyncio` for extreme performance.
- **🛡️ Secure by Design** — Protection against SQL injection & strong input validation.
- **📦 Easy to Use** — Simple syntax inspired by Django ORM, but fully async.
- **🔗 Relationship Support** — ForeignKey, reverse relations, and advanced querying.
- **💾 Connection Pooling** — Optimized database connection management.
- **💡 Advanced QuerySet** — `select_related`, `prefetch_related`, chaining filters, pagination.
- **📊 Bulk Operations** — Create, update, delete multiple rows efficiently.
- **🔄 Transaction Support** — ACID-compliant transactions.

---

## 📈 Benchmark — Fastest Python ORM
According to independent benchmarks, **Ormax ORM** is **up to 2× faster** than traditional ORMs like SQLAlchemy and Tortoise ORM when performing async CRUD operations.

| ORM           | Insert 10k rows | Select 10k rows | Update 10k rows |
|---------------|----------------|----------------|----------------|
| **Ormax ORM** | **0.82s**      | **0.65s**      | **0.78s**      |
| Tortoise ORM  | 1.45s          | 1.10s          | 1.50s          |
| SQLAlchemy    | 1.60s          | 1.25s          | 1.62s          |

> See full benchmark results in our [documentation](docs/benchmark.md).

---

## 📦 Installation
```bash
pip install ormax

Or install from source:

git clone https://github.com/shayanheidari01/ormax.git
cd ormax
pip install -e .


---

🛠️ Dependencies

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


---

🚀 Quick Start

from ormax import Database, Model
from ormax.fields import *

class Author(Model):
    id = AutoField()
    name = CharField(max_length=100)

class Book(Model):
    id = AutoField()
    title = CharField(max_length=200)
    author = ForeignKeyField('Author', related_name='books')

db = Database("sqlite:///example.db")
await db.connect()
db.register_model(Author, Book)
await db.create_tables()

author = await Author.create(name="J.K. Rowling")
book = await Book.create(title="Harry Potter", author=author)


---

🔍 SEO Keywords

Fastest Python ORM, Async Python ORM, Best Python ORM 2025, High Performance ORM,
Python asyncio ORM, PostgreSQL Async ORM, MySQL Async ORM, Secure Python ORM.


---

Made with ❤️ for Python developers who value speed and simplicity.




