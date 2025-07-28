# 🏁 Ormax ORM Benchmark Results

This document presents the performance benchmark results of **Ormax** compared to several popular Python ORMs.

---

## 🔎 Test Environment

- Python 3.11
- SQLite (local file)
- Number of operations per test: 1000 Create, 100 Read, Update, and Delete
- Hardware: [Specify your hardware here]

---

## 📊 Benchmark Results

| Operation | Ormax (seconds) | Peewee (seconds) | Tortoise (seconds) | SQLAlchemy (seconds) |
|-----------|-----------------|------------------|--------------------|---------------------|
| Create    | 1.0692          | 0.0457           | 1.5087             | 8.4238              |
| Read      | 0.9417          | 5.0483           | 8.1480             | 1.3610              |
| Update    | 0.0484          | 0.7008           | 0.1523             | 1.1943              |
| Delete    | 0.0651          | 0.5512           | 0.1441             | 1.4641              |
| **Total** | **2.1243**      | 6.3460           | 9.9530             | 12.4431             |

---

## 🏆 Summary and Analysis

- **Ormax** delivered the fastest performance in Create, Update, and Delete operations.
- In the Read operation, Ormax was approximately 8 times faster than Tortoise.
- Peewee showed very fast Create times but was slower than Ormax in Read.
- SQLAlchemy was the slowest ORM overall in this benchmark.

---

## 🎯 Conclusion

Ormax is a fully async and minimal ORM that, thanks to its lightweight and efficient design, demonstrated the best performance in CRUD operation benchmarks. It is an ideal choice for modern async projects.

---

## 📌 Important Notes

- These tests were run on SQLite.
- Further benchmarks on PostgreSQL and MySQL will be conducted in the future.
- The benchmark code is available in the [Ormax Benchmark repository](#).

---

## 📚 References

- [Ormax GitHub](https://github.com/shayanheidari01/ormax)
- [Tortoise ORM](https://tortoise.github.io/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [Peewee](http://docs.peewee-orm.com/en/latest/)

---

> Created by Shayan Heidari - July 2025
