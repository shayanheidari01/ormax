# Changelog

All notable changes to Ormax are documented here.

## 1.4.0 — 2026-07-31

### Fixed

- Pinned pooled transactions to one physical connection for their complete lifetime.
- Isolated transaction state by asyncio task and rejected inherited cross-task transaction use.
- Serialized concurrent SQLite outer transactions to protect the single connection.
- Ensured rollback is attempted for cancellation and other `BaseException` paths.
- Fixed nested transaction cleanup and savepoint handling, including `microsoft://`, Aurora, and Oracle classification.
- Fixed SQLite operation behavior when `aiosqlite` is unavailable by adding a standard-library async fallback.
- Fixed callable defaults, shared mutable defaults, inherited model fields, and abstract-model inheritance.
- Fixed synchronous lifecycle hooks and public exception consistency.
- Fixed field conversion on inserts, updates, filters, and database row hydration.
- Fixed foreign-key filtering with model instances and normalized `SET_NULL` actions.
- Fixed `select_related()`, reverse prefetching, `only()`, `defer()`, `values()`, and `values_list()` behavior.
- Fixed `count()` and `exists()` to avoid unnecessary full-result materialization.
- Fixed MSSQL `microsoft://` connection-string slicing and affected-row returns.
- Replaced the non-standard Oracle dependency with the official `oracledb` async pool API and Oracle bind syntax.
- Added MariaDB URL routing and backend-compatible MSSQL/Oracle create/drop DDL.

### Added

- Built-in SQLite fallback based on `sqlite3` and `asyncio.to_thread`.
- `class Meta` support alongside the existing `_meta` dictionary.
- Abstract base-model field inheritance.
- `Model.get_related()` and `Model.set_related()` convenience methods.
- `MultipleObjectsReturned` and a unified `OrmaxError` exception hierarchy.
- Query timeouts.
- Automatic non-unique index creation for `index=True` / `db_index=True`.
- Backend-specific `using_index()` hints with identifier validation.
- Public API cleanup, explicit exports, and a PEP 561 `py.typed` marker.
- Regression tests for defaults, inheritance, projections, relationships, transaction isolation, SQLite concurrency, indexes, and exceptions.

### Documentation and tooling

- Replaced unsupported performance claims and stale API examples.
- Added explicit limitations and backend test-scope notes.
- Updated package metadata for Python 3.9–3.13.
- Added CI test matrix and corrected GitHub Pages artifact scope.

## 1.3.0

Previous project baseline.
