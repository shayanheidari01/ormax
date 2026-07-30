# Contributing

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Required checks

```bash
python -m compileall -q ormax tests
pytest -q
```

Keep changes backward-compatible where practical. Add a regression test for every bug fix and document user-visible changes in `CHANGELOG.md`.

## Database adapters

SQLite tests run without external services. Changes to PostgreSQL, MySQL/MariaDB/Aurora, MSSQL, or Oracle adapters should also be tested against a live instance of the affected database before release. Do not infer cross-database correctness from SQLite alone.

## SQL safety

Values must be parameter-bound. APIs that intentionally accept SQL fragments (`annotate`, `having`, `extra`, and `raw`) must be documented as trusted-input APIs.

## Pull requests

A pull request should contain:

1. A focused implementation.
2. Tests covering the changed behavior.
3. Documentation for public API changes.
4. A changelog entry when behavior changes.
