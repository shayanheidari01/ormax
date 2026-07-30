"""Small asyncio-compatible SQLite adapter used when aiosqlite is unavailable.

The fallback keeps SQLite usable with no third-party dependency. Blocking sqlite3
calls are dispatched to worker threads and serialized per connection.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any, Iterable, Optional


class Cursor:
    def __init__(self, cursor: sqlite3.Cursor, lock: asyncio.Lock):
        self._cursor = cursor
        self._lock = lock
        self.lastrowid = cursor.lastrowid
        self.rowcount = cursor.rowcount
        self.description = cursor.description

    async def fetchone(self) -> Optional[tuple]:
        async with self._lock:
            return await asyncio.to_thread(self._cursor.fetchone)

    async def fetchall(self) -> list[tuple]:
        async with self._lock:
            return await asyncio.to_thread(self._cursor.fetchall)

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._cursor.close)


class Connection:
    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection
        self._lock = asyncio.Lock()

    async def execute(self, query: str, params: Iterable[Any] = ()) -> Cursor:
        async with self._lock:
            cursor = await asyncio.to_thread(self._connection.execute, query, tuple(params))
            return Cursor(cursor, self._lock)

    async def commit(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._connection.commit)

    async def rollback(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._connection.rollback)

    async def close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._connection.close)


async def connect(path: str) -> Connection:
    connection = await asyncio.to_thread(
        sqlite3.connect,
        path,
        check_same_thread=False,
    )
    return Connection(connection)
