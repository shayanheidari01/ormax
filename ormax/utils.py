# database_orm/utils.py
import hashlib
import secrets
from typing import Any, Dict
from datetime import datetime


def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token"""
    return secrets.token_hex(length)


def hash_password(password: str, salt: str = None) -> tuple:
    """Hash a password with salt"""
    if salt is None:
        salt = generate_secure_token(16)
    
    hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return hash_obj.hex(), salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Verify a password against hash"""
    hash_obj, _ = hash_password(password, salt)
    return hash_obj == hashed


def sanitize_input(value: Any) -> Any:
    """Basic input sanitization"""
    if isinstance(value, str):
        # Remove potentially dangerous characters
        return value.replace('\x00', '').replace('\x1a', '')
    return value


def dict_to_sql_insert(table_name: str, data: Dict[str, Any], placeholder_func) -> tuple:
    """Convert dictionary to SQL INSERT statement"""
    columns = list(data.keys())
    values = list(data.values())
    placeholders = [placeholder_func() for _ in columns]
    
    query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
    return query, tuple(values)


def dict_to_sql_update(table_name: str, data: Dict[str, Any], where_clause: str, placeholder_func) -> tuple:
    """Convert dictionary to SQL UPDATE statement"""
    set_clauses = []
    values = []
    
    for key, value in data.items():
        set_clauses.append(f"{key} = {placeholder_func()}")
        values.append(value)
    
    query = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {where_clause}"
    return query, tuple(values)