# ormax/exceptions.py
class DatabaseError(Exception):
    """Base exception for database errors"""
    pass


class ValidationError(Exception):
    """Exception for validation errors"""
    pass


class DoesNotExist(Exception):
    """Exception raised when object does not exist"""
    pass
