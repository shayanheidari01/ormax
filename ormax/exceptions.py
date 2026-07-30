"""Exception hierarchy for Ormax."""


class OrmaxError(Exception):
    """Base exception for all Ormax-specific errors."""


class DatabaseError(OrmaxError):
    """Raised when a database operation or configuration fails."""


class ValidationError(OrmaxError):
    """Raised when model or field validation fails."""


class DoesNotExist(OrmaxError):
    """Raised when a query expected one object but found none."""


class MultipleObjectsReturned(OrmaxError):
    """Raised when a query expected one object but found multiple rows."""
