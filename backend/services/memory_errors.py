"""Exceptions for the in-process memory layer (persistence is owned by another service)."""


class MemoryError(Exception):
    """Base error for memory operations."""

    pass


class UserNotFoundError(MemoryError):
    """Raised when a user record is missing."""

    pass
