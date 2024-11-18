"""Custom exceptions for the WhyHow API."""


class NotFoundException(Exception):
    """Exception raised when an item is not found."""

    def __init__(self, message: str):
        """Initialise the NotFoundException."""
        super().__init__(message)
