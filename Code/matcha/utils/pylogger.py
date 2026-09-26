import logging


def get_pylogger(name: str = __name__) -> logging.Logger:
    """Minimal python command-line logger (lightweight version)."""
    return logging.getLogger(name)
