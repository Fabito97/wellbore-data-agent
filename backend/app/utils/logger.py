import logging

from app.core.config import settings


def get_logger(name: str = "wellbore-agent") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            settings.LOG_FORMAT,
            datefmt = "%H:%M:%S"  # Only time: HH:MM:SS
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger