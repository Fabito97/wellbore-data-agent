import logging

def get_logger(name: str = "wellbore-agent") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
       "%(asctime)s - %(name)s [%(levelname)s]:   %(message)s",
            datefmt = "%H:%M:%S"  # Only time: HH:MM:SS
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger