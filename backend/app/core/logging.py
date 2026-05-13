import logging
import os
import sys

def setup_logging():
    """Configure system-wide standardized logging to both console and file."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if os.getenv("LOG_TO_FILE", "false").strip().lower() in {"1", "true", "yes"}:
        handlers.append(logging.FileHandler("server.log", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("aqarai")

logger = setup_logging()
