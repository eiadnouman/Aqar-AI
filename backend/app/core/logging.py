import logging
import sys

def setup_logging():
    """Configure system-wide standardized logging to both console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("server.log", encoding="utf-8")
        ]
    )
    return logging.getLogger("aqarai")

logger = setup_logging()
