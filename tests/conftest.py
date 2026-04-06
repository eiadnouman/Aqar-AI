import os
import sys

import pytest


# Ensure "app.*" imports resolve from backend package during tests.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_PATH = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)


@pytest.fixture
def anyio_backend():
    return "asyncio"
