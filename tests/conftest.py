import os
from pathlib import Path
import tempfile


_TEST_DATABASE = Path(tempfile.gettempdir()) / f"xiaoai-test-{os.getpid()}.db"
os.environ["DATABASE_PATH"] = str(_TEST_DATABASE)


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    try:
        _TEST_DATABASE.unlink(missing_ok=True)
    except PermissionError:
        # Windows may hold a transient test-process lock until interpreter exit.
        pass
