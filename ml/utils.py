import os
import subprocess

from django.conf import settings


def code_version():
    if os.getenv("GIT_COMMIT"):
        return os.getenv("GIT_COMMIT")[:40]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=settings.BASE_DIR,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"