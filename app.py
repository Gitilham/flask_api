"""Compatibility import for deployments that previously referenced app.py.

The executable ASGI application is ``main:app``. The original Flask source is
preserved unchanged in ``legacy/app_flask_v21.py``.
"""

from main import app

__all__ = ["app"]

