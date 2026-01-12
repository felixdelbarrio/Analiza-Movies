# logger y utilidades de logging
from __future__ import annotations

import logging

from server.api.settings import Settings


def configure_logging(settings: Settings) -> logging.Logger:
    """
    Configuración mínima:
    - Respetamos handlers/format de quien ejecute (uvicorn, gunicorn, etc.).
    - Ajustamos nivel global según env.
    """
    root = logging.getLogger()
    root.setLevel(settings.log_level)

    logger = logging.getLogger("analiza_api")
    logger.setLevel(settings.log_level)
    return logger