#!/usr/bin/env python3
from __future__ import annotations

import sys
import uvicorn
from app.config import settings


def main():
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info" if not settings.debug else "debug",
    )


if __name__ == "__main__":
    main()
