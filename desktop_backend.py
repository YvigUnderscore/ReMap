from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path

from backend.app import create_app


_FAULT_LOG_HANDLE = None


def enable_fault_logging() -> None:
    global _FAULT_LOG_HANDLE
    log_dir = Path(__file__).resolve().parent / "backend_state"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "desktop-backend-fault.log"
    _FAULT_LOG_HANDLE = log_path.open("a", encoding="utf-8")
    faulthandler.enable(file=_FAULT_LOG_HANDLE, all_threads=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ReMap desktop backend service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    enable_fault_logging()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
