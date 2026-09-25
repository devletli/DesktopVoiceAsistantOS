#!/usr/bin/env python3
"""
healthcheck.py — Docker HEALTHCHECK via heartbeat file (no HTTP API).

The main loop touches /tmp/jarvis_heartbeat every 5s.
This script checks mtime is within 30s; otherwise unhealthy.
"""
import sys
import time
from pathlib import Path

HEARTBEAT = Path("/tmp/jarvis_heartbeat")
MAX_AGE_SECONDS = 30

def main() -> int:
    if not HEARTBEAT.exists():
        print(f"heartbeat missing: {HEARTBEAT}", file=sys.stderr)
        return 1
    try:
        mtime = HEARTBEAT.stat().st_mtime
        age = time.time() - mtime
        if age > MAX_AGE_SECONDS:
            print(f"heartbeat stale: {age:.1f}s > {MAX_AGE_SECONDS}s", file=sys.stderr)
            return 1
        print(f"healthy: heartbeat age {age:.1f}s")
        return 0
    except Exception as e:
        print(f"healthcheck error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
