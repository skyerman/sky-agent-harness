"""Windows command launcher: wait until the parent assigns our Job Object."""

import subprocess
import sys


def main():
    if sys.stdin.buffer.read(1) != b"1":
        return 1
    try:
        return subprocess.call(sys.argv[1:], stdin=subprocess.DEVNULL, shell=False)
    except OSError as exc:
        print(f"Could not start command: {exc}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
