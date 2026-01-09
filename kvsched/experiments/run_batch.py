from __future__ import annotations

"""Backward-compatible entrypoint.

Allows:
  python -m kvsched.experiments.run_batch ...

Internally delegates to:
  python -m kvsched.cli batch ...
"""

from kvsched.cli import main as cli_main


def main() -> None:
    import sys
    argv = ["batch", *sys.argv[1:]]
    raise SystemExit(cli_main(argv))


if __name__ == "__main__":
    main()
