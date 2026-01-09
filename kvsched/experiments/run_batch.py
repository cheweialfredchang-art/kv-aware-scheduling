from __future__ import annotations

# Backward-compatible module entrypoint:
#   python -m kvsched.experiments.run_batch ...
# Internally delegates to: python -m kvsched.cli batch ...

from kvsched.cli import cmd_batch, main as cli_main


def main():
    # If you call this module directly, we still support the legacy flags.
    # We simply route to the CLI 'batch' subcommand for a single source of truth.
    import sys
    # Insert 'batch' as the subcommand.
    argv = ["batch", *sys.argv[1:]]
    raise SystemExit(cli_main(argv))


if __name__ == "__main__":
    main()
