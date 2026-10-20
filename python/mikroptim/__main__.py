"""``python -m mikroptim <befehl> ...``: die CLI (docs/cli.md) aus dem Python-Paket.

Die Ausgabe erscheint am Ende des Befehls (kein fortlaufendes Iterationsprotokoll) und wird wie
vom Programm ``mikroptim`` als UTF-8 geschrieben.
"""

import sys

from . import run_cli


def _write(stream, text):
    stream.flush()
    stream.buffer.write(text.encode("utf-8"))
    stream.buffer.flush()


def main(argv=None):
    code, out, err = run_cli(sys.argv[1:] if argv is None else argv)
    _write(sys.stdout, out)
    _write(sys.stderr, err)
    return code


if __name__ == "__main__":
    sys.exit(main())
