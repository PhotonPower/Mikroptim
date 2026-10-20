"""Startet die pytest-Suite für ctest (python/CMakeLists.txt).

Exit-Code 77 (ctest: übersprungen), wenn pytest oder NumPy nicht installiert sind; sonst der
Exit-Code von pytest.
"""

import sys

try:
    import numpy  # noqa: F401
    import pytest
except ImportError as e:
    print(f"übersprungen: {e}")
    sys.exit(77)

sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
