"""Put the scaffold's scripts dir on sys.path so tests can import mlhelpers."""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
