"""Put src/ on sys.path so tests can import the modules by their plain names.

The scripts use flat imports (`from config import load_config`) because Python
adds a script's own directory to sys.path when you run it directly. pytest does
not, so this file does it explicitly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
