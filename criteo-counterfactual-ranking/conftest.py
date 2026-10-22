"""Ensures the project root (containing parser.py, features.py, policy.py,
estimators.py, train.py) is importable from tests/ regardless of pytest's
rootdir/import-mode, since this project has no package __init__.py files."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
