import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

LOCAL_PYCACHE = os.path.join(PROJECT_ROOT, ".local", "pycache")
os.makedirs(LOCAL_PYCACHE, exist_ok=True)
sys.pycache_prefix = LOCAL_PYCACHE
