"""check_setup.py - verify Python version, libraries and dataset files."""
import sys
from importlib import import_module
from pathlib import Path

print("Python version:", sys.version)
base = Path(__file__).parent
print("Project folder:", base.resolve())
print("-" * 50)

for name in ["Fake.csv", "True.csv"]:
    p = base / "dataset" / name
    if p.exists():
        print(f"[OK] Found {name} ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"[MISSING] {p}")

print("[OK] models folder" if (base / "models").exists() else "[MISSING] models folder")

for lib in ["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "streamlit", "joblib"]:
    try:
        import_module(lib)
        print(f"[OK] {lib} installed")
    except ImportError:
        print(f"[MISSING] {lib} -> run: pip install -r requirements.txt")
