import os
import sys

# Clear any existing paths - keep only system paths
sys.path = [p for p in sys.path if "site-packages" in p or "python3.10" in p]

# Add only the project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

print("\nCLEANED PATH:")
print("\n".join(sys.path))

print("\nProject root:", project_root)

# Try creating __init__.py files if they don't exist
if not os.path.exists("datasets/__init__.py"):
    print("\nCreating datasets/__init__.py")
    open("datasets/__init__.py", "a").close()

try:
    print("\nTrying direct import:")
    from datasets.fisher import FisherDataset

    print("Success!")
except ImportError as e:
    print("Failed:", str(e))
    print("Current working directory:", os.getcwd())
    print("Contents of datasets:", os.listdir("datasets"))

print("\nFINAL PATH:")
print("\n".join(sys.path))
print("Cleaned Python path:", sys.path)


# Try different import patterns
try:
    print("\nTrying absolute import:")
    from mod_add.datasets.fisher import FisherDataset
except ImportError as e:
    print("Failed:", str(e))

try:
    print("\nTrying relative import:")
    from .datasets.fisher import FisherDataset
except ImportError as e:
    print("Failed:", str(e))

try:
    print("\nTrying direct import:")
    from datasets.fisher import FisherDataset
except ImportError as e:
    print("Failed:", str(e))

# Also test models import
try:
    print("\nTrying models import:")
    from models.constructor import Model

    print("Models import worked!")
except ImportError as e:
    print("Failed:", str(e))

print("\nPython path:", sys.path)
