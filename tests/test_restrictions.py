"""
Static analysis test ensuring 100% compliance with assignment restrictions.
Ensures no torch.optim, tensorflow.keras.optimizers, keras.optimizers, or autograd are used anywhere.
"""
import os
import re

PROHIBITED_PATTERNS = [
    r"torch\.optim",
    r"tensorflow\.keras\.optimizers",
    r"keras\.optimizers",
    r"import\s+torch",
    r"import\s+tensorflow",
    r"from\s+torch\b",
    r"from\s+tensorflow\b",
    r"import\s+jax",
    r"from\s+jax\b",
    r"import\s+autograd",
    r"from\s+autograd\b",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_prohibited_libraries_used():
    """Scan all Python files in the repository for prohibited optimizer imports."""
    scanned_files = []
    violations = []

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip virtualenvs and git directory
        if any(ignored in root for ignored in [".git", "venv", ".venv", "__pycache__", ".pytest_cache"]):
            continue
        for f in files:
            if f.endswith(".py") and f != "test_restrictions.py":
                file_path = os.path.join(root, f)
                scanned_files.append(file_path)
                with open(file_path, "r", encoding="utf-8") as py_file:
                    content = py_file.read()
                    for pattern in PROHIBITED_PATTERNS:
                        matches = re.findall(pattern, content)
                        if matches:
                            violations.append(f"Violation in {file_path}: found prohibited pattern '{pattern}'")

    assert len(violations) == 0, f"Found prohibited imports/libraries:\n" + "\n".join(violations)
    assert len(scanned_files) >= 5, f"Expected at least 5 source files scanned, found {len(scanned_files)}"
