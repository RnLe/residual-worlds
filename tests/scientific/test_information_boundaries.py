"""Static information-boundary checks.

The target world is privileged simulation truth. Model, training, and
planning code must never import it -- only the simulator/evaluation
side may. This test parses the actual import statements, so a violation
cannot hide behind a convenience re-export.
"""

import ast
from pathlib import Path

import pytest

from residual_worlds.paths import repository_root

pytestmark = pytest.mark.scientific

FORBIDDEN_MODULE = "residual_worlds.physics.target"
RESTRICTED_PACKAGES = ("models", "training", "planning")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
            imports.update(f"{node.module}.{alias.name}" for alias in node.names)
    return imports


def test_learned_and_planning_code_cannot_import_target_physics() -> None:
    package_root = repository_root() / "src" / "residual_worlds"
    violations: list[str] = []
    for package in RESTRICTED_PACKAGES:
        directory = package_root / package
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            for module in _imported_modules(path):
                if module.startswith(FORBIDDEN_MODULE) or module.endswith("physics.target"):
                    violations.append(f"{path.relative_to(package_root)} imports {module}")
    assert violations == [], "\n".join(violations)


def test_target_module_exists_where_expected() -> None:
    # Guard against the boundary test silently passing because the file moved.
    assert (repository_root() / "src" / "residual_worlds" / "physics" / "target.py").exists()
