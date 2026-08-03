import importlib
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_declared_cli_entrypoints_resolve_to_callables() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)
    scripts = project["project"]["scripts"]

    for script_name, target in scripts.items():
        module_name, callable_name = target.split(":", maxsplit=1)
        module = importlib.import_module(module_name)

        assert callable(getattr(module, callable_name)), script_name
