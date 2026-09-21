import importlib
import pkgutil

from app.core import plugins


def load_plugins() -> None:
    for module in pkgutil.iter_modules(
        plugins.__path__, prefix=plugins.__name__ + "."
    ):
        importlib.import_module(module.name)
