"""Utility to reload all modified Python modules in current directory."""

from typing import Optional
import importlib
import os
import sys
import types


def reload_all() -> None:
    """
    Reload all Python modules in current working directory.

    Returns:
        None: Modules are reloaded in-place
    """
    base_dir: str = os.getcwd()  # current working directory of the notebook
    print(f"Reloading modules in directory: {base_dir}")

    for mod_name, mod in list(sys.modules.items()):
        if not isinstance(mod, types.ModuleType):
            continue

        mod_file: Optional[str] = getattr(mod, "__file__", None)
        if not mod_file:
            continue

        mod_file = os.path.abspath(mod_file)
        mod_dir: str = os.path.dirname(mod_file)

        # Only reload modules in the notebook's directory (no subdirs)
        if mod_dir == base_dir:
            # Check if file actually exists before reloading
            if os.path.exists(mod_file):
                importlib.reload(mod)
