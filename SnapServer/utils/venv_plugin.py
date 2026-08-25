import os
import site
import sys
from utils.inference_plugin import InferencePlugin
from utils.logger import get_logger

logger = get_logger(__name__)


class VenvPlugin(InferencePlugin):
    """
    Base class for plugins that need a separate virtual environment.

    HOW IT WORKS
    ------------
    The server calls load() on your plugin. VenvPlugin intercepts this via __init_subclass__ to inject path setup BEFORE your load() runs.
    Your load() method is called normally.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Only wrap if the subclass defines its own load()
        if "load" in cls.__dict__:
            original_load = cls.__dict__["load"]

            def _load_with_venv(self, model_id: str, config: dict) -> None:
                # Step 1: Set up venv paths (transparent to the user)
                self._setup_paths(config)
                # Step 2: Call the user's original load()
                original_load(self, model_id, config)

            cls.load = _load_with_venv



    # ── Internal path setup — do NOT override ────────────────────────────────

    def _setup_paths(self, config: dict) -> None:
        """
        Add source_dir and its venv site-packages to sys.path.
        Called automatically before setup().
        """
        source_dir = config.get("source_dir")

        if not source_dir:
            return  # No source_dir — plugin uses server's own dependencies

        # 1. Add source directory to sys.path
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)

        # 2. Find and add venv site-packages
        venv_path = self._find_venv(config, source_dir)
        if venv_path:
            site_packages = self._get_site_packages(venv_path)
            if site_packages and os.path.exists(site_packages):
                if site_packages not in sys.path:
                    sys.path.insert(0, site_packages)
                    site.addsitedir(site_packages)
                logger.info(
                    f"[{type(self).__name__}] Using venv: {site_packages}"
                )
            else:
                logger.warning(
                    f"[{type(self).__name__}] Venv found at '{venv_path}' "
                    "but site-packages not found. Using server's dependencies."
                )
        else:
            logger.info(
                f"[{type(self).__name__}] No venv found in '{source_dir}'. "
                "Using server's dependencies."
            )

    def _find_venv(self, config: dict, source_dir: str) -> str:
        """
        Find the venv path. Priority:
          1. Explicit 'venv_path' in config
          2. Explicit 'venv_path' in config (backward compat)
          3. Auto-detect: {source_dir}/.venv, {source_dir}/venv, {source_dir}/env
        """
        # Explicit path from config
        venv_path = config.get("venv_path")
        if venv_path:
            return venv_path

        # Auto-detect common venv names
        for venv_name in (".venv", "venv", "env"):
            candidate = os.path.join(source_dir, venv_name)
            if os.path.isdir(candidate):
                return candidate

        return None

    @staticmethod
    def _get_site_packages(venv_path: str) -> str:
        """
        Get the site-packages path for a venv.
        Handles both Windows (Lib/site-packages) and Linux (lib/pythonX.Y/site-packages).
        """
        # Windows
        win_path = os.path.join(venv_path, "Lib", "site-packages")
        if os.path.exists(win_path):
            return win_path

        # Linux/Mac — find lib/pythonX.Y/site-packages
        lib_dir = os.path.join(venv_path, "lib")
        if os.path.isdir(lib_dir):
            for entry in os.listdir(lib_dir):
                if entry.startswith("python"):
                    sp = os.path.join(lib_dir, entry, "site-packages")
                    if os.path.exists(sp):
                        return sp

        return None