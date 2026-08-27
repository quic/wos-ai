# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Config Manager - loads models.yaml with environment variable expansion.
"""

import os
import re
import yaml
from typing import Dict, List, Any
from utils.logger import get_logger

logger = get_logger(__name__)

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} in strings. Useful in linux system"""
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(i) for i in value]
    return value


class ConfigManager:
    def __init__(self, config_path: str = "config/models.yaml"):
        self._config_path = config_path
        self._config: Dict = {}

    async def load_config(self) -> None:
        with open(self._config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._config = _expand_env(raw or {})
        logger.info(f"Config loaded from {self._config_path} "
                    f"({len(self._config.get('models', []))} models)")

    def get_config(self) -> Dict:
        return self._config

    def get_models(self) -> List[Dict]:
        return self._config.get("models", [])