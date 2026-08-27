# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Model Registry - maps model IDs to their config dicts.
"""

from typing import Dict, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    def __init__(self, config_manager):
        self._config_manager = config_manager
        self._models: Dict[str, Dict] = {}

    async def initialize(self) -> None:
        for model in self._config_manager.get_models():
            mid = model.get("id")
            if mid:
                self._models[mid] = model
        logger.info(f"ModelRegistry: {len(self._models)} models registered")

    async def list_models(self) -> List[Dict]:
        return list(self._models.values())

    async def get_model(self, model_id: str) -> Optional[Dict]:
        return self._models.get(model_id)

    async def close(self) -> None:
        self._models.clear()