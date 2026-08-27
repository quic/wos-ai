# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Plugin Template
=================
Replace "MyPlugin" with your class name and fill in the sections marked ← FILL IN.

Sample example on how to Add data in models.yaml
  - id: my-model
    backend: plugin
    plugin_path: examples/my_plugin.py
    plugin_class: MyPlugin
    source_dir: C:/path/to/my/app    # where your app code lives
    # ... your model-specific config keys


TWO WAYS TO HANDLE VENV/PATHS
------------------------------
  Option A (Recommended — simple function call):
    from utils.venv_plugin import setup_venv_paths

    class MyPlugin:
        def load(self, model_id, config):
            setup_venv_paths(config)   # ← one line, done!
            from my_app import ...     

  Option B (Class inheritance — automatic, zero lines):
    from utils.venv_plugin import VenvPlugin

    class MyPlugin(VenvPlugin):
        def load(self, model_id, config):
            from my_app import ...     

SUPPORTED CAPABILITIES (implement these 3 methods)
-------------------------------------------------
  load(model_id, config)            → initialize your model # Mandatory

  # Any 1 of the API's, examples:
  transcribe(audio_bytes, **kwargs) → dict   POST /v1/audio/transcriptions
  translate(audio_bytes, **kwargs)  → dict   POST /v1/audio/translations
  generate(prompt, ...)             → iter   POST /v1/chat/completions
  embed(input, **kwargs)            → list   POST /v1/embeddings
  synthesize(text, ...)             → bytes  POST /v1/audio/speech
  moderate(input, **kwargs)         → dict   POST /v1/moderations
  image_generate(prompt, ...)       → list   POST /v1/images/generations

  unload()                          → free resources # Mandatory
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# OPTION A: Simple function call (recommended)
from utils.venv_plugin import setup_venv_paths

# OPTION B: Class inheritance (zero lines in load)
# from utils.venv_plugin import VenvPlugin


# OPTION A: Plain class, call setup_venv_paths() in load() 
class MyPlugin:  # ← FILL IN: rename to your class name
    """
    ← FILL IN: describe what this plugin does.
    """

    # REQUIRED: Initialize your model 

    def load(self, model_id: str, config: dict) -> None:
        """Called ONCE when the model loads."""
        setup_venv_paths(config)   # ← sets up source_dir + venv paths

        # ← FILL IN: import your app and initialize your model
        # from my_app import MyPipeline
        # self.pipeline = MyPipeline(
        #     model_path=config["model_path"],
        # )
        raise NotImplementedError("Implement load()")

    # REQUIRED: Implement the capability your model supports 
    # Delete the methods you don't need. Keep only what your model does.

    # Example 1: Audio Transcription (for ASR models) 
    def transcribe(self, audio_bytes: bytes, **kwargs) -> dict:
        """
        Called for POST /v1/audio/transcriptions.
        Returns: {"text": "transcribed text"}
        """
        # ← FILL IN:
        # import soundfile as sf, io
        # audio, sr = sf.read(io.BytesIO(audio_bytes))
        # return {"text": self.pipeline.transcribe(audio, sr)}
        raise NotImplementedError("Implement transcribe()")

    # Example 2: Text Generation (for LLM) 
    def generate(self, prompt: str, max_tokens: int = 512,
                 temperature: float = 1.0, stop=None):
        """
        Called for POST /v1/chat/completions.
        Must be a generator — yield tokens one by one.
        """
        # ← FILL IN:
        # for token in self.pipeline.generate(prompt, max_tokens):
        #     yield token
        raise NotImplementedError("Implement generate()")
        yield  # makes this a generator

    # Example 3: Embeddings models
    def embed(self, input: list, **kwargs) -> list:
        """
        Called for POST /v1/embeddings.
        Returns: list of embedding vectors [[0.1, 0.2, ...], ...]
        """
        # ← FILL IN:
        # return [self.pipeline.embed(text) for text in input]
        raise NotImplementedError("Implement embed()")


    # REQUIRED: Clean up resources
    def unload(self) -> None:
        """Called when the model is unloaded. Free your resources here."""
        # ← FILL IN (optional):
        # if hasattr(self, "pipeline"):
        #     del self.pipeline
        pass
