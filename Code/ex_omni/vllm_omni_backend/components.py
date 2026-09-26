"""Load shared modules without importing the generation loop."""

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(None)
def source_module(relative_path, name):
    path = Path(__file__).resolve().parents[1] / relative_path
    qualified = f"ex_omni.vllm_omni_backend.{name}"
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(qualified, None)
        raise
    return module


def speech_components():
    return source_module("model/speech_generator/speech_generator.py", "_speech")


def blendshape_class():
    source_module("model/blendshape_generator/blendshape_utils.py", "blendshape_utils")
    return source_module(
        "model/blendshape_generator/blendshape_generator.py", "_blendshape"
    ).BlendshapeGenerator
