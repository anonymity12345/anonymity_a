"""Optional vLLM-Omni backend. Importing this package does not import CUDA."""

VLLM_VERSION = "0.14.0"
OMNI_COMMIT = "ed89c8b0436999e9210f11363f4eb512330a9dfa"


def register():
    """Entry point loaded in both the orchestrator and spawned engine workers."""
    from vllm.model_executor.models import ModelRegistry
    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    for registry in (ModelRegistry, OmniModelRegistry):
        for stage in ("Thinker", "Talker"):
            registry.register_model(
                f"ExOmni{stage}",
                f"ex_omni.vllm_omni_backend.models:ExOmni{stage}",
            )


def check_runtime():
    from importlib.metadata import version

    actual = version("vllm").split("+")[0]
    if actual != VLLM_VERSION:
        raise RuntimeError(f"Requires vllm=={VLLM_VERSION}; found {actual}")
    actual = version("vllm-omni")
    if actual not in ("0.14.0", "0.14.0rc1"):
        raise RuntimeError(f"Requires vLLM-Omni v0.14.0 source; found {actual}")
    register()
