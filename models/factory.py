"""Backend factory: lazily build an HF-transformers or vLLM dynamics model."""
from __future__ import annotations


def get_model(model_id: str, backend: str = "hf", tp: int = 1):
    if backend == "vllm":
        from models.vllm_model import VLLMModel
        return VLLMModel(model_id, tensor_parallel=tp)
    from models.llm import LLMModel
    return LLMModel(model_id)
