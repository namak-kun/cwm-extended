"""vLLM-backed dynamics model (high throughput; tensor-parallel for big models).
Same interface as models.llm.LLMModel. Run only inside the .venv_vllm venv.
"""
from __future__ import annotations

from models.prompting import SYS, build_prompt, extract_json


class VLLMModel:
    def __init__(self, model_id: str, tensor_parallel: int = 1,
                 max_new_tokens: int = 192, max_model_len: int = 6144):
        from vllm import LLM, SamplingParams
        self._SP = SamplingParams
        self.name = "vllm:" + model_id.split("/")[-1]
        self.max_new_tokens = max_new_tokens
        self.llm = LLM(model=model_id, tensor_parallel_size=tensor_parallel,
                       dtype="bfloat16", gpu_memory_utilization=0.90,
                       max_model_len=max_model_len, enforce_eager=False)
        self.tok = self.llm.get_tokenizer()
        self.sp_greedy = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def _chat(self, user: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

    def predict_requests(self, requests: list[dict]):
        prompts = [self._chat(build_prompt(r["code"], r["history"], r["action"], r.get("rng_log")))
                   for r in requests]
        outs = self.llm.generate(prompts, self.sp_greedy, use_tqdm=False)
        return [extract_json(o.outputs[0].text) for o in outs]

    def predict(self, code, history, action, rng_log=None):
        return self.predict_requests(
            [{"code": code, "history": history, "action": action, "rng_log": rng_log}])[0]

    def sample_batch(self, users: list[str], k: int = 6, temperature: float = 0.9):
        sp = self._SP(temperature=temperature, top_p=0.95, max_tokens=self.max_new_tokens, n=k)
        prompts = [self._chat(u) for u in users]
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        return [[c.text for c in o.outputs] for o in outs]
