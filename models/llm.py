"""LLM-backed dynamics model (transformers). A stand-in for CWM: a strong open
code LLM prompted to act as a program interpreter that predicts the next symbolic
state. Greedy decoding for determinism; robust JSON extraction.
"""
from __future__ import annotations

import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.base import DynamicsModel
from models.prompting import SYS, build_prompt, extract_json  # re-exported for callers


class LLMModel(DynamicsModel):
    def __init__(self, model_id: str, dtype=torch.bfloat16, device="cuda:0",
                 max_new_tokens: int = 320):
        self.name = "llm:" + model_id.split("/")[-1]
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    def _chat(self, user: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate_batch(self, users: list[str]) -> list[str]:
        prompts = [self._chat(u) for u in users]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                  do_sample=False, pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)

    @torch.no_grad()
    def sample_batch(self, users: list[str], k: int = 5, temperature: float = 0.8):
        """Return k sampled completions per user (for distributional / support eval)."""
        prompts = [self._chat(u) for u in users]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                  do_sample=True, temperature=temperature, top_p=0.95,
                                  num_return_sequences=k, pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(gen, skip_special_tokens=True)
        return [texts[i * k:(i + 1) * k] for i in range(len(users))]

    def predict(self, code, history, action, rng_log=None):
        txt = self.generate_batch([build_prompt(code, history, action, rng_log)])[0]
        return extract_json(txt)

    def predict_requests(self, requests: list[dict]) -> list[Any]:
        """Batched. Each request: {code, history, action, rng_log}."""
        users = [build_prompt(r["code"], r["history"], r["action"], r.get("rng_log"))
                 for r in requests]
        outs = []
        B = 16
        for i in range(0, len(users), B):
            for txt in self.generate_batch(users[i:i + B]):
                outs.append(extract_json(txt))
        return outs
