"""LoRA SFT trainer for CWM-32B on 4x A6000 (no NVLink).

Faithful path (per §14): LoRA on CWM directly. Custom training loop with
device_map="auto" (naive pipeline across the 4 GPUs — fits the 64GB bf16 base
weights), gradient checkpointing, LoRA on the standard Llama-style linear modules.
Trains on tokenized {input_ids, labels} from build_sft_data.py.

Saves the adapter for vLLM eval (vLLM supports LoRA serving).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


def load_data(path):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["input_ids"], d["labels"]))
    return rows


def collate(batch, pad_id):
    maxlen = max(len(x[0]) for x in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        padn = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * padn)
        labels.append(lab + [-100] * padn)
        attn.append([1] * len(ids) + [0] * padn)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--data", default="data/sft_oop_expanded.jsonl")
    ap.add_argument("--out", default="adapters/cwm_oop_expanded")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--init_adapter", default=None, help="resume/continue training from an existing adapter")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model_path)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    print("loading CWM-32B (device_map=auto across 4 GPUs)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    if a.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
        print(f"  resumed from adapter {a.init_adapter}", flush=True)
    else:
        lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=a.r, lora_alpha=2 * a.r,
                          lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.enable_input_require_grads()  # needed for grad ckpt + frozen base

    data = load_data(a.data)
    print(f"data: {len(data)} examples", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    total_steps = min(a.max_steps, int(len(data) * a.epochs / (a.bs * a.grad_accum)))

    def lr_at(step):
        if step < a.warmup:
            return a.lr * step / max(1, a.warmup)
        prog = (step - a.warmup) / max(1, total_steps - a.warmup)
        return a.lr * 0.5 * (1 + math.cos(math.pi * prog))

    emb_device = model.get_input_embeddings().weight.device
    model.train()
    step, micro, running = 0, 0, 0.0
    t0 = time.time()
    order = list(range(len(data)))
    random.shuffle(order)
    idx = 0
    while step < total_steps:
        batch = [data[order[(idx + j) % len(data)]] for j in range(a.bs)]
        idx += a.bs
        input_ids, labels, attn = collate(batch, pad_id)
        input_ids, labels, attn = input_ids.to(emb_device), labels.to(emb_device), attn.to(emb_device)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        loss = out.loss / a.grad_accum
        loss.backward()
        running += out.loss.item()
        micro += 1
        if micro % a.grad_accum == 0:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            step += 1
            if step % 5 == 0 or step == 1:
                avg = running / (a.grad_accum * (5 if step % 5 == 0 else 1))
                print(f"  step {step}/{total_steps} loss={avg:.4f} lr={lr_at(step):.2e} "
                      f"({(time.time()-t0)/step:.1f}s/step)", flush=True)
                running = 0.0

    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved adapter -> {a.out} (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
