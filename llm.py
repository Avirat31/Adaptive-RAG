"""LLM backends. vLLM with tensor parallelism is the default multi-GPU path; the HF backend
shards the model across GPUs with device_map='auto' (slower, pipeline-style)."""
from __future__ import annotations

import os

from .config import Config
from .gpu_utils import get_gpu_ids
from .utils import chunks


class LLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        gpus = get_gpu_ids(cfg.num_gpus)
        self.backend = cfg.llm_backend
        if self.backend == "vllm":
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            from vllm import LLM as VLLM
            tp = cfg.tp_size or max(1, len(gpus))
            self.engine = VLLM(
                model=cfg.llm_model, tensor_parallel_size=tp, dtype=cfg.dtype,
                gpu_memory_utilization=cfg.gpu_mem_util, max_model_len=cfg.max_model_len,
                enable_prefix_caching=True, trust_remote_code=True)
        elif self.backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(cfg.llm_model, padding_side="left", trust_remote_code=True)
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.llm_model, torch_dtype=getattr(torch, cfg.dtype), device_map="auto",
                trust_remote_code=True).eval()
        else:
            raise ValueError(f"unknown llm_backend {self.backend}")

    def chat(self, conversations: list, max_tokens: int = 384, temperature: float = 0.0) -> list:
        """conversations: list of [{'role','content'}, ...]. Returns one string per conversation."""
        if not conversations:
            return []
        if self.backend == "vllm":
            from vllm import SamplingParams
            sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            outs = self.engine.chat(conversations, sp, use_tqdm=False)
            return [o.outputs[0].text.strip() for o in outs]
        return self._hf_chat(conversations, max_tokens, temperature)

    def _hf_chat(self, conversations, max_tokens, temperature, bs: int = 16):
        import torch
        results = []
        for batch in chunks(conversations, bs):
            prompts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in batch]
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.cfg.max_model_len - max_tokens).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=temperature > 0,
                                          temperature=temperature if temperature > 0 else None,
                                          pad_token_id=self.tok.pad_token_id)
            results.extend(t.strip() for t in self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                                                    skip_special_tokens=True))
        return results
