import os
import re
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from preprocessing import *
from RAG_utils import *

# ----------------------------
# 6) Local HF LLM wrapper
# ----------------------------
class HFLocalLLM:
    """
    Minimal wrapper so switching HF models is easy.
    On T4: use_4bit=True is usually safest.
    """
    def __init__(self, model_name: str, use_4bit: bool = True):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        kwargs = {"device_map": "auto"}
        if use_4bit:
            kwargs["load_in_4bit"] = True
        else:
            kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nASSISTANT:\n"

    @torch.inference_mode()
    def generate(self, system: str, user: str, max_new_tokens: int = 250, temperature: float = 0.2, top_p: float = 0.95):
        prompt = self._format(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        in_tokens = int(inputs["input_ids"].shape[1])

        out_ids = self.model.generate(
            **inputs,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        out_tokens = int(out_ids.shape[1] - in_tokens)
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[-1].strip()

        return {"text": text, "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens, "total_tokens": in_tokens + out_tokens}}
