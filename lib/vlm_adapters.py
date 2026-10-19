"""Thin adapters that give remote-code VLMs the interface the evaluator (exp/s582_evalrepair.run) and S747 use from exp/uni.Uni:
IMG, chat(), encode(), tok, model, fam, forward_extra.  Kimi-VL keeps its own template; ERNIE-4.5-VL drops the "Picture n:" labels its
template adds before each image (so that no model reads the index from text) and skips reasoning ("<think>\\n\\n</think>\\n\\n")."""
import importlib, subprocess, sys
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM


def _ensure(pkgs):
    """Remote code may import packages the image lacks (Kimi: tiktoken); install them into the running container."""
    for p in pkgs:
        try:
            importlib.import_module(p)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", p], check=True)


class Adapter:
    def __init__(self, name, device_map="auto"):
        n = name.lower()
        self.fam = "kimi" if "kimi" in n else ("ernie" if "ernie" in n else None)
        assert self.fam, name
        self.name = name
        _ensure(["tiktoken", "blobfile"] if self.fam == "kimi" else [])
        self.processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        self.tok = self.processor.tokenizer
        kw = {}
        if self.fam == "ernie":
            # ERNIE's VL config sets moe_use_hard_gate=True and then calls its parent's __init__, which resets it to False and leaves the
            # text experts without a gate (MOELayer fails on gate None); restore the VL default
            cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
            cfg.moe_use_hard_gate = True
            kw["config"] = cfg
        self.model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device_map, **kw).eval()
        if self.fam == "ernie":                              # the model card's two setup calls: no labels, and uint8 patches normalised inside
            self.processor.eval()
            self.model.add_image_preprocess(self.processor)
        self.forward_extra = {}
        self.IMG = ("<|media_start|>image<|media_content|><|media_pad|><|media_end|>" if self.fam == "kimi"
                    else "<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>")

    @property
    def device(self):
        return next(self.model.parameters()).device

    def chat(self, turns, add_generation_prompt=True, prefix_assistant=""):
        if self.fam == "kimi":
            s = "<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|>"
            for role, c in turns:
                s += ("<|im_user|>user" if role == "user" else "<|im_assistant|>assistant") + "<|im_middle|>" + c + "<|im_end|>"
            if add_generation_prompt:
                s += "<|im_assistant|>assistant<|im_middle|>" + prefix_assistant
            return s
        s = "<|begin_of_sentence|>"
        for role, c in turns:
            s += ("User: " if role == "user" else "\nAssistant: ") + c
        if add_generation_prompt:
            s += "\nAssistant: <think>\n\n</think>\n\n" + prefix_assistant
        return s

    def encode(self, text, images):
        inp = self.processor(text=[text], images=images if images else None, return_tensors="pt")
        inp = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        if self.fam == "ernie" and inp.get("token_type_ids") is not None:
            # ERNIE's forward wants one token type beyond the input (the next token's); its own generate appends a zero, as here
            t = inp["token_type_ids"]
            inp["token_type_ids"] = torch.cat([t, torch.zeros_like(t[:, -1:])], -1)
        return inp
