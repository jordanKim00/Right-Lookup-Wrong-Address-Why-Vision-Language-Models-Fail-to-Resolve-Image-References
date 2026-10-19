"""One harness for every family, so the same experiment runs unchanged across models.

The families differ in three places that matter to us and nowhere else that we touch:
the placeholder string used to mark an image in the prompt, the chat template, and where
the language model's decoder layers live in the module tree.  Everything downstream --
attention knockout, layer-wise readout, restricted-label scoring -- is written against
this interface, so adding a family means adding a row to SPEC, not rewriting experiments.

Deliberately NOT abstracted: position-id surgery.  M-RoPE exists only in the Qwen family,
so anything that rewrites visual coordinates stays in vlmkit and is reported as a
Qwen-specific result rather than silently faked elsewhere.
"""
import inspect
import torch, re
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig

SPEC = {
    "qwen2vl":    dict(tag="<|vision_start|><|image_pad|><|vision_end|>", tmpl="qwen"),
    "qwen25vl":   dict(tag="<|vision_start|><|image_pad|><|vision_end|>", tmpl="qwen"),
    "qwen3vl":    dict(tag="<|vision_start|><|image_pad|><|vision_end|>", tmpl="qwen"),
    # Qwen3.5 is Qwen3_5ForConditionalGeneration with a vision_config and the same
    # vision_start/image_pad/vision_end triple, so it takes the qwen chat template too
    "qwen35":     dict(tag="<|vision_start|><|image_pad|><|vision_end|>", tmpl="qwen"),
    "llava":      dict(tag="<image>", tmpl="llava"),
    "llava_next": dict(tag="<image>", tmpl="qwen"),
    "llava_ov":   dict(tag="<image>", tmpl="qwen"),
    "internvl":   dict(tag="<IMG_CONTEXT>", tmpl="internvl"),
    "idefics2":   dict(tag="<image>", tmpl="idefics"),
}


def family(name):
    n = name.lower()
    if "internvl" in n: return "internvl"
    if "idefics" in n: return "idefics2"
    if "onevision" in n: return "llava_ov"
    if "interleave" in n: return "llava_next"
    if "llava" in n: return "llava"
    if "qwen3-vl" in n: return "qwen3vl"
    if "qwen3.5" in n or "qwen3_5" in n: return "qwen35"
    if "qwen2.5-vl" in n: return "qwen25vl"
    if "qwen2-vl" in n: return "qwen2vl"
    raise ValueError(f"unknown family for {name}")


class Uni:
    def __init__(self, name, device="cuda:0", attn="sdpa", dtype=torch.bfloat16,
                 device_map=None, max_memory=None):
        """device_map spreads the layers over several cards.  Needed when a neighbouring
        job holds ~8 GB on every GPU and a 7B model's weights no longer fit on one: split
        four ways the weights are about 4 GB per card, and because output_attentions keeps
        one T x T tensor per layer, those follow their layers and spread too."""
        self.name, self.fam = name, family(name)
        self.spec = SPEC[self.fam]
        self.processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        if device_map:
            self.model = AutoModelForImageTextToText.from_pretrained(
                name, dtype=dtype, attn_implementation=attn,
                trust_remote_code=True, device_map=device_map,
                **({"max_memory": max_memory} if max_memory else {})).eval()   # e.g. 27B on two cards + CPU offload (S626)
            device = str(next(self.model.parameters()).device)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                name, dtype=dtype, attn_implementation=attn,
                trust_remote_code=True).to(device).eval()
        self.device, self.tok = device, self.processor.tokenizer
        self.cfg = self.model.config
        tcfg = getattr(self.cfg, "text_config", self.cfg)
        self.n_layer = tcfg.num_hidden_layers
        self.n_head = tcfg.num_attention_heads
        self.IMG = self.spec["tag"]
        self.image_token_id = self._img_id()
        self.layers = self._layers()
        # Newer HF multimodal decoders otherwise project every (often thousands of)
        # sequence positions through the full vocabulary head although all experiments
        # score only the next token.  OneVision at five images allocated an avoidable
        # 5.25 GiB logits tensor.  Detect the API instead of assuming every family/version
        # names this optimization identically.
        self.forward_extra = ({"logits_to_keep": 1}
                              if "logits_to_keep" in inspect.signature(self.model.forward).parameters
                              else {})
        self.sys = "You are a helpful assistant."
        self.fast_ok = True

    # ---------- structure discovery ----------
    def _img_id(self):
        for k in ("image_token_id", "image_token_index"):
            v = getattr(self.cfg, k, None)
            if isinstance(v, int): return v
        for t in ("<|image_pad|>", "<image>", "<IMG_CONTEXT>"):
            i = self.tok.convert_tokens_to_ids(t)
            if i is not None and i != self.tok.unk_token_id: return i
        raise ValueError("no image token id")

    def _layers(self):
        """The decoder-layer ModuleList, wherever the family happens to put it."""
        for path in ("model.language_model.layers", "model.language_model.model.layers",
                     "language_model.model.layers", "language_model.layers",
                     "model.text_model.layers", "model.model.text_model.layers",
                     "text_model.layers", "model.model.layers", "model.layers"):
            o = self.model
            try:
                for p in path.split("."): o = getattr(o, p)
                if isinstance(o, torch.nn.ModuleList) and len(o) == self.n_layer:
                    self.layer_path = path; return o
            except AttributeError:
                continue
        # fall back to a search, so a new family costs a name in SPEC and nothing else
        for n, m in self.model.named_modules():
            if isinstance(m, torch.nn.ModuleList) and len(m) == self.n_layer:
                self.layer_path = n; return m
        raise ValueError(f"decoder layers not found in {self.name}")

    # ---------- prompting ----------
    def chat(self, turns, add_generation_prompt=True, prefix_assistant=""):
        t = self.spec["tmpl"]
        if t in ("qwen", "internvl"):
            s = f"<|im_start|>system\n{self.sys}<|im_end|>\n"
            for role, c in turns: s += f"<|im_start|>{role}\n{c}<|im_end|>\n"
            if add_generation_prompt: s += "<|im_start|>assistant\n" + prefix_assistant
            return s
        if t == "idefics":
            s = ""
            for role, c in turns:
                s += ("User: " if role == "user" else "Assistant: ") + c + "<end_of_utterance>\n"
            if add_generation_prompt: s += "Assistant: " + prefix_assistant
            return s
        s = ""
        for role, c in turns:
            s += ("USER: " if role == "user" else "ASSISTANT: ") + c + \
                 ("" if role == "user" else "</s>")
        if add_generation_prompt: s += " ASSISTANT: " + prefix_assistant
        return s

    def encode(self, text, images):
        kw = dict(text=[text], return_tensors="pt", padding=False)
        if images: kw["images"] = images
        inp = self.processor(**kw)
        self._last_counts = None                                                   # per-image token counts (LLaVA anyres: sizes differ)
        if images and "image_sizes" in inp and hasattr(self.processor, "_get_number_of_features"):
            try:
                ip = self.processor.image_processor
                H = ip.size.get("height", ip.size.get("shortest_edge")); W = ip.size.get("width", ip.size.get("shortest_edge"))
                self._last_counts = [int(self.processor._get_number_of_features(int(h), int(w), H, W)) for h, w in inp["image_sizes"].tolist()]
            except Exception:
                self._last_counts = None
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inp.items()}

    # ---------- cached vision path ----------
    # The processor re-runs the vision preprocessor on every call, so asking eight
    # questions about the same eight images preprocessed them sixty-four times.  The
    # visual tensors depend only on the images, so they are computed once per trial and
    # the text is expanded by hand: the placeholder's image token is simply repeated as
    # many times as the processor itself repeated it.  That rule is family-independent
    # because it is stated in terms of image_token_id rather than any template.
    def prep(self, images):
        # A separator in this throwaway probe keeps variable-length OneVision anyres
        # expansions as distinct runs.  The separators are not copied into the real
        # prompt; only the exact per-image expansion counts and visual tensors are kept.
        probe = " ".join(self.IMG for _ in images)
        out = self.processor(text=[probe], images=images,
                             return_tensors="pt", padding=False)
        # ``mm_token_type_ids`` is derived from the *full* rendered token sequence.
        # Caching the value produced by a bare-placeholder prompt gave Qwen2.5/Qwen3
        # the correct input_ids but a shorter multimodal type vector.  Keep only the
        # image-dependent tensors here and reconstruct this cheap text-side vector in
        # ``fast_encode`` below.
        needs_mm_token_type_ids = "mm_token_type_ids" in out
        vis = {k: (v.to(self.device) if hasattr(v, "to") else v)
               for k, v in out.items() if k not in ("input_ids", "attention_mask")}
        vis.pop("mm_token_type_ids", None)
        counts = [len(b) for b in self.image_blocks(out["input_ids"], len(images))]
        if len(counts) != len(images):
            # Some LLaVA-OneVision checkpoints concatenate variable-length anyres
            # image expansions into one uninterrupted token run.  There is then no
            # lossless way to recover per-image counts from input_ids alone.  This is
            # a cache limitation, not an invalid model input, so use the official
            # processor path instead of aborting the evaluation.
            self.fast_ok = False
            print(f"  cannot cache {len(images)} variable image blocks; "
                  "falling back to the processor for this model", flush=True)
            return None
        return {"vis": vis, "counts": counts,
                "needs_mm_token_type_ids": needs_mm_token_type_ids}

    def fast_encode(self, text, pack):
        """Same input_ids the processor would build, without touching the pixels again."""
        imgtok = self.tok.decode([self.image_token_id])
        parts, k = text.split(self.IMG), 0
        s = parts[0]
        for p in parts[1:]:
            expanded = self.IMG.replace(imgtok, imgtok * pack["counts"][k])
            # HF's InternVL processor surrounds every repeated context block with
            # explicit instance delimiters.  Omitting these two tokens made the
            # generic fast path eight tokens short for four images and forced the
            # experiment to preprocess all pixels again for every question.
            if self.fam == "internvl":
                expanded = "<img>" + expanded + "</img>"
            s += expanded + p
            k += 1
        ids = self.tok(s, return_tensors="pt", add_special_tokens=False)
        inp = {"input_ids": ids["input_ids"].to(self.device),
               "attention_mask": ids["attention_mask"].to(self.device)}
        inp.update(pack["vis"])
        if pack.get("needs_mm_token_type_ids"):
            if not hasattr(self.processor, "create_mm_token_type_ids"):
                raise RuntimeError("processor returned mm_token_type_ids but cannot rebuild it")
            values = self.processor.create_mm_token_type_ids(ids["input_ids"].tolist())
            inp["mm_token_type_ids"] = torch.as_tensor(
                values, dtype=torch.long, device=self.device)
        return inp

    def check_fast(self, text, images):
        """The fast path must reproduce the processor exactly; asserted, not assumed."""
        exact = self.encode(text, images)
        pack = self.prep(images)
        if pack is None:
            self.fast_ok = False
            return False
        fast = self.fast_encode(text, pack)
        a = exact["input_ids"][0].tolist()
        b = fast["input_ids"][0].tolist()
        # Qwen3 carries an input_token_type_ids tensor whose length depends on the full
        # rendered prompt. Matching input_ids alone allowed a cached tensor built from
        # bare image placeholders to pass this check and fail inside get_rope_index.
        # Every tensor key and shape is therefore part of the cache contract.
        exact_shapes = {key: tuple(value.shape) for key, value in exact.items()
                        if hasattr(value, "shape")}
        fast_shapes = {key: tuple(value.shape) for key, value in fast.items()
                       if hasattr(value, "shape")}
        ok = a == b and exact_shapes == fast_shapes
        # A family whose processor inserts anything beyond the repeated image token cannot
        # use the fast path.  Rather than producing a length mismatch on every forward, the
        # harness records the failure once and falls back to the processor for that model.
        self.fast_ok = ok
        detail = ""
        if not ok:
            detail = (f"  ({len(a)} vs {len(b)} tokens; shapes "
                      f"{exact_shapes} vs {fast_shapes}); falling back to the processor "
                      "for this model")
        print(f"  fast encode matches processor: {ok}" + detail, flush=True)
        return ok

    # ---------- introspection ----------
    def image_blocks(self, input_ids, n=None):
        """Contiguous runs of image tokens, one per image.

        LLaVA-style templates place nothing between consecutive image placeholders, so
        four images become a single run of image tokens rather than four.  When the caller
        knows how many images there are and the run divides evenly, the run is split into
        equal parts -- correct here because every image is fed at the same resolution and
        therefore contributes the same number of tokens."""
        b = self._runs(input_ids)
        if n is None or len(b) == n: return b
        flat = [i for r in b for i in r]
        c = getattr(self, "_last_counts", None)
        if c and len(c) == n and sum(c) == len(flat):                              # the processor's own per-image counts (exact)
            out, s0 = [], 0
            for k in c:
                out.append(flat[s0:s0 + k]); s0 += k
            return out
        if len(flat) % n == 0:
            w = len(flat) // n
            return [flat[k * w:(k + 1) * w] for k in range(n)]
        return b

    def _runs(self, input_ids):
        pos = (input_ids[0] == self.image_token_id).nonzero().flatten().tolist()
        if not pos: return []
        blocks, cur = [], [pos[0]]
        for a, b in zip(pos, pos[1:]):
            if b == a + 1: cur.append(b)
            else: blocks.append(cur); cur = [b]
        blocks.append(cur)
        return blocks

    @torch.no_grad()
    def forward(self, text, images, output_hidden_states=False, output_attentions=False,
                pack=None):
        use = pack is not None and self.fast_ok
        inp = self.fast_encode(text, pack) if use else self.encode(text, images)
        try:
            out = self.model(**inp, use_cache=False, return_dict=True, **self.forward_extra,
                             output_hidden_states=output_hidden_states,
                             output_attentions=output_attentions)
        except (RuntimeError, IndexError, ValueError) as e:
            # Matching input_ids is not sufficient for every family: some models derive
            # internal masks from tensors the cached vision pack does not reproduce, and
            # the failure only appears at the forward.  One retry through the processor
            # settles it, and the fast path is switched off for this model afterwards.
            if not use: raise
            self.fast_ok = False
            print(f"  fast path rejected by the model ({e}); using the processor",
                  flush=True)
            inp = self.encode(text, images)
            out = self.model(**inp, use_cache=False, return_dict=True, **self.forward_extra,
                             output_hidden_states=output_hidden_states,
                             output_attentions=output_attentions)
        return out, inp

    @torch.no_grad()
    def label_logprobs(self, text, images, first_ids, pack=None):
        """log-softmax restricted to the candidate labels present in this trial."""
        out, _ = self.forward(text, images, pack=pack)
        g = out.logits[0, -1].float()
        return torch.log_softmax(g[torch.tensor(first_ids)].double(), 0), g


class LayerMasks:
    """Swap the additive attention mask inside chosen layers.  Portable because every
    decoder layer in these families takes attention_mask as a keyword."""
    def __init__(self, uni):
        self.masks = {}
        self.h = [l.register_forward_pre_hook(self._mk(i), with_kwargs=True)
                  for i, l in enumerate(uni.layers)]

    def _mk(self, i):
        def hook(mod, args, kwargs):
            if i in self.masks:
                m = self.masks[i]
                # with a sharded model each layer sits on its own card, so the mask has to
                # follow the hidden states rather than assuming one device
                dev = args[0].device if args else kwargs.get("hidden_states").device
                kwargs["attention_mask"] = m if m.device == dev else m.to(dev)
            return args, kwargs
        return hook

    def clear(self): self.masks = {}

    def remove(self):
        for h in self.h: h.remove()


def base_mask(T, device, dtype):
    m = torch.full((T, T), torch.finfo(dtype).min, device=device, dtype=dtype)
    m = torch.triu(m, diagonal=1)
    return m[None, None]


def find_sub(idl, sub, after=0):
    for i in range(after, len(idl) - len(sub) + 1):
        if idl[i:i + len(sub)] == sub: return i
    return -1
