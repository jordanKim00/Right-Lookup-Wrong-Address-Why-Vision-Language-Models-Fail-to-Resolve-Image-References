"""S872b runner with the registration-time DeltaNet kernel (10-22): S872b (registered 10-21 20:12) ran S872's batched vector-Jacobian
products (torch.autograd.grad with is_grads_batched=True, i.e. vmap) through the torch DeltaNet path; the resident worker now uses the fla 0.3.0
Triton kernels, whose backward cannot take vmap's batched tensors ("Cannot access data pointer of Tensor that doesn't have storage", 416, 10-22
10:56).  This runner swaps transformers' module-level torch_chunk_gated_delta_rule (wrapped to dispatch to fla) for its unwrapped torch
implementation for the duration of the job, runs exp/s872_jlens.py unchanged with the job's argv, and restores the fla dispatch afterwards."""
import inspect, runpy, sys

import transformers.models.qwen3_5.modeling_qwen3_5 as M

orig = M.torch_chunk_gated_delta_rule
torch_impl = inspect.unwrap(orig)
assert torch_impl is not orig and torch_impl.__module__ == M.__name__, "could not unwrap the torch chunk_gated_delta_rule"
M.torch_chunk_gated_delta_rule = torch_impl
print(f"[s872b_fallback] DeltaNet chunk rule -> {torch_impl.__module__}.{torch_impl.__name__} (torch path) for this job", flush=True)
try:
    sys.argv = ["/vlm/exp/s872_jlens.py"] + sys.argv[1:]
    runpy.run_path("/vlm/exp/s872_jlens.py", run_name="__main__")
finally:
    M.torch_chunk_gated_delta_rule = orig
    print("[s872b_fallback] fla dispatch restored", flush=True)
