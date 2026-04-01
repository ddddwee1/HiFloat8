import numpy as np
import torch

from quant_cy import QType, quant_dequant_float, quant_func


# This demo is the hif8 counterpart of the hifx4 quant_func example.
# It verifies both:
# 1. quant_func forward matches quant_dequant_float.
# 2. quant_func backward behaves as a straight-through estimator.
np.random.seed(42)
torch.manual_seed(42)

# Use model-like sizes that are all multiples of 128.
M = 128
N = 256

# Construct reproducible bf16 input data with varied value ranges.
x = (0.2 * np.random.randn(M, N) + np.random.uniform(-0.03, 0.04, (M, N))).astype(np.float32)
x_torch = torch.from_numpy(x).bfloat16()

# This tensor serves as the known upstream gradient for the backward check.
grad_seed = torch.randint(-3, 4, (M, N), dtype=torch.int32).to(torch.bfloat16)

print(x_torch.shape)

# Quantize along dim 0 for consistency with the existing quant_dequant demos.
qtype_str = "hif8"
print("Qtype string: %s " % (qtype_str))
quant_type = QType(qtype_str).dim(0)

# Python reference path used as the correctness baseline.
y_ref = quant_dequant_float(x_torch, quant_type, force_py=True, force_fp32=True).cpu().float()

# CUDA autograd path under test.
x_cuda = x_torch.cuda().detach().clone().requires_grad_(True)
grad_seed_cuda = grad_seed.cuda()
y_cuda = quant_func(x_cuda, quant_type, force_py=False)

# Forward result should exactly match the reference.
forward_diff = (y_ref - y_cuda.detach().cpu().float()).abs().max().item()
print("ABS diff max (quant_dequant_float <-> quant_func):", forward_diff)

# The loss is chosen so its gradient with respect to y_cuda is exactly grad_seed.
loss = (y_cuda.float() * grad_seed_cuda.float()).sum()
loss.backward()

assert x_cuda.grad is not None, "quant_func backward did not produce input gradients"

# Straight-through backward means grad(x) == upstream gradient.
grad_diff = (x_cuda.grad.float() - grad_seed_cuda.float()).abs().max().item()
print("ABS diff max (input grad <-> upstream grad):", grad_diff)

assert forward_diff == 0.0
assert grad_diff == 0.0
print("quant_func bf16 test passed")
