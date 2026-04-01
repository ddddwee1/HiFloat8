import torch
import torch.nn as nn

from quant_cy import QLinear
from quant_cy.utils.utils import replace_linear


# This demo shows module-level replacement for hif8.
# The script is meant to be presentation-friendly, so each step has an explicit
# structural or numerical check attached to it.
torch.manual_seed(42)


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Nested Linear layers let us demonstrate recursive replacement inside
        # submodules, not just top-level attributes.
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward(self, x):
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(128, 256)
        self.block = ToyBlock()
        self.head = nn.Linear(256, 128)

    def forward(self, x):
        # The data flow is intentionally simple so the demo can focus on
        # replace_linear rather than model architecture.
        x = self.embed(x)
        x = self.block(x)
        return self.head(x)


# Prepare the model on CUDA in bf16, then replace all nn.Linear modules.
model = ToyModel().cuda().bfloat16()
linear_names = [n for n, m in model.named_modules() if type(m) is nn.Linear]
print("Linear layers before replace:", linear_names)

# quant_grad=False means the backward pass keeps original gradients and does not
# quantize grad_out inside QLinear backward.
replace_linear(model, "hif8", in_Q="hif8", quant_grad=False)

replaced_names = [n for n, m in model.named_modules() if isinstance(m, QLinear)]
remaining_linear_names = [n for n, m in model.named_modules() if type(m) is nn.Linear]

print("QLinear layers after replace:", replaced_names)
print("Remaining nn.Linear layers:", remaining_linear_names)

# After replacement, the set of module names should be preserved, only the class
# of each layer changes from nn.Linear to QLinear.
assert replaced_names == linear_names
assert not remaining_linear_names

# Run a standard training-style step to prove the replaced model still behaves
# like a normal autograd module on CUDA bf16 inputs.
# Use batch=1 and 128-aligned dimensions to reflect common model layouts.
x = torch.randn(1, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
out = model(x)
loss = out.float().square().mean()
loss.backward()

# Validate runtime health of outputs and gradients.
assert x.grad is not None, "Model backward did not produce input gradients"
assert torch.isfinite(out.float()).all().item()
assert torch.isfinite(x.grad.float()).all().item()

# Validate parameter gradients on each replaced layer.
for name in replaced_names:
    module = dict(model.named_modules())[name]
    assert module.weight.grad is not None, f"{name}.weight grad is missing"
    assert torch.isfinite(module.weight.grad.float()).all().item(), f"{name}.weight grad is not finite"

# Keep a compact summary in stdout so the demo output is easy to show.
print("Output dtype:", out.dtype)
print("Loss:", float(loss))
print("replace_linear bf16 test passed")
