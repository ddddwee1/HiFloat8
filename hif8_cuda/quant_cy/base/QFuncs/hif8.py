import torch 
from ..QType import QType 
from torch import Tensor 


@torch.no_grad()
def quant_hif8(x: Tensor, Q: QType, qdim: int): 
    print('HIF8 PYTORCH')
    x_fp32 = x.to(torch.float32)
    absx = torch.abs(x_fp32)
    res = torch.empty_like(x_fp32)

    finite_mask = torch.isfinite(x_fp32)
    overflow_mask = finite_mask & (absx >= 2.0**15 * 1.25)
    underflow_mask = finite_mask & (absx < 2.0**-23)
    quant_mask = finite_mask & ~(overflow_mask | underflow_mask)

    res[~finite_mask] = x_fp32[~finite_mask]
    res[underflow_mask] = 0.0
    res[overflow_mask] = torch.copysign(torch.full_like(x_fp32[overflow_mask], torch.inf), x_fp32[overflow_mask])

    if torch.any(quant_mask):
        absx_q = absx[quant_mask]
        e = torch.floor(torch.log2(absx_q))
        e = torch.where(e == -23.0, torch.full_like(e, -22.0), e)

        abs_e = torch.abs(e)
        mant_bits = torch.zeros_like(e)
        mant_bits = torch.where(abs_e <= 15.0, torch.ones_like(e), mant_bits)
        mant_bits = torch.where(abs_e <= 7.0, torch.full_like(e, 2.0), mant_bits)
        mant_bits = torch.where(abs_e <= 3.0, torch.full_like(e, 3.0), mant_bits)

        scale = torch.pow(2.0, -e + mant_bits)
        quant_abs = torch.floor(absx_q * scale + 0.5) * torch.pow(2.0, e - mant_bits)
        res[quant_mask] = torch.copysign(quant_abs, x_fp32[quant_mask])

    return res.to(x.dtype)
