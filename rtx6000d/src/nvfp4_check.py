import torch
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant

torch.manual_seed(0)
E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device='cuda')

def dequant_nvfp4(packed, scales, global_scale, M, K):
    # packed: (M, K//2) uint8, two e2m1 per byte (low nibble first per NVFP4 convention)
    lo = (packed & 0xF).long()
    hi = (packed >> 4).long()
    vals = torch.stack([E2M1[lo], E2M1[hi]], dim=-1).reshape(M, K)
    bs = scales.to(torch.float32)  # (M, K//16) e4m3
    vals = vals.reshape(M, K // 16, 16) * bs.reshape(M, K // 16, 1).to(torch.float32)
    return vals.reshape(M, K) * global_scale

M, N, K = 512, 512, 512
g = torch.tensor(1.0, device='cuda')
x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16).clamp(-1, 1)
w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16).clamp(-1, 1)
x_fp4, x_bs = scaled_fp4_quant(x, g, is_sf_swizzled_layout=True, backend="cutlass", padded_n=K)
w_fp4, w_bs = scaled_fp4_quant(w, g, is_sf_swizzled_layout=True, backend="cutlass", padded_n=K)
alpha = torch.tensor(1.0, device='cuda', dtype=torch.float32)
out = cutlass_scaled_fp4_mm(x_fp4, w_fp4, x_bs, w_bs, alpha, torch.bfloat16)

# reference: dequant then matmul. NOTE: scales may be swizzled; unswizzle if needed
try:
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
    # x_bs may already be swizzled by quant; try direct first
    xd = dequant_nvfp4(x_fp4, x_bs, 1.0, M, K)
    wd = dequant_nvfp4(w_fp4, w_bs, 1.0, N, K)
    ref = xd @ wd.t() * g * g
except Exception as e:
    print("dequant path fail:", e)
    ref = None

if ref is not None:
    diff = (out.float() - ref).abs()
    rel = diff.mean() / ref.abs().mean()
    print(f"# ref-check: mean|out|={out.float().abs().mean():.4f} mean|ref|={ref.abs().mean():.4f} mean_rel_diff={rel.item():.4f}")
    # nvfp4 quantization error itself: compare ref vs original fp matmul
    orig = x.float() @ w.float().t()
    qerr = (ref - orig).abs().mean() / orig.abs().mean()
    print(f"# quant-noise: mean_rel_quant_err={qerr.item():.4f}")
    ok = rel.item() < 0.15  # kernel-vs-ref should be small if layout right
    print(f"CSV,nvfp4_correctness,kernel_vs_ref_rel,{rel.item():.4f},ratio,{'OK' if ok else 'LAYOUT-MISMATCH'}")
else:
    print("CSV,nvfp4_correctness,kernel_vs_ref_rel,-1,ratio,dequant-failed")
