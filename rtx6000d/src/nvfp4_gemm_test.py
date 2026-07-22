import torch, inspect
from vllm._custom_ops import cutlass_scaled_fp4_mm
try:
    from vllm.model_executor.layers.quantization.fp_quant import scaled_fp4_quant
    print("# scaled_fp4_quant from fp_quant")
except Exception:
    from vllm._custom_ops import scaled_fp4_quant
    print("# scaled_fp4_quant from _custom_ops")
print("# sig:", inspect.signature(scaled_fp4_quant))

def bench(fn, warmup=5, reps=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts)//2]

M = N = K = 8192
flops = 2.0 * M * N * K
print(f"# torch {torch.__version__} dev {torch.cuda.get_device_name(0)}")

g = torch.tensor(1.0, device='cuda')
try:
    x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16).clamp(-1, 1)
    r = scaled_fp4_quant(x, g, is_sf_swizzled_layout=True, backend="cutlass", padded_n=K)
    x_fp4, x_bs = r
    w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16).clamp(-1, 1)
    w_fp4, w_bs = scaled_fp4_quant(w, g, is_sf_swizzled_layout=True, backend="cutlass", padded_n=K)
    print("# x_fp4", x_fp4.shape, x_fp4.dtype, "x_bs", x_bs.shape, x_bs.dtype)
    print("# w_fp4", w_fp4.shape, w_fp4.dtype, "w_bs", w_bs.shape, w_bs.dtype)
    alpha = torch.tensor(1.0, device='cuda', dtype=torch.float32)
    out = cutlass_scaled_fp4_mm(x_fp4, w_fp4, x_bs, w_bs, alpha, torch.bfloat16)
    print("# out", out.shape, out.dtype, "finite:", torch.isfinite(out).all().item())
    ms = bench(lambda: cutlass_scaled_fp4_mm(x_fp4, w_fp4, x_bs, w_bs, alpha, torch.bfloat16))
    print(f"CSV,vllm_cutlass,nvfp4_gemm,{flops/(ms*1e-3)/1e12:.2f},TFLOPS,nvfp4 block16 e4m3-scale bf16-out")
except Exception as ex:
    import traceback; traceback.print_exc()
    print(f"CSV,vllm_cutlass,nvfp4_gemm,0,TFLOPS,FAIL: {str(ex)[:200]}")

# also fp8 cutlass in this image for cross-check
try:
    from vllm import _custom_ops as ops
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16).clamp(-1, 1)
    aq, sa = ops.scaled_fp8_quant(a)
    b = torch.randn(N, K, device='cuda', dtype=torch.bfloat16).clamp(-1, 1)
    bq, sb = ops.scaled_fp8_quant(b)
    ms = bench(lambda: ops.cutlass_scaled_mm(aq, bq.t(), sa, sb, out_dtype=torch.bfloat16))
    print(f"CSV,vllm_cutlass,fp8_gemm,{flops/(ms*1e-3)/1e12:.2f},TFLOPS,")
except Exception as ex:
    print(f"CSV,vllm_cutlass,fp8_gemm,0,TFLOPS,FAIL: {str(ex)[:120]}")
