import torch, time, sys

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
print(f"# torch {torch.__version__} cuda {torch.version.cuda} dev {torch.cuda.get_device_name(0)}")

# --- torch._scaled_mm FP8 (per-tensor scale) ---
try:
    a = torch.randn(M, K, device='cuda').clamp(-1, 1).to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device='cuda').clamp(-1, 1).to(torch.float8_e4m3fn).t()
    sa = torch.tensor(1.0, device='cuda'); sb = torch.tensor(1.0, device='cuda')
    c = torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16)
    ms = bench(lambda: torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16))
    print(f"CSV,torch_scaled_mm,fp8e4m3_pertensor,{flops/(ms*1e-3)/1e12:.2f},TFLOPS,bf16 out")
except Exception as ex:
    print(f"CSV,torch_scaled_mm,fp8e4m3_pertensor,0,TFLOPS,FAIL: {str(ex)[:120]}")

# --- torch._scaled_mm FP8 rowwise ---
try:
    a = torch.randn(M, K, device='cuda').clamp(-1, 1).to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device='cuda').clamp(-1, 1).to(torch.float8_e4m3fn).t()
    sa = torch.ones(M, 1, device='cuda'); sb = torch.ones(1, N, device='cuda')
    c = torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16)
    ms = bench(lambda: torch._scaled_mm(a, b, sa, sb, out_dtype=torch.bfloat16))
    print(f"CSV,torch_scaled_mm,fp8e4m3_rowwise,{flops/(ms*1e-3)/1e12:.2f},TFLOPS,bf16 out")
except Exception as ex:
    print(f"CSV,torch_scaled_mm,fp8e4m3_rowwise,0,TFLOPS,FAIL: {str(ex)[:120]}")

# --- vLLM cutlass FP8 ---
try:
    from vllm import _custom_ops as ops
    a = torch.randn(M, K, device='cuda', dtype=torch.bfloat16).clamp(-1, 1)
    aq, sa = ops.scaled_fp8_quant(a)
    b = torch.randn(N, K, device='cuda', dtype=torch.bfloat16).clamp(-1, 1)
    bq, sb = ops.scaled_fp8_quant(b)
    c = ops.cutlass_scaled_mm(aq, bq.t(), sa, sb, out_dtype=torch.bfloat16)
    ms = bench(lambda: ops.cutlass_scaled_mm(aq, bq.t(), sa, sb, out_dtype=torch.bfloat16))
    print(f"CSV,vllm_cutlass,fp8e4m3_pertensor,{flops/(ms*1e-3)/1e12:.2f},TFLOPS,bf16 out")
except Exception as ex:
    print(f"CSV,vllm_cutlass,fp8e4m3,0,TFLOPS,FAIL: {str(ex)[:120]}")

# --- FP16 reference via torch.matmul ---
try:
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = a @ b
    ms = bench(lambda: a @ b)
    print(f"CSV,torch_matmul,fp16,{flops/(ms*1e-3)/1e12:.2f},TFLOPS,")
except Exception as ex:
    print(f"CSV,torch_matmul,fp16,0,TFLOPS,FAIL: {str(ex)[:120]}")

# --- nvfp4 via modelopt / compressed-tensors if present ---
try:
    import modelopt  # noqa
    print("CSV,modelopt,present,1,bool,")
except Exception:
    print("CSV,modelopt,present,0,bool,")
try:
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import cutlass_fp4_gemm  # noqa
    print("CSV,vllm_nvfp4_utils,present,1,bool,")
except Exception as ex:
    print(f"CSV,vllm_nvfp4_utils,present,0,bool,{str(ex)[:80]}")
