# NVFP4 block-scaled GEMM via native torch._scaled_mm (no vllm dependency).
# Falls back gracefully if the build lacks Float4_e2m1fn_x2 / nvfp4 support.
import torch

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

try:
    fp4 = torch.float4_e2m1fn_x2
except AttributeError:
    print("CSV,nvfp4_torch_scaled_mm,support,0,bool,no float4_e2m1fn_x2 dtype")
    raise SystemExit(0)

a = (torch.rand(M, K // 2, device='cuda') * 255).to(torch.uint8).view(fp4)
b = (torch.rand(N, K // 2, device='cuda') * 255).to(torch.uint8).view(fp4)
sf8 = torch.float8_e4m3fn
# block scales: 16 elements per block along K
sa = (torch.rand(M, K // 16, device='cuda') * 0.1 + 0.05).to(sf8)
sb = (torch.rand(N, K // 16, device='cuda') * 0.1 + 0.05).to(sf8)
try:
    fn = lambda: torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
    fn()
except Exception as ex:
    print(f"CSV,nvfp4_torch_scaled_mm,support,0,bool,{type(ex).__name__}")
    raise SystemExit(0)
ms = bench(fn)
print(f"CSV,nvfp4_torch_scaled_mm,gemm_{M}x{N}x{K},{flops/ms/1e9:.1f},TFLOPS,native torch._scaled_mm fp4e2m1 block16")

# bf16 reference on the same GPU
x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
w = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)
ms16 = bench(lambda: x @ w.t())
print(f"CSV,bf16_torch_mm,gemm_{M}x{N}x{K},{flops/ms16/1e9:.1f},TFLOPS,torch matmul bf16")
