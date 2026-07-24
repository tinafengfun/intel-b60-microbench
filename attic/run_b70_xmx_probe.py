# WARNING: DCE-FLAWED intermediate version — see attic/README.md. Do not use for measurements.
#!/usr/bin/env python3
"""
B70 (BMG-G31) XMX/DPAS scaling probe — designed to localize why full-GPU DPAS
throughput (87.6 TF) is far below the single-EU extrapolation (256 x 0.535 = 137 TF).

P1: Core scaling   — sg_per_wg=8 (fill one Xe core), n_wg = 1..256, ILP=16.
    Per-EU rate vs number of active cores -> knee localizes the shared bottleneck.
P2: EU scaling     — n_wg=1 (one core), sg_per_wg = 1..32, ILP=16.
    Per-EU rate vs active EUs within a core -> tests EU-pair XMX sharing.
P3: Sustained load + telemetry — long kernel, sample xpu-smi (util/power/freq/
    EU active/EU stall) concurrently -> rules out in-kernel DVFS/power throttle.
'peak' mode: single full-machine run (n_wg=32, sg=8) for clock-scaling sweeps
    driven externally via xpu-smi --frequencyrange.
"""
import subprocess, sys, re, csv, time, threading
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DEVICE = "bmg-g31"
SPIRV_RUNNER = SCRIPT_DIR / "spirv_runner"
RESULTS_DIR = SCRIPT_DIR / "results"
GHZ = 2.4
FLOPS_PER_DPAS = 4096


def run_cmd(cmd, cwd=SCRIPT_DIR, check=True):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"ERROR: {cmd}\n{r.stderr}", file=sys.stderr)
    return r


def gen_ilp_kernel(ilp, n_iter):
    """ILP-independent-accumulator DPAS throughput kernel (bf16 tiles as ushort)."""
    lines = [
        "; SPIR-V 1.4",
        f"; DPAS ILP={ilp} throughput, {n_iter} iters",
        "               OpCapability Addresses",
        "               OpCapability Kernel",
        "               OpCapability Int64",
        "               OpCapability Int16",
        "               OpCapability CooperativeMatrixKHR",
        '               OpExtension "SPV_KHR_cooperative_matrix"',
        '          %1 = OpExtInstImport "OpenCL.std"',
        "               OpMemoryModel Physical64 OpenCL",
        "               OpEntryPoint Kernel %main \"dpas_ilp\"",
        "               OpExecutionMode %main SubgroupSize 16",
        "     %void   = OpTypeVoid",
        "     %bool   = OpTypeBool",
        "     %uint   = OpTypeInt 32 0",
        "     %ulong  = OpTypeInt 64 0",
        "     %float  = OpTypeFloat 32",
        "     %ushort = OpTypeInt 16 0",
        "   %uint_0   = OpConstant %uint 0",
        "   %uint_1   = OpConstant %uint 1",
        "   %uint_2   = OpConstant %uint 2",
        "   %uint_3   = OpConstant %uint 3",
        "   %uint_8   = OpConstant %uint 8",
        "  %uint_16   = OpConstant %uint 16",
        f"  %uint_{n_iter}  = OpConstant %uint {n_iter}",
        "%ptr_cross_ushort = OpTypePointer CrossWorkgroup %ushort",
        "%ptr_cross_float  = OpTypePointer CrossWorkgroup %float",
        "  %cm_acc = OpTypeCooperativeMatrixKHR %float %uint_3 %uint_8 %uint_16 %uint_2",
        "  %cm_a   = OpTypeCooperativeMatrixKHR %ushort %uint_3 %uint_8 %uint_16 %uint_0",
        "  %cm_b   = OpTypeCooperativeMatrixKHR %ushort %uint_3 %uint_16 %uint_16 %uint_1",
        "  %fn_type = OpTypeFunction %void %ptr_cross_ushort %ptr_cross_ushort %ptr_cross_float",
        "  %main = OpFunction %void None %fn_type",
        "  %buf_a = OpFunctionParameter %ptr_cross_ushort",
        "  %buf_b = OpFunctionParameter %ptr_cross_ushort",
        "  %buf_c = OpFunctionParameter %ptr_cross_float",
        "  %entry = OpLabel",
    ]
    for i in range(ilp):
        lines.append(f"  %a_tile{i} = OpCooperativeMatrixLoadKHR %cm_a %buf_a %uint_0 %uint_16 None")
        lines.append(f"  %b_tile{i} = OpCooperativeMatrixLoadKHR %cm_b %buf_b %uint_0 %uint_16 None")
        lines.append(f"  %acc_init{i} = OpCooperativeMatrixLoadKHR %cm_acc %buf_c %uint_0 %uint_16 None")
    lines.append("               OpBranch %lh")
    lines.append("%lh = OpLabel")
    lines.append("               OpLoopMerge %lx %lh None")
    for i in range(ilp):
        lines.append(f"  %acc_phi{i} = OpPhi %cm_acc %acc_init{i} %entry %acc_next{i} %lb")
    lines.append("    %i_phi = OpPhi %uint %uint_0 %entry %i_next %lb")
    lines.append(f"      %cond = OpULessThan %bool %i_phi %uint_{n_iter}")
    lines.append("               OpBranchConditional %cond %lb %lx")
    lines.append("%lb = OpLabel")
    for i in range(ilp):
        lines.append(f"  %acc_next{i} = OpCooperativeMatrixMulAddKHR %cm_acc %a_tile{i} %b_tile{i} %acc_phi{i}")
    lines.append("    %i_next = OpIAdd %uint %i_phi %uint_1")
    lines.append("               OpBranch %lh")
    lines.append("%lx = OpLabel")
    lines.append("               OpCooperativeMatrixStoreKHR %buf_c %acc_phi0 %uint_0 %uint_16 None")
    lines.append("               OpReturn")
    lines.append("               OpFunctionEnd")
    return "\n".join(lines)


def build_and_compile(spvasm_text, name):
    spvasm = SCRIPT_DIR / f"probe_{name}.spvasm"
    spv = SCRIPT_DIR / f"probe_{name}.spv"
    spvasm.write_text(spvasm_text)
    r = run_cmd(f"spirv-as --target-env spv1.4 {spvasm} -o {spv}", check=False)
    if r.returncode != 0:
        print(f"  ASSEMBLE FAILED: {r.stderr[:200]}")
        return None
    r = run_cmd(f"ocloc compile -spirv_input -file {spv} -device {DEVICE} -output {name}", check=False)
    if r.returncode != 0 or "error" in (r.stderr + r.stdout).lower():
        print(f"  COMPILE FAILED: {(r.stderr + r.stdout)[:200]}")
        return None
    return spv


def cleanup(name):
    for f in SCRIPT_DIR.glob(f"probe_{name}*"):
        f.unlink(missing_ok=True)
    for f in SCRIPT_DIR.glob(f"{name}*"):
        if f.suffix in (".bin", ".spv", ".spvasm") or "_bmg" in f.name:
            f.unlink(missing_ok=True)


def run_benchmark(spv_path, wg_x, n_wg, repeats=30):
    r = run_cmd(f"{SPIRV_RUNNER} {spv_path} dpas_ilp {wg_x} {n_wg} {repeats}", check=False)
    if r.returncode != 0:
        return None, r.stderr[:200]
    m = re.search(r'Median=([\d.]+)\s+ns', r.stdout)
    if m:
        return float(m.group(1)), None
    return None, "parse failed: " + r.stdout[:100]


def tflops(median_ns, ilp, n_iter, total_sgs):
    flops = ilp * n_iter * FLOPS_PER_DPAS * total_sgs
    return flops / (median_ns * 1e-9) / 1e12


def experiment_p1(spv, ilp, n_iter):
    """Core scaling: sg=8/WG (fill one core), sweep n_wg."""
    print("\n=== P1: Core scaling (sg=8/WG, ILP=%d, n_iter=%d) ===" % (ilp, n_iter))
    print(f"  {'n_wg':>5s} {'cores~':>6s} {'SGs':>5s} {'Median(ns)':>12s} {'TFLOPS':>8s} {'TF/core':>8s} {'TF/EU':>8s}")
    results = []
    for n_wg in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        sg = 8
        total_sgs = sg * n_wg
        median, err = run_benchmark(spv, sg * 16, n_wg)
        if median is None:
            print(f"  {n_wg:>5d} FAILED: {err}")
            continue
        tf = tflops(median, ilp, n_iter, total_sgs)
        cores = min(n_wg, 32)
        print(f"  {n_wg:>5d} {cores:>6d} {total_sgs:>5d} {median:>12.0f} {tf:>8.2f} {tf/cores:>8.3f} {tf/total_sgs:>8.4f}")
        results.append({'exp': 'P1_core_scaling', 'n_wg': n_wg, 'sg_per_wg': sg,
                        'total_sgs': total_sgs, 'median_ns': median, 'tflops': tf,
                        'tf_per_eu': tf / total_sgs})
    return results


def experiment_p2(spv, ilp, n_iter):
    """EU scaling within one core: n_wg=1, sweep sg_per_wg."""
    print("\n=== P2: EU scaling within one core (n_wg=1, ILP=%d, n_iter=%d) ===" % (ilp, n_iter))
    print(f"  {'sg/wg':>5s} {'Median(ns)':>12s} {'TFLOPS':>8s} {'TF/EU':>8s} {'rel_to_1eu':>10s}")
    results = []
    base = None
    for sg in [1, 2, 4, 8, 16, 32]:
        median, err = run_benchmark(spv, sg * 16, 1)
        if median is None:
            print(f"  {sg:>5d} FAILED: {err}")
            continue
        tf = tflops(median, ilp, n_iter, sg)
        per_eu = tf / min(sg, 8)  # sg>8 => 2+ threads/EU, still 8 EUs
        if base is None:
            base = per_eu
        print(f"  {sg:>5d} {median:>12.0f} {tf:>8.2f} {per_eu:>8.4f} {per_eu/base:>10.3f}")
        results.append({'exp': 'P2_eu_scaling_1core', 'n_wg': 1, 'sg_per_wg': sg,
                        'total_sgs': sg, 'median_ns': median, 'tflops': tf,
                        'tf_per_eu': per_eu})
    return results


def experiment_p3(spv_name="ilp16_long"):
    """Sustained full-machine load with concurrent xpu-smi telemetry."""
    print("\n=== P3: Sustained load + telemetry (n_wg=32, sg=8, long kernel) ===")
    n_iter = 100000
    ilp = 16
    spv = build_and_compile(gen_ilp_kernel(ilp, n_iter), spv_name)
    if not spv:
        return []
    samples = []

    def sample():
        r = subprocess.run(
            "xpu-smi dump -d 0 -m 0,1,2,9,10 -i 100 -n 30",
            shell=True, capture_output=True, text=True)
        samples.append(r.stdout)

    t = threading.Thread(target=sample)
    t.start()
    time.sleep(0.3)
    median, err = run_benchmark(spv, 128, 32, repeats=3)
    t.join()
    tf = tflops(median, ilp, n_iter, 256) if median else 0
    print(f"  kernel median: {median:.0f} ns  ({tf:.2f} TFLOPS)" if median else f"  FAILED: {err}")
    print("  telemetry (util%, power W, freq MHz, EU active%, EU stall%):")
    for s in samples:
        for line in s.splitlines():
            print("   ", line)
    cleanup(spv_name)
    return [{'exp': 'P3_sustained', 'n_wg': 32, 'sg_per_wg': 8, 'total_sgs': 256,
             'median_ns': median or 0, 'tflops': tf, 'tf_per_eu': tf / 256}]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    if not SPIRV_RUNNER.exists():
        run_cmd("g++ -std=c++17 -O2 -o spirv_runner spirv_runner.cpp -lze_loader -lm")

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "peak":
        # Single full-machine run for external clock-scaling sweeps.
        ilp, n_iter = 16, 512
        spv = build_and_compile(gen_ilp_kernel(ilp, n_iter), "ilp16_peak")
        median, err = run_benchmark(spv, 128, 32, repeats=30)
        tf = tflops(median, ilp, n_iter, 256)
        print(f"PEAK n_wg=32 sg=8: median={median:.0f} ns  {tf:.2f} TFLOPS")
        cleanup("ilp16_peak")
        return

    ilp, n_iter = 16, 512
    spv = build_and_compile(gen_ilp_kernel(ilp, n_iter), "ilp16_probe")
    if not spv:
        sys.exit(1)

    results = []
    if mode in ("all", "p1"):
        results += experiment_p1(spv, ilp, n_iter)
    if mode in ("all", "p2"):
        results += experiment_p2(spv, ilp, n_iter)
    cleanup("ilp16_probe")
    if mode in ("all", "p3"):
        results += experiment_p3()

    if results:
        path = RESULTS_DIR / "b70_xmx_scaling.csv"
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
