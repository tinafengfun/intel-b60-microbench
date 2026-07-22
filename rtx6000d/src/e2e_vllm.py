# End-to-end vLLM throughput benchmark for RTX 6000D (GPU 0), NVFP4 model.
# Measures: prefill tok/s, decode tok/s @ batch=1 and large batch.
import json, os, sys, time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def main():
    from vllm import LLM, SamplingParams

    MODEL = os.environ.get("MODEL_PATH", "/disk/Qwen3.6-27B-NVFP4")
    OUT = os.environ.get("OUT_JSON", "/work/results/e2e_vllm.json")

    results = {}

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=512,
        enforce_eager=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )

    tok = llm.get_tokenizer()

    def make_prompt(n_tokens, seed_word="the"):
        # deterministic ~n_tokens prompt
        base = tok.encode("Explain the internals of GPU tensor cores. " * 2000)
        ids = base[:n_tokens]
        return tok.decode(ids)

    def run_case(name, prompts, max_tokens, ignore_eos=True):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=ignore_eos)
        # warmup with the same shape once
        llm.generate(prompts[:1], SamplingParams(temperature=0.0, max_tokens=min(max_tokens, 32), ignore_eos=True))
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        in_toks = sum(len(o.prompt_token_ids) for o in outs)
        out_toks = sum(len(o.outputs[0].token_ids) for o in outs)
        res = {
            "n_prompts": len(prompts),
            "in_tokens_total": in_toks,
            "out_tokens_total": out_toks,
            "wall_s": round(dt, 3),
            "decode_tok_per_s": round(out_toks / dt, 1),
            "total_tok_per_s": round((in_toks + out_toks) / dt, 1),
        }
        results[name] = res
        print(f"[{name}] {json.dumps(res)}", flush=True)
        return res

    # 1) decode, batch=1
    p_short = "Write a long, detailed essay about the history of computing."
    run_case("decode_bs1", [p_short], 256)

    # 2) decode, large batch (32 distinct prompts to avoid prefix-cache collapse)
    prompts32 = [f"Essay topic #{i}: describe a different planet in detail." for i in range(32)]
    run_case("decode_bs32", prompts32, 256)

    # 3) prefill, batch=1, ~2048-token prompt, 1 output token
    p_long = make_prompt(2048)
    run_case("prefill_2048_bs1", [p_long], 1)
    # recompute prefill throughput from wall time of the 1-token case
    r = results["prefill_2048_bs1"]
    r["prefill_tok_per_s"] = round(r["in_tokens_total"] / r["wall_s"], 1)

    # 4) prefill heavy: 8 x 2048-token prompts, 1 output token
    pl8 = [make_prompt(2048) + f" Variant {i}." for i in range(8)]
    run_case("prefill_2048_bs8", pl8, 1)
    r = results["prefill_2048_bs8"]
    r["prefill_tok_per_s"] = round(r["in_tokens_total"] / r["wall_s"], 1)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
