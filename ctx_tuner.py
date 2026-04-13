#!/usr/bin/env python3
# ctx_tuner.py
#
# Run standalone:
#   python ctx_tuner.py                    # tests all models, default ctx values
#   python ctx_tuner.py --model gemma2:2b  # test one model
#   python ctx_tuner.py --model gemma2:2b --ctx 512 1024 2048 4096
#
# Tests one or more models across a range of num_ctx values using a fixed
# prompt. Saves results to the ctx_benchmarks SQLite table and prints a
# tradeoff table. Reuses profiler.py, monitor.py, ollama_client.py, database.py.

import sys
import uuid
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

from config        import cfg
from ollama_client import list_models, is_running
from profiler      import stream_with_stats, RunStats
from monitor       import take_snapshot
from database      import init_ctx_table, save_ctx_result


# ── Default num_ctx ladder ───────────────────────────────────
#
# Why these values?
#   512  — minimum useful context; fastest possible baseline
#  1024  — enough for a short multi-turn conversation
#  2048  — sweet spot for most chat applications
#  4096  — default Ollama value; comfortable for long documents
#  8192  — large context; tests whether your Mac has enough RAM
#
# Values above 8192 require models that support them (Llama 3 and
# some Mistral variants do; phi3:mini caps at 4096).

DEFAULT_CTX_VALUES = [512, 1024, 2048, 4096, 8192]

# Fixed prompt — long enough to generate meaningful output but
# identical across all runs so only num_ctx varies.
CTX_PROMPT = (
    "Explain how a computer processor executes instructions, "
    "starting from fetching an instruction from memory through "
    "to writing the result back. Cover the fetch-decode-execute cycle "
    "in detail. Write at least 200 words."
)
SYSTEM_PROMPT = "You are a clear, technical educator. Be thorough."


# ── Result dataclass ─────────────────────────────────────────

@dataclass
class CtxResult:
    """One result row: one model at one num_ctx value."""
    run_id:               str
    model:                str
    num_ctx:              int

    # From profiler
    tokens_per_second:    float = 0.0
    time_to_first_token:  float = 0.0
    tokens_generated:     int   = 0
    prompt_tokens:        int   = 0
    generation_duration:  float = 0.0
    prompt_eval_duration: float = 0.0   # how long Ollama spent reading the prompt
    total_duration:       float = 0.0

    # From monitor
    ram_used_gb:          float = 0.0
    ram_percent:          float = 0.0
    cpu_percent:          float = 0.0
    model_processor:      str   = "—"

    status:               str   = "ok"
    error_msg:            str   = ""

    def to_db_dict(self) -> dict:
        d = asdict(self)
        return {k: (v if v is not None else "") for k, v in d.items()}


# ── Core run logic ───────────────────────────────────────────

def _run_one_ctx(
    run_id:  str,
    model:   str,
    num_ctx: int,
    quiet:   bool = False,
) -> CtxResult:
    """
    Runs one model at one num_ctx value and returns a populated CtxResult.
    Errors are caught and recorded — never crash the outer loop.
    """
    result = CtxResult(run_id=run_id, model=model, num_ctx=num_ctx)
    stats_holder: dict[str, RunStats] = {}

    def capture(s: RunStats) -> None:
        stats_holder["s"] = s

    try:
        if not quiet:
            print(f"    ctx={num_ctx:>5}  ", end="", flush=True)

        token_count = 0
        for _ in stream_with_stats(
            model         = model,
            messages      = [{"role": "user", "content": CTX_PROMPT}],
            system_prompt = SYSTEM_PROMPT,
            num_ctx       = num_ctx,
            on_stats      = capture,
        ):
            token_count += 1
            if not quiet and token_count % 25 == 0:
                print(".", end="", flush=True)

        if not quiet:
            print()

        if "s" in stats_holder:
            s = stats_holder["s"]
            result.tokens_per_second    = round(s.tokens_per_second,      2)
            result.time_to_first_token  = round(s.time_to_first_token_s,  3)
            result.tokens_generated     = s.tokens_generated
            result.prompt_tokens        = s.tokens_in_prompt
            result.generation_duration  = round(s.generation_duration_s,  3)
            result.prompt_eval_duration = round(s.prompt_duration_s,      3)
            result.total_duration       = round(s.total_duration_s,       3)

        snap = take_snapshot()
        result.ram_used_gb     = round(snap.ram_used_gb, 2)
        result.ram_percent     = round(snap.ram_percent,  1)
        result.cpu_percent     = round(snap.cpu_percent,  1)
        result.model_processor = snap.model_processor

    except Exception as exc:
        result.status    = "error"
        result.error_msg = str(exc)[:200]
        if not quiet:
            print(f"  ERROR: {exc}")

    return result


def run_ctx_benchmark(
    models:      Optional[list[str]] = None,
    ctx_values:  Optional[list[int]] = None,
    quiet:       bool = False,
    run_id:      Optional[str] = None,
) -> tuple[str, list[CtxResult]]:
    """
    Main entry point. Callable from app.py or the CLI.

    Args:
        models:     List of model names. None = all installed models.
        ctx_values: List of num_ctx integers to test.
                    None = DEFAULT_CTX_VALUES.
        quiet:      Suppress progress dots.
        run_id:     Optional UUID. Auto-generated if not provided.

    Returns:
        (run_id, list[CtxResult])

    Every result is also written to SQLite immediately after collection.
    """
    run_id     = run_id or str(uuid.uuid4())
    ctx_values = ctx_values or DEFAULT_CTX_VALUES

    init_ctx_table()   # safe no-op if table already exists

    if models is None:
        models = list_models()
    if not models:
        print("No models found. Run: ollama pull phi3:mini")
        return run_id, []

    all_results: list[CtxResult] = []

    if not quiet:
        print()
        print("=" * 62)
        print(f"  Context window tuning  {run_id[:8]}…")
        print(f"  Models:     {', '.join(models)}")
        print(f"  ctx values: {ctx_values}")
        print(f"  Prompt:     {len(CTX_PROMPT)} chars  (~{len(CTX_PROMPT)//4} tokens)")
        print("=" * 62)

    for model in models:
        if not quiet:
            print(f"\n  Model: {model}")
            print(f"  {'ctx':>6}  {'tok/s':>7}  {'TTFT':>7}  "
                  f"{'prompt_s':>8}  {'gen_s':>6}  {'RAM GB':>6}")
            print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*6}  {'-'*6}")

        for num_ctx in ctx_values:
            result = _run_one_ctx(run_id, model, num_ctx, quiet=quiet)
            all_results.append(result)
            save_ctx_result(result.to_db_dict())

            # Print inline result row immediately — don't wait for the loop
            if not quiet and result.status == "ok":
                print(
                    f"  {result.num_ctx:>6}  "
                    f"{result.tokens_per_second:>7.1f}  "
                    f"{result.time_to_first_token:>7.3f}  "
                    f"{result.prompt_eval_duration:>8.3f}  "
                    f"{result.generation_duration:>6.2f}  "
                    f"{result.ram_used_gb:>6.1f}"
                )
            elif not quiet and result.status == "error":
                print(
                    f"  {result.num_ctx:>6}  "
                    f"{'ERROR':>7}  —  —  —  —  {result.error_msg[:30]}"
                )

    if not quiet:
        _print_summary(all_results, models, ctx_values)

    return run_id, all_results


# ── Terminal summary ─────────────────────────────────────────

def _print_summary(
    results:    list[CtxResult],
    models:     list[str],
    ctx_values: list[int],
) -> None:
    """Prints recommendation and speed-drop % for each model."""
    print()
    print("=" * 62)
    print("  Speed vs context summary")
    print("=" * 62)

    for model in models:
        model_rows = [r for r in results
                      if r.model == model and r.status == "ok"]
        if not model_rows:
            continue

        model_rows.sort(key=lambda r: r.num_ctx)
        baseline = model_rows[0].tokens_per_second  # fastest = smallest ctx

        print(f"\n  {model}")
        print(f"  {'ctx':>6}  {'tok/s':>7}  {'vs baseline':>12}  "
              f"{'TTFT':>7}  {'prompt_s':>8}")
        print(f"  {'-'*6}  {'-'*7}  {'-'*12}  {'-'*7}  {'-'*8}")

        for r in model_rows:
            if baseline > 0:
                delta_pct = ((r.tokens_per_second - baseline) / baseline) * 100
                delta_str = f"{delta_pct:+.0f}%"
            else:
                delta_str = "—"

            # Flag: is this ctx value the best value-for-speed tradeoff?
            # Heuristic: first ctx where tok/s drops less than 10% from baseline
            flag = ""
            if abs(delta_pct if baseline > 0 else 0) < 10 and r.num_ctx >= 2048:
                flag = "  ← sweet spot"

            print(
                f"  {r.num_ctx:>6}  "
                f"{r.tokens_per_second:>7.1f}  "
                f"{delta_str:>12}  "
                f"{r.time_to_first_token:>7.3f}  "
                f"{r.prompt_eval_duration:>8.3f}"
                f"{flag}"
            )

    print()
    print(f"  Results saved to: {cfg.db_path}")
    print(f"  View in Streamlit → Context Tuning tab")
    print("=" * 62)
    print()


# ── CLI ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark one or more Ollama models across num_ctx values."
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Model to test (e.g. gemma2:2b). Default: all installed models.",
    )
    parser.add_argument(
        "--ctx", nargs="+", type=int, default=None,
        help=f"num_ctx values to test. Default: {DEFAULT_CTX_VALUES}",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args()

    if not is_running():
        print("\n  ERROR: Ollama is not running.")
        print("  Start it:  open -a Ollama")
        sys.exit(1)

    models     = [args.model] if args.model else None
    ctx_values = args.ctx or DEFAULT_CTX_VALUES

    run_id, results = run_ctx_benchmark(
        models=models, ctx_values=ctx_values, quiet=args.quiet
    )

    ok  = sum(1 for r in results if r.status == "ok")
    err = sum(1 for r in results if r.status == "error")
    print(f"  Done. {ok} runs succeeded, {err} failed.")
    print(f"  Run ID: {run_id}")


if __name__ == "__main__":
    main()