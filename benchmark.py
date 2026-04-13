#!/usr/bin/env python3

import sys
import uuid
import argparse
from dataclasses import dataclass, asdict
from typing import Optional

from config import cfg
from ollama_client import list_models, is_running
from profiler import stream_with_stats, RunStats
from monitor import take_snapshot
from database import init_benchmarks_table, save_benchmark_result

BENCHMARK_PROMPTS = [
    {
        "id": "p1_short",
        "text": "What is the capital of France? Reply in one sentence.",
        "desc": "Short factual — tests TTFT",
    },
    {
        "id": "p2_medium",
        "text": (
            "Explain the difference between RAM and storage in simple terms. "
            "Use an analogy. Keep your answer to 3 short paragraphs."
        ),
        "desc": "Medium output — tests sustained tok/s",
    },
    {
        "id": "p3_list",
        "text": (
            "List 15 programming languages with one sentence describing "
            "each one's primary use case. Number them."
        ),
        "desc": "Long list — tests throughput at scale",
    },
]

SYSTEM_PROMPT = "You are a concise assistant. Answer directly without preamble."
NUM_CTX = 2048


@dataclass
class BenchmarkResult:
    run_id: str
    model: str
    prompt_id: str
    prompt_text: str

    tokens_per_second: float = 0.0
    time_to_first_token: float = 0.0
    tokens_generated: int = 0
    prompt_tokens: int = 0
    generation_duration: float = 0.0
    total_duration: float = 0.0

    ram_used_gb: float = 0.0
    ram_percent: float = 0.0
    cpu_percent: float = 0.0
    model_processor: str = "—"

    num_ctx: int = NUM_CTX
    status: str = "ok"
    error_msg: str = ""

    def to_db_dict(self) -> dict:
        d = asdict(self)
        return {k: (v if v is not None else "") for k, v in d.items()}


def _run_one(
    run_id: str,
    model: str,
    prompt: dict,
    quiet: bool = False,
) -> BenchmarkResult:
    result = BenchmarkResult(
        run_id=run_id,
        model=model,
        prompt_id=prompt["id"],
        prompt_text=prompt["text"],
    )

    stats_holder: dict[str, RunStats] = {}

    def capture(s: RunStats) -> None:
        stats_holder["s"] = s

    try:
        if not quiet:
            print(f"    [{prompt['id']}] ", end="", flush=True)

        tokens_seen = 0
        for _token in stream_with_stats(
            model=model,
            messages=[{"role": "user", "content": prompt["text"]}],
            system_prompt=SYSTEM_PROMPT,
            num_ctx=NUM_CTX,
            on_stats=capture,
        ):
            tokens_seen += 1
            if not quiet and tokens_seen % 20 == 0:
                print(".", end="", flush=True)

        if not quiet:
            print()

        if "s" in stats_holder:
            s = stats_holder["s"]
            result.tokens_per_second = round(s.tokens_per_second, 2)
            result.time_to_first_token = round(s.time_to_first_token_s, 3)
            result.tokens_generated = s.tokens_generated
            result.prompt_tokens = s.tokens_in_prompt
            result.generation_duration = round(s.generation_duration_s, 3)
            result.total_duration = round(s.total_duration_s, 3)

        snap = take_snapshot()
        result.ram_used_gb = round(snap.ram_used_gb, 2)
        result.ram_percent = round(snap.ram_percent, 1)
        result.cpu_percent = round(snap.cpu_percent, 1)
        result.model_processor = snap.model_processor

    except Exception as exc:
        result.status = "error"
        result.error_msg = str(exc)[:200]
        if not quiet:
            print(f" ERROR: {exc}")

    return result


def run_benchmark(
    models: Optional[list[str]] = None,
    quiet: bool = False,
    run_id: Optional[str] = None,
) -> tuple[str, list[BenchmarkResult]]:
    run_id = run_id or str(uuid.uuid4())

    init_benchmarks_table()

    if models is None:
        models = list_models()

    if not models:
        print("No models found. Run: ollama pull phi3:mini")
        return run_id, []

    total_runs = len(models) * len(BENCHMARK_PROMPTS)
    all_results: list[BenchmarkResult] = []

    if not quiet:
        print()
        print("=" * 60)
        print(f"  Benchmark run  {run_id[:8]}…")
        print(f"  Models:  {', '.join(models)}")
        print(f"  Prompts: {len(BENCHMARK_PROMPTS)}  ×  num_ctx={NUM_CTX}")
        print(f"  Total:   {total_runs} inference calls")
        print("=" * 60)

    for model in models:
        if not quiet:
            print(f"\n  Model: {model}")

        if not quiet:
            print("    [warmup] loading model… ", end="", flush=True)

        warmup_done = False
        try:
            for _ in stream_with_stats(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                system_prompt="Be brief.",
                num_ctx=NUM_CTX,
            ):
                pass
            warmup_done = True
        except Exception:
            pass

        if not quiet:
            print("ready" if warmup_done else "skipped (error)")

        for prompt in BENCHMARK_PROMPTS:
            result = _run_one(run_id, model, prompt, quiet=quiet)
            all_results.append(result)
            save_benchmark_result(result.to_db_dict())

    if not quiet:
        _print_report(all_results, models)

    return run_id, all_results


def _print_report(results: list[BenchmarkResult], models: list[str]) -> None:
    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)

    agg = {}
    for r in results:
        if r.status != "ok":
            continue

        if r.model not in agg:
            agg[r.model] = {
                "tok_s": [],
                "ttft": [],
                "tokens": [],
                "ram": [],
                "cpu": [],
                "proc": r.model_processor,
            }

        agg[r.model]["tok_s"].append(r.tokens_per_second)
        agg[r.model]["ttft"].append(r.time_to_first_token)
        agg[r.model]["tokens"].append(r.tokens_generated)
        agg[r.model]["ram"].append(r.ram_used_gb)
        agg[r.model]["cpu"].append(r.cpu_percent)

    if not agg:
        print("  No successful results to report.")
        return

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    ranked = sorted(
        agg.items(),
        key=lambda x: avg(x[1]["tok_s"]),
        reverse=True,
    )

    print(f"\n  {'Rank':<5} {'Model':<22} {'tok/s':>7} {'TTFT':>7} {'Tokens':>7} {'RAM GB':>7} {'Processor':<14}")
    print(f"  {'-'*5} {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*14}")

    for rank, (model, data) in enumerate(ranked, start=1):
        tok_s = avg(data["tok_s"])
        ttft = avg(data["ttft"])
        tokens = avg(data["tokens"])
        ram = avg(data["ram"])
        proc = data["proc"]

        medal_map = {1: "  WINNER", 2: "  2nd", 3: "  3rd"}
        medal = medal_map.get(rank, "")

        print(
            f"  {rank:<5} {model:<22} {tok_s:>7.1f} {ttft:>7.2f} "
            f"{int(tokens):>7} {ram:>7.1f} {proc:<14}{medal}"
        )

    print()
    print("  Per-prompt breakdown:")
    print()

    for prompt in BENCHMARK_PROMPTS:
        pid = prompt["id"]
        print(f"  {prompt['desc']}")
        prompt_results = [r for r in results if r.prompt_id == pid and r.status == "ok"]
        prompt_results.sort(key=lambda r: r.tokens_per_second, reverse=True)

        for r in prompt_results:
            bar_width = int(r.tokens_per_second / 2)
            bar = "█" * min(bar_width, 30)
            print(f"    {r.model:<22} {r.tokens_per_second:>6.1f} tok/s  {bar}")
        print()

    errors = [r for r in results if r.status == "error"]
    if errors:
        print("  Errors:")
        for r in errors:
            print(f"    {r.model} / {r.prompt_id}: {r.error_msg}")
        print()

    print(f"  Results saved to: {cfg.db_path}")
    print("=" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark all locally installed Ollama models."
    )
    parser.add_argument(
        "--model", "-m",
        help="Benchmark a single specific model (e.g. phi3:mini). Default: all installed models.",
        default=None,
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output. Results table still prints.",
    )
    args = parser.parse_args()

    if not is_running():
        print("\n  ERROR: Ollama is not running.")
        print("  Start it with:  open -a Ollama")
        sys.exit(1)

    models = [args.model] if args.model else None

    run_id, results = run_benchmark(models=models, quiet=args.quiet)

    if not results:
        sys.exit(1)

    ok_count = sum(1 for r in results if r.status == "ok")
    err_count = sum(1 for r in results if r.status == "error")
    print(f"  Done. {ok_count} runs succeeded, {err_count} failed.")
    print(f"  Run ID: {run_id}")


if __name__ == "__main__":
    main()