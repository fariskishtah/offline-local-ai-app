import time
from dataclasses import dataclass
from typing import Generator, Callable
from ollama_client import OLLAMA_URL

import requests
import json


@dataclass
class RunStats:
    model: str = ""
    tokens_generated: int = 0
    tokens_in_prompt: int = 0
    tokens_per_second: float = 0.0
    time_to_first_token_s: float = 0.0
    generation_duration_s: float = 0.0
    prompt_duration_s: float = 0.0
    total_duration_s: float = 0.0
    load_duration_s: float = 0.0
    context_length: int = 0
    done: bool = False


def stream_with_stats(
    model: str,
    messages: list[dict],
    system_prompt: str,
    num_ctx: int = 4096,
    on_stats: Callable[[RunStats], None] | None = None,
) -> Generator[str, None, None]:

    parts = [f"System: {system_prompt}"]
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        parts.append(f"{role}: {msg['content']}")
    parts.append("Assistant:")
    prompt = "\n\n".join(parts)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_ctx": num_ctx},
    }

    stats = RunStats(model=model, context_length=num_ctx)
    start = time.perf_counter()
    first_token = False

    try:
        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=120,
        ) as r:
            r.raise_for_status()

            for line in r.iter_lines():
                if not line:
                    continue

                data = json.loads(line)

                token = data.get("response", "")
                if token:
                    if not first_token:
                        stats.time_to_first_token_s = time.perf_counter() - start
                        first_token = True
                    yield token

                if data.get("done"):
                    ns = 1_000_000_000

                    stats.tokens_generated = data.get("eval_count", 0)
                    stats.tokens_in_prompt = data.get("prompt_eval_count", 0)
                    stats.generation_duration_s = data.get("eval_duration", 0) / ns
                    stats.prompt_duration_s = data.get("prompt_eval_duration", 0) / ns
                    stats.total_duration_s = data.get("total_duration", 0) / ns
                    stats.load_duration_s = data.get("load_duration", 0) / ns
                    stats.done = True

                    if stats.generation_duration_s > 0:
                        stats.tokens_per_second = (
                            stats.tokens_generated / stats.generation_duration_s
                        )

                    if on_stats:
                        on_stats(stats)

                    break

    except requests.exceptions.ConnectionError:
        yield "\n\n[Error: Ollama is not running]"
    except requests.exceptions.Timeout:
        yield "\n\n[Error: Request timed out]"


def format_stats(s: RunStats) -> dict[str, str]:
    if not s.done:
        return {}

    return {
        "Tokens generated": str(s.tokens_generated),
        "Tokens per second": f"{s.tokens_per_second:.1f} tok/s",
        "Time to first token": f"{s.time_to_first_token_s:.2f}s",
        "Generation time": f"{s.generation_duration_s:.2f}s",
        "Prompt eval time": f"{s.prompt_duration_s:.2f}s",
        "Total wall time": f"{s.total_duration_s:.2f}s",
        "Model load time": f"{s.load_duration_s:.3f}s",
        "Context length": f"{s.context_length:,} tokens",
        "Prompt tokens": str(s.tokens_in_prompt),
    }