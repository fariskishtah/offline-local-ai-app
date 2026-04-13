# ollama_client.py
import requests
import json
from typing import Generator

OLLAMA_URL  = "http://localhost:11434"


def is_running() -> bool:
    """Returns True if Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_URL}", timeout=2)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def list_models() -> list[str]:
    """Returns list of locally installed model names."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = r.json().get("models", [])
        return [m["name"] for m in models]
    except Exception:
        return []


def stream_response(
    model: str,
    messages: list[dict],
    system_prompt: str,
) -> Generator[str, None, None]:
    """
    Streams tokens from Ollama one at a time.
    `messages` is a list of {"role": "user"|"assistant", "content": "..."}
    Yields each token string as it arrives.
    """
    # Build the full prompt from message history
    prompt_parts = [f"System: {system_prompt}\n"]
    for msg in messages:
        role  = "User" if msg["role"] == "user" else "Assistant"
        prompt_parts.append(f"{role}: {msg['content']}")
    prompt_parts.append("Assistant:")
    full_prompt = "\n\n".join(prompt_parts)

    payload = {
        "model":  model,
        "prompt": full_prompt,
        "stream": True,
    }

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
                data  = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield "\n\n[Error: Cannot reach Ollama. Is it running?]"
    except requests.exceptions.Timeout:
        yield "\n\n[Error: Request timed out. Try a shorter message.]"