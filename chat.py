import requests
import json
import sys

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"
SYSTEM = "You are a helpful, concise AI assistant running locally on the user's Mac. Keep answers short and practical."

def ask(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": f"System: {SYSTEM}\n\nUser: {prompt}\n\nAssistant:",
        "stream": True,
    }

    full_response = ""

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("response", "")
                print(token, end="", flush=True)
                full_response += token
                if data.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot reach Ollama. Is it running?")
        print("Run: open -a Ollama")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("\n[ERROR] Request timed out.")
        sys.exit(1)

    print()
    return full_response

def main():
    print(f"\nLocal AI Chat | model: {MODEL}")
    print("Type 'quit' or Ctrl+C to exit\n")
    print("-" * 42)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\n\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        print("\nAI: ", end="")
        ask(user_input)

if __name__ == "__main__":
    main()