from pathlib import Path
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Config:
    # Read from environment variables with fallback defaults
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    default_model: str = os.getenv("DEFAULT_MODEL", "phi3:mini")
    default_num_ctx: int = int(os.getenv("DEFAULT_NUM_CTX", 4096))
    default_system_prompt: str = os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        "You are a helpful, concise AI assistant running locally on the user's Mac. Keep answers clear and practical."
    )
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", 120))
    db_path: Path = Path(os.getenv("DB_PATH", "chat_history.db"))
    log_dir: Path = Path(os.getenv("LOG_DIR", "logs"))
    app_title: str = os.getenv("APP_TITLE", "Local AI Chat")

    @property
    def ollama_generate_url(self) -> str:
        return f"{self.ollama_host}/api/generate"

    @property
    def ollama_tags_url(self) -> str:
        return f"{self.ollama_host}/api/tags"

cfg = Config()