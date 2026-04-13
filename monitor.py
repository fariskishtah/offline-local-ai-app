# monitor.py
# Captures a full system resource snapshot in < 150ms.
# Designed to be called once per chat response — not on a timer.
# Never import this at module level in a hot path; import at call site.

import os
import subprocess
import psutil
from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemSnapshot:
    # ── System-wide ──────────────────────────────────────────
    ram_used_gb:     float   # bytes in use across whole machine
    ram_total_gb:    float   # physical RAM installed
    ram_percent:     float   # percentage used (0–100)
    ram_available_gb: float  # what macOS will give a new process

    # ── CPU ──────────────────────────────────────────────────
    cpu_percent:     float   # system-wide 1-second sample
    cpu_core_count:  int     # physical cores (not logical)

    # ── This Python / Streamlit process ──────────────────────
    app_ram_mb:      float   # RSS memory of this process
    app_cpu_percent: float   # CPU % used by this process

    # ── Ollama process ───────────────────────────────────────
    ollama_ram_mb:   float   # RSS memory of Ollama daemon
    ollama_pid:      Optional[int]

    # ── Active model (from `ollama ps`) ──────────────────────
    active_model:    str     # model name or "none"
    model_size_gb:   float   # size reported by ollama ps
    model_processor: str     # "100% GPU", "100% CPU", etc.
    ollama_ps_raw:   str     # raw output for debugging


# ── Internal helpers ─────────────────────────────────────────

def _find_ollama_process() -> Optional[psutil.Process]:
    """Scan running processes for the Ollama server daemon."""
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            if "ollama" in proc.info["name"].lower():
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def _parse_ollama_ps() -> tuple[str, str, float, str]:
    """
    Runs `ollama ps` and parses the first model row.

    ollama ps output looks like:
      NAME            ID        SIZE    PROCESSOR    UNTIL
      phi3:mini       abc123    2.2 GB  100% GPU     forever

    Returns: (raw_output, model_name, size_gb, processor_str)
    """
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        raw = result.stdout.strip()

        if not raw or "NAME" not in raw:
            return raw, "none", 0.0, "—"

        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if len(lines) < 2:
            return raw, "none", 0.0, "—"

        # Parse data row — columns are space-separated but SIZE has a space
        # e.g.: "phi3:mini   abc123   2.2 GB   100% GPU   4 minutes from now"
        parts = lines[1].split()
        if len(parts) < 4:
            return raw, parts[0] if parts else "unknown", 0.0, "—"

        model_name = parts[0]

        # Find SIZE — look for a float followed by "GB" or "MB"
        size_gb = 0.0
        for i, token in enumerate(parts):
            if token in ("GB", "gb") and i > 0:
                try:
                    size_gb = float(parts[i - 1])
                except ValueError:
                    pass
            elif token in ("MB", "mb") and i > 0:
                try:
                    size_gb = float(parts[i - 1]) / 1024
                except ValueError:
                    pass

        # Find PROCESSOR — look for "%" token
        processor = "—"
        for i, token in enumerate(parts):
            if "%" in token and i + 1 < len(parts):
                processor = f"{token} {parts[i + 1]}"
                break

        return raw, model_name, size_gb, processor

    except FileNotFoundError:
        return "ollama not in PATH", "none", 0.0, "—"
    except subprocess.TimeoutExpired:
        return "ollama ps timed out", "none", 0.0, "—"
    except Exception as e:
        return str(e), "none", 0.0, "—"


# ── Public API ───────────────────────────────────────────────

def take_snapshot() -> SystemSnapshot:
    """
    Captures all resource metrics in a single call.
    Takes ~100–150ms due to the cpu_percent(interval=0.1) sample.
    Call once after each model response completes — not during streaming.
    """
    vm  = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)  # 100ms sample — minimum reliable

    # This Streamlit process
    this_proc     = psutil.Process(os.getpid())
    app_ram_mb    = this_proc.memory_info().rss / (1024 ** 2)
    app_cpu       = this_proc.cpu_percent(interval=None)

    # Ollama daemon process
    ollama_proc   = _find_ollama_process()
    ollama_ram_mb = 0.0
    ollama_pid    = None
    if ollama_proc:
        try:
            ollama_ram_mb = ollama_proc.memory_info().rss / (1024 ** 2)
            ollama_pid    = ollama_proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Active model from ollama ps
    ps_raw, active_model, model_size_gb, model_processor = _parse_ollama_ps()

    return SystemSnapshot(
        ram_used_gb      = vm.used      / (1024 ** 3),
        ram_total_gb     = vm.total     / (1024 ** 3),
        ram_percent      = vm.percent,
        ram_available_gb = vm.available / (1024 ** 3),
        cpu_percent      = cpu,
        cpu_core_count   = psutil.cpu_count(logical=False) or 0,
        app_ram_mb       = app_ram_mb,
        app_cpu_percent  = app_cpu,
        ollama_ram_mb    = ollama_ram_mb,
        ollama_pid       = ollama_pid,
        active_model     = active_model,
        model_size_gb    = model_size_gb,
        model_processor  = model_processor,
        ollama_ps_raw    = ps_raw,
    )


def ram_health(snap: SystemSnapshot) -> str:
    """
    Returns 'ok', 'warn', or 'critical' based on available RAM.
    Used to colour the sidebar indicator.

    ok       → > 2 GB available  (safe to run any small model)
    warn     → 1–2 GB available  (model may slow down)
    critical → < 1 GB available  (swap likely, performance will tank)
    """
    avail = snap.ram_available_gb
    if avail > 2.0:
        return "ok"
    elif avail > 1.0:
        return "warn"
    return "critical"


def format_snapshot(snap: SystemSnapshot) -> dict[str, str]:
    """
    Human-readable strings for each metric.
    All floats are pre-rounded — never display raw floats in UI.
    """
    health = ram_health(snap)
    health_icon = {"ok": "●", "warn": "◐", "critical": "○"}[health]

    return {
        "RAM used":      f"{snap.ram_used_gb:.1f} / {snap.ram_total_gb:.0f} GB",
        "RAM free":      f"{snap.ram_available_gb:.1f} GB  {health_icon}",
        "RAM %":         f"{snap.ram_percent:.0f}%",
        "CPU":           f"{snap.cpu_percent:.0f}%  ({snap.cpu_core_count} cores)",
        "Ollama RAM":    f"{snap.ollama_ram_mb:.0f} MB"
                         if snap.ollama_ram_mb > 0 else "not found",
        "Active model":  snap.active_model,
        "Model size":    f"{snap.model_size_gb:.1f} GB"
                         if snap.model_size_gb > 0 else "—",
        "Processor":     snap.model_processor,
        "App RAM":       f"{snap.app_ram_mb:.0f} MB",
    }