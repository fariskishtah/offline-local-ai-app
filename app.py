import streamlit as st
import pandas as pd

from config import cfg
from ollama_client import is_running, list_models
from profiler import stream_with_stats, format_stats
from monitor import take_snapshot, format_snapshot as format_sys
from ctx_tuner import run_ctx_benchmark
from database import (
    init_db,
    create_session,
    get_all_sessions,
    save_message,
    load_messages,
    delete_session,
    auto_title_from_first_message,
    get_ctx_runs,
    get_ctx_results,
)

# ── Boot ─────────────────────────────────────────────────────
init_db()

st.set_page_config(
    page_title=cfg.app_title,
    page_icon="🤖",
    layout="wide",
)

# ── Session State ─────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = cfg.default_model

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = cfg.default_system_prompt

if "last_stats" not in st.session_state:
    st.session_state.last_stats = None

if "num_ctx" not in st.session_state:
    st.session_state.num_ctx = cfg.default_num_ctx

if "last_snapshot" not in st.session_state:
    st.session_state.last_snapshot = None

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title(cfg.app_title)

    running = is_running()
    models = list_models() if running else []

    if running:
        st.success("Ollama running ✅")
    else:
        st.error("Ollama offline — start Ollama on your Mac first")

    st.caption(f"Host: `{cfg.ollama_host}`")

    with st.expander("Debug connection"):
        st.write("Running:", running)
        st.write("Models:", models if models else "[]")

    st.divider()

    if models:
        current_index = models.index(st.session_state.model) if st.session_state.model in models else 0
        st.session_state.model = st.selectbox(
            "Model",
            options=models,
            index=current_index,
        )
    else:
        st.warning("No models found. Run: ollama pull phi3:mini")

    st.session_state.system_prompt = st.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        height=100,
    )

    st.divider()

    st.session_state.num_ctx = st.select_slider(
        "Context window",
        options=[512, 1024, 2048, 4096, 8192],
        value=st.session_state.num_ctx,
    )

    if st.button("🔄 Refresh status", use_container_width=True):
        st.rerun()

    # ── Performance stats ─────────────────────────────────────
    if st.session_state.last_stats:
        stats = format_stats(st.session_state.last_stats)
        st.divider()
        st.markdown("**Performance**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("tok/s", stats["Tokens per second"])
            st.metric("Gen time", stats["Generation time"])
        with col2:
            st.metric("First token", stats["Time to first token"])
            st.metric("Tokens out", stats["Tokens generated"])

    # ── System monitor ────────────────────────────────────────
    st.divider()
    st.markdown("**System**")
    snap = st.session_state.last_snapshot
    if snap:
        sys_fmt = format_sys(snap)
        st.progress(int(snap.ram_percent), text=f"RAM {sys_fmt['RAM used']}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU", sys_fmt["CPU"])
            st.metric("Processor", sys_fmt["Processor"])
        with col2:
            st.metric("Available", sys_fmt["RAM free"])
            st.metric("Ollama RAM", sys_fmt["Ollama RAM"])
        if "CPU" in snap.model_processor and snap.active_model != "none":
            st.warning("Model is on CPU, not GPU — performance will be slow.", icon="⚠️")
    else:
        st.caption("Send a message to see resource usage.")

    st.divider()

    # ── Session controls ──────────────────────────────────────
    if st.button("＋ New Chat", use_container_width=True, type="primary", disabled=not running):
        sid = create_session(st.session_state.model)
        st.session_state.session_id = sid
        st.session_state.messages = []
        st.rerun()

    st.markdown("**Chats**")
    sessions = get_all_sessions()
    if not sessions:
        st.caption("No saved chats yet.")

    for s in sessions:
        col1, col2 = st.columns([4, 1])
        with col1:
            label = ("▶ " if s["id"] == st.session_state.session_id else "") + s["title"]
            if st.button(label, key=f"load_{s['id']}", use_container_width=True):
                st.session_state.session_id = s["id"]
                st.session_state.messages = load_messages(s["id"])
                st.rerun()
        with col2:
            if st.button("✕", key=f"del_{s['id']}"):
                delete_session(s["id"])
                if st.session_state.session_id == s["id"]:
                    st.session_state.session_id = None
                    st.session_state.messages = []
                st.rerun()

# ── Main content ─────────────────────────────────────────────
if not running:
    st.title(cfg.app_title)
    st.warning("Ollama is not reachable from the app.")
    st.code(f"Current host: {cfg.ollama_host}")
    st.info("Start Ollama, then click Refresh status in the sidebar.")
    st.stop()

if st.session_state.session_id is None:
    st.title(cfg.app_title)
    st.info("Click **＋ New Chat** in the sidebar to begin.", icon="👈")
    st.stop()

tab_chat, tab_ctx = st.tabs(["Chat", "Context Tuning"])

# ══════════════════════════════════════════════════════════════
# CHAT TAB
# ══════════════════════════════════════════════════════════════
with tab_chat:
    st.title(f"Chat #{st.session_state.session_id}")
    st.caption(f"Model: `{st.session_state.model}` · Host: `{cfg.ollama_host}`")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask anything"):
        if len(st.session_state.messages) == 0:
            auto_title_from_first_message(st.session_state.session_id, user_input)

        save_message(st.session_state.session_id, "user", user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            stats_holder = {}

            def capture(s):
                stats_holder["stats"] = s

            full = st.write_stream(
                stream_with_stats(
                    model=st.session_state.model,
                    messages=st.session_state.messages,
                    system_prompt=st.session_state.system_prompt,
                    num_ctx=st.session_state.num_ctx,
                    on_stats=capture,
                )
            )

            if "stats" in stats_holder:
                st.session_state.last_stats = stats_holder["stats"]

        save_message(st.session_state.session_id, "assistant", full)
        st.session_state.messages.append({"role": "assistant", "content": full})

        st.session_state.last_snapshot = take_snapshot()

        if len(st.session_state.messages) == 2:
            st.rerun()

# ══════════════════════════════════════════════════════════════
# CONTEXT TUNING TAB
# ══════════════════════════════════════════════════════════════
with tab_ctx:
    st.title("Context window tuning")
    st.caption(
        "Measures how tok/s, TTFT, and RAM change as num_ctx increases."
    )

    col_model, col_ctx, col_btn = st.columns([2, 3, 1])

    with col_model:
        ctx_model = st.selectbox(
            "Model to tune",
            options=models or [st.session_state.model],
            key="ctx_model_select",
        )

    with col_ctx:
        ctx_values = st.multiselect(
            "num_ctx values to test",
            options=[512, 1024, 2048, 4096, 8192],
            default=[512, 1024, 2048, 4096],
            key="ctx_values_select",
        )

    with col_btn:
        st.write("")
        run_ctx_btn = st.button(
            "Run",
            type="primary",
            use_container_width=True,
            disabled=not ctx_values,
        )

    if run_ctx_btn and ctx_values:
        with st.spinner(f"Testing {ctx_model} at {len(ctx_values)} ctx values…"):
            run_id, results = run_ctx_benchmark(
                models=[ctx_model],
                ctx_values=sorted(ctx_values),
                quiet=True,
            )
        ok_count = sum(1 for r in results if r.status == "ok")
        st.success(f"Done! {ok_count}/{len(results)} runs succeeded.")
        st.rerun()

    st.divider()

    ctx_runs = get_ctx_runs()
    if not ctx_runs:
        st.info("No context tuning runs yet. Select a model above and click Run.")
        st.stop()

    run_options = {
        f"{r['model']} · ctx {r['ctx_min']}–{r['ctx_max']} · {r['started_at'][:16]}": (r["run_id"], r["model"])
        for r in ctx_runs
    }

    chosen_label = st.selectbox(
        "Show run",
        options=list(run_options.keys()),
        key="ctx_run_select"
    )
    chosen_run_id, chosen_model = run_options[chosen_label]

    rows = get_ctx_results(chosen_run_id, chosen_model)
    if not rows:
        st.warning("No results found for this run.")
        st.stop()

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        st.error("All runs errored. Check Ollama and retry.")
        st.stop()

    df = pd.DataFrame(ok_rows).sort_values("num_ctx")

    best_row = df.loc[df["tokens_per_second"].idxmax()]
    worst_row = df.loc[df["tokens_per_second"].idxmin()]
    speed_drop = (
        (best_row["tokens_per_second"] - worst_row["tokens_per_second"])
        / best_row["tokens_per_second"] * 100
        if best_row["tokens_per_second"] > 0 else 0
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Fastest", f"{best_row['tokens_per_second']:.1f} tok/s", help=f"ctx={int(best_row['num_ctx'])}")
    with m2:
        st.metric("Slowest", f"{worst_row['tokens_per_second']:.1f} tok/s", help=f"ctx={int(worst_row['num_ctx'])}")
    with m3:
        st.metric("Speed drop", f"{speed_drop:.0f}%")
    with m4:
        st.metric("Peak RAM", f"{df['ram_used_gb'].max():.1f} GB")

    st.divider()
    st.markdown(f"**{chosen_model} — speed vs context window**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Tokens per second")
        st.line_chart(df.set_index("num_ctx")[["tokens_per_second"]].rename(columns={"tokens_per_second": "tok/s"}))
    with c2:
        st.markdown("Time to first token (s)")
        st.line_chart(df.set_index("num_ctx")[["time_to_first_token"]].rename(columns={"time_to_first_token": "TTFT (s)"}))

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("Prompt eval time (s)")
        st.line_chart(df.set_index("num_ctx")[["prompt_eval_duration"]].rename(columns={"prompt_eval_duration": "prompt eval (s)"}))
    with c4:
        st.markdown("RAM used (GB)")
        st.line_chart(df.set_index("num_ctx")[["ram_used_gb"]].rename(columns={"ram_used_gb": "RAM (GB)"}))

    st.divider()
    st.markdown("**Recommendation**")

    max_tok_s = df["tokens_per_second"].max()
    sweet_df = df[df["tokens_per_second"] >= max_tok_s * 0.90]
    sweet_ctx = int(sweet_df["num_ctx"].max()) if not sweet_df.empty else int(df["num_ctx"].min())
    sweet_row = df[df["num_ctx"] == sweet_ctx].iloc[0]

    st.success(
        f"**Recommended num_ctx for {chosen_model}: `{sweet_ctx}`**  \n"
        f"Delivers `{sweet_row['tokens_per_second']:.1f} tok/s` — within 10% of peak speed, "
        f"`{sweet_row['time_to_first_token']:.3f}s` TTFT, `{sweet_row['ram_used_gb']:.1f} GB` RAM."
    )

    if st.button(f"Apply num_ctx={sweet_ctx} to chat"):
        st.session_state.num_ctx = sweet_ctx
        st.success(f"Context window set to {sweet_ctx}. Switch to the Chat tab.")

    with st.expander("Raw data"):
        display_cols = {
            "num_ctx": "num_ctx",
            "tokens_per_second": "tok/s",
            "time_to_first_token": "TTFT (s)",
            "prompt_eval_duration": "prompt eval (s)",
            "generation_duration": "gen (s)",
            "total_duration": "total (s)",
            "tokens_generated": "tokens out",
            "prompt_tokens": "prompt tokens",
            "ram_used_gb": "RAM (GB)",
            "ram_percent": "RAM %",
            "model_processor": "processor",
        }
        display_df = df[[c for c in display_cols if c in df.columns]].rename(columns=display_cols)
        st.dataframe(display_df, use_container_width=True, hide_index=True)