# Offline Local AI App

An offline-first local AI application built on Mac using Ollama and small language models.

## Overview

This project is a practical local AI system that runs without relying on cloud APIs.

It includes:
- a local chat interface built with Streamlit
- persistent chat history with SQLite
- session-based conversation management
- performance profiling
- RAM / CPU monitoring
- model benchmarking
- context window tuning

The app is designed for experimenting with local small language models in a real working setup.

---

## Features

- Local AI chat UI with Streamlit
- Ollama integration for running local models
- Persistent chat sessions with SQLite
- Performance metrics:
  - tokens per second
  - time to first token
  - generation time
  - prompt tokens
- System monitoring:
  - RAM usage
  - CPU usage
  - active model / processor info
- Model benchmarking across installed local models
- Context window tuning and comparison

---

## Tech Stack

- Python
- Streamlit
- Ollama
- SQLite
- Pandas
- psutil

---

## Project Structure

```bash
local-ai-app/
├── app.py
├── ollama_client.py
├── database.py
├── profiler.py
├── monitor.py
├── benchmark.py
├── ctx_tuner.py
├── config.py
├── .env.example
├── requirements.txt
├── chat_history.db
└── venv/
