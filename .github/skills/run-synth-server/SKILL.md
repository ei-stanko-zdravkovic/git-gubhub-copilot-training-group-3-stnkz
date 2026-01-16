---
name: run-synth-server
description: Instructions for starting the MCP Synth server (FastAPI) for synthetic data generation and tools.
---

# Run Synth Server

The MCP Synth Server is a FastAPI application that exposes tools for generating synthetic airline data, inspecting models, previewing data, and exporting archives.

## Prerequisites

1. **Python environment must be active**: The `.venv` environment should be activated.
2. **Dependencies installed**: `uvicorn` and `fastapi` must be installed (included in dev dependencies).

   Verify with:
   ```bash
   python -c "import uvicorn; import fastapi; print('OK')"
   ```

   If not installed, run from `airline-discount-ml/`:
   ```bash
   pip install -e ".[dev]"
   ```

## Standard Server Start Procedure

Run the following commands in order:

1. **Navigate to Project Root**:
   ```bash
   cd airline-discount-ml
   ```

2. **Start the Server (Background Mode - Recommended)**:
   
   Use `nohup` to run the server in the background so it persists when new terminal commands are run:
   ```bash
   nohup uvicorn src.mcp_synth.server:app --host 127.0.0.1 --port 8010 --reload > synth_server.log 2>&1 &
   ```

   This will:
   - Run the server in the background (`&`)
   - Keep it running even after terminal commands (`nohup`)
   - Log output to `synth_server.log`

   **Alternative: Foreground Mode** (blocks terminal):
   ```bash
   uvicorn src.mcp_synth.server:app --host 127.0.0.1 --port 8010 --reload
   ```

   **Options**:
   - `--host 127.0.0.1`: Bind to localhost only (safe for development)
   - `--port 8010`: Default port (change if occupied)
   - `--reload`: Auto-reload on code changes (development mode)

   **Production mode** (no auto-reload):
   ```bash
   nohup uvicorn src.mcp_synth.server:app --host 0.0.0.0 --port 8010 > synth_server.log 2>&1 &
   ```

3. **Verify Server is Running**:
   ```bash
   curl http://127.0.0.1:8010/healthz
   ```
   Expected response: `{"status":"ok"}`

   Check version:
   ```bash
   curl http://127.0.0.1:8010/version
   ```
   
   View server logs:
   ```bash
   tail -f synth_server.log
   ```

## Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check |
| `/version` | GET | Server version |
| `/synth_generate` | POST | Generate synthetic data |
| `/mcp` | POST | MCP JSON-RPC endpoint for tools |

## Stopping the Server

- Press `Ctrl+C` in the terminal running the server
- Or find and kill the process: `lsof -i :8010 | awk 'NR>1 {print $2}' | xargs kill`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8010 in use | Use `--port 8011` or kill existing process |
| Module not found | Run `pip install -e ".[dev]"` from `airline-discount-ml/` |
| Connection refused | Ensure server started successfully, check terminal output |
