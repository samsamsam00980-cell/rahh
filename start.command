#!/bin/bash
# Mac: double-click to start the Knowledge Galaxy (opens your browser).
cd "$(dirname "$0")"
echo "Starting the Knowledge Galaxy. Its brain runs on this computer through Ollama (https://ollama.com/download)."
python3 server.py
echo; read -n 1 -s -r -p "Server stopped. Press any key to close this window."
