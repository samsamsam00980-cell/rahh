#!/bin/bash
# Mac: double-click to start the Knowledge Galaxy (opens your browser).
cd "$(dirname "$0")"
python3 server.py
echo; read -n 1 -s -r -p "Server stopped. Press any key to close this window."
