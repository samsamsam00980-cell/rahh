# Knowledge Galaxy

A 3D galaxy of your markdown notes, plus an AI "brain" you can ask anything. It uses your notes when they are relevant and its own knowledge otherwise.

**Easiest:** double-click `start.command` (Mac) or `start.bat` (Windows). Your browser opens on the galaxy.

Or from a terminal:

```
python3 server.py        # re-indexes ./notes, serves http://127.0.0.1:4700 and opens your browser
```

Set `GALAXY_NO_BROWSER=1` to skip opening the browser.

- `notes/` — markdown notes (30 sample notes about Harbor Street Coffee). Sub-folder = group/colour.
- `build.py` — standard-library indexer → `viewer/graph-data.js` (`const GRAPH = {nodes, links}`; node `id` = its index).
  Run it on its own with `python3 build.py [notes_dir]`.
- `viewer/index.html` — single page, three.js + 3d-force-graph from jsDelivr. No npm, no build step.
- `server.py` — standard library, port 4700, serves only `viewer/`, plus `POST /chat` and `POST /chat/reset`.
- `config.json` — project root, git-ignored, never served. Read on every request, so edits apply without a restart:

```json
{"openai_api_key": "sk-...", "model": "gpt-6-astra"}
```

`POST /chat` with `{"question": "...", "session": "..."}` returns `{"answer": "...", "nodes": [note indexes used]}`.
If something goes wrong it returns `{"error": "...", "code": "...", "nodes": [...]}` instead. Codes: `missing_api_key`,
`bad_api_key`, `bad_model`, `rate_limited`, `network_error`, `bad_config`.
