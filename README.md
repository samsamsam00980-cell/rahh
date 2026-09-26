# Knowledge Galaxy

A 3D galaxy of your markdown notes, plus a chat "brain" that answers only from them.

```
python3 server.py        # re-indexes ./notes, then serves http://127.0.0.1:4700
```

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
