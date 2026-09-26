# Knowledge Galaxy

A 3D galaxy of your markdown notes, with its own AI brain you can ask anything.

The brain is an open AI model that runs **on your computer** through [Ollama](https://ollama.com).
It is not Claude, ChatGPT or any online service: no account, no API key, no cost, and nothing you
ask leaves your machine. It uses your notes when they are relevant and its own knowledge for everything else.

## Start it

1. Install **Ollama** (free): https://ollama.com/download. Just install it; you don't need to open or set up anything.
2. Double-click **`start.command`** (Mac) or **`start.bat`** (Windows).
   Or from a terminal: `python3 server.py`

Your browser opens on the galaxy. The first time, the galaxy starts Ollama and downloads its brain
(about 1.9 GB). The **Brain** line in the top-left shows the progress, then turns green: *Ready*.
After that it works offline.

## Use it

- **Ask anything** in the bottom bar. The brain decides for itself which notes to open; the camera
  flies to each one it reads, and the notes it used light up under the answer.
- **Let it think**: the brain reads every note without being asked and reports risks, conflicts and
  opportunities that span several notes. Click a card to light up its notes. Ask follow-ups.
- **Stop** interrupts it. **New conversation** makes it forget the chat so far.
- Click any star to read that note; click a folder in the legend to see just that group.

## Change the brain

`config.json` (created next to `server.py`, never served to the browser):

```json
{"model": "qwen2.5:3b"}
```

Any model from https://ollama.com/library works. Save the file and the galaxy downloads and switches
to it by itself. Suggestions:

| Model | Download | Good for |
|---|---|---|
| `qwen2.5:3b` (default) | 1.9 GB | Most laptops (8 GB memory) |
| `qwen2.5:1.5b` | 1.0 GB | Older or slower computers |
| `qwen2.5:7b` | 4.7 GB | Better answers, needs 16 GB memory |

## Your own notes

Put `.md` files in `notes/` (each subfolder becomes a colour group) and restart. Notes link up when
one mentions another's title (the filename without `.md`) or they share `[[wikilinks]]`.

## Files

- `build.py`: standard-library indexer → `viewer/graph-data.js` (`const GRAPH = {nodes, links}`; node `id` = its index).
- `viewer/index.html`: single page, three.js + 3d-force-graph from jsDelivr. No npm, no build step.
- `server.py`: standard library, port 4700 (`GALAXY_PORT` to change), serves only `viewer/`.
  `GET /brain/status`, `POST /chat` and `POST /think` (streamed JSON lines), `POST /chat/reset`.
- `brain.py`: talks to Ollama on this computer; starts it, downloads the model, keeps it loaded.
