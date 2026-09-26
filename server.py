#!/usr/bin/env python3
"""Knowledge galaxy server — Python standard library only.

    python3 server.py            -> http://127.0.0.1:4700  (opens your browser)

* Serves ONLY the viewer/ folder as static files.
* The galaxy's own AI runs on this computer through Ollama (see brain.py).
  No Claude, no OpenAI, no API keys; nothing you ask leaves this machine.
* GET  /brain/status  what the brain is doing (downloading, ready, ...)
* POST /chat         {"question": "...", "session": "..."}
                     -> streamed NDJSON: step / token events, then
                        {"type": "done", "answer": "...", "nodes": [note indexes used]}
* POST /think        {"session": "..."} -> the brain reads every note on its own and
                     streams back {"type": "insights", "insights": [...]}
* POST /chat/reset   {"session": "..."} clears that conversation's history.

config.json (project root, never served): {"model": "qwen2.5:3b"}
"""
import collections
import json
import math
import os
import re
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import brain
import build
from brain import BrainError

ROOT = os.path.dirname(os.path.abspath(__file__))
VIEWER_DIR = os.path.realpath(os.path.join(ROOT, "viewer"))
CONFIG_PATH = os.path.join(ROOT, "config.json")
HOST = os.environ.get("GALAXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("GALAXY_PORT", "4700"))

DEFAULT_CONFIG = {"model": brain.DEFAULT_MODEL}

TOP_K = 6
MAX_NOTES_READ = 4
NOTE_CHARS_FOR_MODEL = 2500
NOTE_CHARS_FOR_THINK = 1500
HISTORY_MESSAGES = 6          # last 3 question/answer pairs per conversation
MAX_SESSIONS = 200
MAX_BODY_BYTES = 16 * 1024
MAX_QUESTION_CHARS = 1000

IDENTITY = """You are the Knowledge Galaxy's own AI. You run entirely on the user's own computer ({model}, an open model running privately through Ollama). You are not ChatGPT, Claude, Gemini or any online service, and nothing the user types leaves their computer. If asked what you are, say that."""

ANSWER_PROMPT = IDENTITY + """

The user can ask you anything: questions about their notes (a small coffee roastery and café business), general knowledge, advice, ideas, writing help, maths, small talk. Think for yourself.
- If notes are given below, use them and reason across them: connect dots, notice conflicts, do the arithmetic. Say "From your notes..." for facts that come from them. Never invent facts about the user's business.
- If no notes are given, or they don't fit the question, answer from your own knowledge like any capable assistant.
- Be conversational and concise: usually two to five sentences. Plain text, no markdown headings or tables.
- After your answer, write one last line: SOURCES: then the numbers of the notes you used, e.g. "SOURCES: 7, 12", or "SOURCES: none".

{notes}"""

PICK_PROMPT = """You help an AI decide which of the user's notes to open before it answers.
The notes (number, title, folder, start of text):
{catalogue}

Earlier in the conversation the user asked: {previous}
Latest message: {question}

Which notes, if any, would help answer the latest message? Pick at most {k}, most useful first.
If the message is general knowledge, chit-chat or otherwise not about the user's business, pick none.
Reply with JSON only: {{"notes": [numbers]}}"""

PICK_SCHEMA = {"type": "object", "properties": {"notes": {"type": "array", "items": {"type": "integer"}}},
               "required": ["notes"]}

THINK_PROMPT = IDENTITY + """

Nobody has asked you anything. Think for yourself: read all of the user's notes below and find the 4 things the owner most needs to notice right now that are NOT obvious from any single note: plans that collide with cash, promises that conflict, risks building up, opportunities hiding across notes. Each one must connect at least two notes, cite concrete facts (dates, amounts, names) and say what you would do about it. Use only facts that are in the notes.

Reply with JSON only, in this shape:
{{"insights": [{{"kind": "risk | conflict | opportunity | connection", "headline": "at most 9 words", "detail": "2-3 plain sentences", "notes": [note numbers]}}]}}

NOTES:
{notes}"""

THINK_SCHEMA = {
    "type": "object",
    "properties": {"insights": {"type": "array", "items": {
        "type": "object",
        "properties": {"kind": {"type": "string", "enum": ["risk", "conflict", "opportunity", "connection"]},
                       "headline": {"type": "string"}, "detail": {"type": "string"},
                       "notes": {"type": "array", "items": {"type": "integer"}}},
        "required": ["kind", "headline", "detail", "notes"]}}},
    "required": ["insights"],
}

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how i if
in into is it its itself just me more most my myself no nor not now of off on once only or other our ours out over own
same she should so some such than that the their theirs them then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your yours yourself tell
know anything thing things there's what's whats did does much many any also us let lets please give show get got
""".split())


# --------------------------------------------------------------------------- notes + scoring

def tokenize(text):
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
    out = []
    for w in words:
        w = w.split("'")[0]
        if len(w) < 2 or w in STOPWORDS:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude plural folding: invoices -> invoice
        out.append(w)
    return out


class NoteIndex:
    def __init__(self, notes):
        self.notes = notes
        self.title_tokens = [set(tokenize(n["label"])) for n in notes]
        self.body_counts = [collections.Counter(tokenize(n["text"])) for n in notes]
        self.catalogue = "\n".join(
            "[%d] %s (%s): %s" % (n["id"], n["label"], n["group"], " ".join(body_only(n).split())[:100])
            for n in notes)

    def ranked(self, question, previous_question=""):
        """[(note index, score)] best first, by keyword overlap; title matches weigh more."""
        terms = collections.Counter()
        for t in set(tokenize(question)):
            terms[t] += 1.0
        for t in set(tokenize(previous_question)):   # keeps follow-ups on topic
            terms[t] = max(terms[t], 0.5)
        q_lower = " " + " ".join(re.findall(r"[a-z0-9]+", question.lower())) + " "
        scored = []
        for i, note in enumerate(self.notes):
            s = 0.0
            for term, weight in terms.items():
                if term in self.title_tokens[i]:
                    s += 3.0 * weight
                tf = self.body_counts[i].get(term, 0)
                if tf:
                    s += weight * (1.0 + math.log(tf))
            title_phrase = " " + " ".join(re.findall(r"[a-z0-9]+", note["label"].lower())) + " "
            if title_phrase.strip() and title_phrase in q_lower:
                s += 6.0                                   # whole title named in the question
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:TOP_K]

    def named_in(self, question):
        """Notes whose full title appears in the question, e.g. "what's in the Hiring Plan?"."""
        q = _words(question)
        return [i for i, n in enumerate(self.notes) if _words(n["label"]).strip() and _words(n["label"]) in q]


def _words(text):
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def body_only(note):
    """Note text without a first line that just repeats the title."""
    first, _, rest = note["text"].partition("\n")
    return rest.strip() if build.normalize(first) == build.normalize(note["label"]) else note["text"]


def load_config():
    """Read config.json, creating it (or replacing the old OpenAI placeholder file) when needed."""
    cfg = None
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        old_placeholder = (isinstance(cfg, dict) and "openai_api_key" in cfg
                           and "PUT-YOUR" in str(cfg.get("openai_api_key", "")).upper())
        if not old_placeholder:
            return cfg if isinstance(cfg, dict) else {}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
        f.write("\n")
    return dict(DEFAULT_CONFIG)


class ChatError(Exception):
    def __init__(self, status, message, code):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


SOURCES_RE = re.compile(r"^\W*sources\W*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def split_sources(text, read):
    """Pull the SOURCES line off the reply -> (answer, [note ids used])."""
    matches = list(SOURCES_RE.finditer(text))
    if not matches:
        return text.strip(), list(read)
    m = matches[-1]
    answer = (text[:m.start()] + text[m.end():]).strip()
    used = []
    for x in re.findall(r"\d+", m.group(1)):
        i = int(x)
        if i in read and i not in used:
            used.append(i)
    return answer, used


# --------------------------------------------------------------------------- conversation memory

class Conversations:
    def __init__(self):
        self.lock = threading.Lock()
        self.sessions = collections.OrderedDict()

    def get(self, sid):
        with self.lock:
            convo = self.sessions.get(sid, {"messages": [], "last_question": ""})
            return list(convo["messages"]), convo["last_question"]

    def append(self, sid, question, answer):
        with self.lock:
            convo = self.sessions.pop(sid, {"messages": [], "last_question": ""})
            convo["messages"] += [{"role": "user", "content": question},
                                  {"role": "assistant", "content": answer}]
            convo["messages"] = convo["messages"][-HISTORY_MESSAGES:]
            convo["last_question"] = question
            self.sessions[sid] = convo
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)

    def reset(self, sid):
        with self.lock:
            self.sessions.pop(sid, None)


class ClientGone(Exception):
    """The browser closed the connection (Stop button, closed tab)."""


# --------------------------------------------------------------------------- HTTP

class Handler(SimpleHTTPRequestHandler):
    server_version = "KnowledgeGalaxy/2.0"
    index = None
    conversations = None
    brain = None

    # ---- guard: only answer requests addressed to this machine (blocks DNS-rebinding pages)
    def host_ok(self):
        if HOST not in ("127.0.0.1", "localhost", "::1"):
            return True
        host = (self.headers.get("Host") or "").lower()
        return host in {f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"[::1]:{PORT}"}

    # ---- static files: viewer/ only
    def translate_path(self, path):
        full = super().translate_path(path)   # already rooted at viewer/ and strips ".."
        real = os.path.realpath(full)
        if real != VIEWER_DIR and not real.startswith(VIEWER_DIR + os.sep):
            return os.path.join(VIEWER_DIR, "__forbidden__")
        if any(part.startswith(".") for part in os.path.relpath(real, VIEWER_DIR).split(os.sep) if part != "."):
            return os.path.join(VIEWER_DIR, "__forbidden__")
        return full

    def list_directory(self, path):
        self.send_error(404, "Not found")
        return None

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def do_GET(self):
        if not self.host_ok():
            return self.send_json(403, {"error": "Forbidden", "code": "forbidden"})
        if self.path.split("?", 1)[0] == "/brain/status":
            return self.send_json(200, self.brain.status())
        return super().do_GET()

    def do_HEAD(self):
        if not self.host_ok():
            return self.send_json(403, {"error": "Forbidden", "code": "forbidden"})
        return super().do_HEAD()

    # ---- JSON + streaming helpers
    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def start_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True          # HTTP/1.0: the stream ends when we close

    def emit(self, **event):
        try:
            self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            raise ClientGone()

    def read_json(self):
        ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype != "application/json":   # forces a CORS preflight for cross-site pages
            raise ChatError(415, "Send the request as application/json.", "bad_request")
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and origin.split("://", 1)[-1] != host:
            raise ChatError(403, "Cross-origin requests are not allowed.", "forbidden")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ChatError(400, "Request body missing or too large.", "bad_request")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ChatError(400, "Request body is not valid JSON.", "bad_request")
        if not isinstance(data, dict):
            raise ChatError(400, "Request body must be a JSON object.", "bad_request")
        return data

    def session_id(self, data):
        return re.sub(r"[^A-Za-z0-9_-]", "", str(data.get("session", "default")))[:64] or "default"

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        streaming = False
        try:
            if not self.host_ok():
                raise ChatError(403, "Forbidden.", "forbidden")
            if path == "/chat":
                data = self.read_json()
                question = str(data.get("question", "")).strip()
                if not question:
                    raise ChatError(400, "Ask a question first.", "bad_request")
                if len(question) > MAX_QUESTION_CHARS:
                    raise ChatError(400, f"Question is too long (max {MAX_QUESTION_CHARS} characters).", "bad_request")
                model = self.ready_model_or_error()
                self.start_stream(); streaming = True
                self.stream_chat(self.session_id(data), question, model)
            elif path == "/think":
                data = self.read_json()
                model = self.ready_model_or_error()
                self.start_stream(); streaming = True
                self.stream_think(self.session_id(data), model)
            elif path == "/chat/reset":
                data = self.read_json()
                self.conversations.reset(self.session_id(data))
                self.send_json(200, {"ok": True})
            else:
                self.send_json(404, {"error": "Not found", "code": "not_found"})
        except ClientGone:
            pass                                   # Stop button: nothing to answer
        except ChatError as e:
            if streaming:
                self.safe_emit(type="error", code=e.code, error=e.message)
            else:
                self.send_json(e.status, {"error": e.message, "code": e.code})
        except BrainError as e:
            if e.code in ("ollama_down", "model_missing"):
                self.brain.mark_down(e.message)
            msg = friendly_brain_error(e)
            if streaming:
                self.safe_emit(type="error", code=e.code, error=msg)
            else:
                self.send_json(503, {"error": msg, "code": e.code})
        except Exception as e:  # never let one bad request take the server down
            self.log_message("request failed: %r", e)
            if streaming:
                self.safe_emit(type="error", code="internal", error="Something went wrong inside the brain. See the server window.")
            else:
                self.send_json(500, {"error": "Internal error. See the server window.", "code": "internal"})

    def safe_emit(self, **event):
        try:
            self.emit(**event)
        except ClientGone:
            pass

    def do_PUT(self):
        self.send_json(405, {"error": "Method not allowed"})
    do_DELETE = do_PATCH = do_PUT

    def ready_model_or_error(self):
        try:
            return self.brain.ready_model()
        except BrainError as e:
            st = self.brain.status()
            raise ChatError(503, st.get("message") or e.message, e.code)

    # ---- the brain at work
    def pick_notes(self, ollama, model, question, last_question, history):
        """The brain decides for itself which notes to open. Keyword search is the safety net."""
        ranked = self.index.ranked(question, last_question)
        previous = last_question or "(nothing yet)"
        prompt = PICK_PROMPT.format(catalogue=self.index.catalogue, previous=previous,
                                    question=question, k=MAX_NOTES_READ)
        picked = None
        try:
            reply = ollama.chat_json(model, [{"role": "user", "content": prompt}], PICK_SCHEMA)
            raw = reply.get("notes", []) if isinstance(reply, dict) else []
            picked = []
            for x in raw:
                try:
                    i = int(x)
                except (TypeError, ValueError):
                    continue
                if 0 <= i < len(self.index.notes) and i not in picked:
                    picked.append(i)
        except BrainError as e:
            if e.code in ("ollama_down", "model_missing"):
                raise
            picked = None                        # the picker stumbled: fall back to keywords
        if picked is None:
            picked = [i for i, s in ranked if s >= 3][:3]
        # A note named outright in the question is always worth opening.
        for i in reversed(self.index.named_in(question)):
            if i not in picked:
                picked.insert(0, i)
        return picked[:MAX_NOTES_READ]

    def stream_chat(self, sid, question, model):
        _, ollama = self.brain.client()
        history, last_question = self.conversations.get(sid)
        notes = self.index.notes

        self.emit(type="step", verb="Thinking", text="deciding which notes to open")
        picked = self.pick_notes(ollama, model, question, last_question, history)
        for i in picked:
            self.emit(type="step", verb="Read", id=i)
        if not picked:
            self.emit(type="step", verb="Answering", text="from its own knowledge")

        if picked:
            context = "Notes you opened for this question:\n\n" + "\n\n".join(
                f"[{i}] {notes[i]['label']} (folder: {notes[i]['group']})\n{body_only(notes[i])[:NOTE_CHARS_FOR_MODEL]}"
                for i in picked)
        else:
            context = "No notes were opened for this question."
        messages = [{"role": "system", "content": ANSWER_PROMPT.format(model=model, notes=context)}]
        messages += history
        messages.append({"role": "user", "content": question})

        text = ""
        holder = {}
        try:
            for piece in ollama.chat_stream(model, messages, on_open=lambda r: holder.setdefault("resp", r)):
                text += piece
                self.emit(type="token", text=piece)
        except ClientGone:
            if holder.get("resp"):
                holder["resp"].close()             # stop generating on Stop
            raise
        answer, used = split_sources(text, picked)
        if not answer:
            answer = "I'm not sure how to answer that one. Try asking another way."
        self.conversations.append(sid, question, answer)
        self.emit(type="done", answer=answer, nodes=used)

    def stream_think(self, sid, model):
        _, ollama = self.brain.client()
        notes = self.index.notes
        self.emit(type="step", verb="Reading", text=f"all {len(notes)} notes")
        body = "\n\n".join(f"[{n['id']}] {n['label']} ({n['group']})\n{body_only(n)[:NOTE_CHARS_FOR_THINK]}" for n in notes)
        prompt = THINK_PROMPT.format(model=model, notes=body)
        text, chars = "", 0
        holder = {}
        try:
            for piece in ollama.chat_stream(model, [{"role": "user", "content": prompt}], num_ctx=12288,
                                            fmt=THINK_SCHEMA, on_open=lambda r: holder.setdefault("resp", r)):
                text += piece
                if len(text) - chars > 40:         # heartbeat so the page knows it's writing
                    chars = len(text)
                    self.emit(type="progress", chars=chars)
        except ClientGone:
            if holder.get("resp"):
                holder["resp"].close()
            raise
        insights = clean_insights(text, len(notes))
        if not insights:
            raise ChatError(502, "The brain's thoughts came out jumbled. Press Let it think to try again.", "bad_json")
        self.emit(type="insights", insights=insights)
        summary = "\n".join(f"{k + 1}. {x['headline']}: {x['detail']} (notes {', '.join(map(str, x['notes']))})"
                            for k, x in enumerate(insights))
        self.conversations.append(sid, "Look through all my notes and tell me what you notice on your own.", summary)
        self.emit(type="done", nodes=sorted({i for x in insights for i in x["notes"]}))


def clean_insights(text, note_count):
    text = brain.strip_think(text)
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except ValueError:
            data = {}
    items = data.get("insights") if isinstance(data, dict) else data if isinstance(data, list) else []
    out = []
    for x in items or []:
        if not isinstance(x, dict) or not str(x.get("headline", "")).strip():
            continue
        ids = []
        for n in x.get("notes") or []:
            try:
                i = int(n)
            except (TypeError, ValueError):
                continue
            if 0 <= i < note_count and i not in ids:
                ids.append(i)
        kind = str(x.get("kind", "connection")).lower()
        out.append({"kind": kind if kind in ("risk", "conflict", "opportunity", "connection") else "connection",
                    "headline": str(x["headline"]).strip()[:120], "detail": str(x.get("detail", "")).strip()[:700],
                    "notes": ids})
    return out[:6]


def friendly_brain_error(e):
    if e.code == "ollama_down":
        return "I lost touch with Ollama, the app that runs my brain. Is it still open? I'll reconnect by myself."
    if e.code == "model_missing":
        return "My brain isn't downloaded yet. The galaxy is fetching it now; watch the status in the top-left."
    return f"My brain hit a problem: {e.message}"


def main():
    notes, nodes, links = build.build()          # re-index so node ids match what we score
    load_config()                                # creates config.json if missing
    Handler.index = NoteIndex(notes)
    Handler.conversations = Conversations()
    Handler.brain = brain.BrainManager(load_config)
    Handler.brain.start()
    handler = partial(Handler, directory=VIEWER_DIR)
    httpd = ThreadingHTTPServer((HOST, PORT), handler)
    httpd.daemon_threads = True
    model, _ = Handler.brain.settings()
    print(f"Knowledge galaxy: http://{HOST if HOST != '0.0.0.0' else 'localhost'}:{PORT}  "
          f"({len(nodes)} notes, {len(links)} links)  Ctrl+C to stop")
    print(f"Brain: {model}, running on this computer through Ollama ({brain.DOWNLOAD_PAGE})")
    if os.environ.get("GALAXY_NO_BROWSER") != "1":
        url = f"http://127.0.0.1:{PORT}/"
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()   # open the galaxy for you
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
