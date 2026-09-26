#!/usr/bin/env python3
"""Knowledge galaxy server — Python standard library only.

    python3 server.py            -> http://127.0.0.1:4700

* Serves ONLY the viewer/ folder as static files.
* POST /chat        {"question": "...", "session": "..."}
                    -> {"answer": "...", "nodes": [note indexes used]}
* POST /chat/reset  {"session": "..."} clears that conversation's history.

The OpenAI key lives in ./config.json (project root, outside viewer/), is read
fresh on every request, and is never sent to the browser.
"""
import collections
import json
import math
import os
import re
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import build

ROOT = os.path.dirname(os.path.abspath(__file__))
VIEWER_DIR = os.path.realpath(os.path.join(ROOT, "viewer"))
CONFIG_PATH = os.path.join(ROOT, "config.json")
HOST = os.environ.get("GALAXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("GALAXY_PORT", "4700"))

PLACEHOLDER_KEY = "PUT-YOUR-KEY-HERE"
DEFAULT_CONFIG = {"openai_api_key": PLACEHOLDER_KEY, "model": "gpt-6-astra"}
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

TOP_K = 6
NOTE_CHARS_FOR_MODEL = 3000
HISTORY_MESSAGES = 6          # last 3 question/answer pairs per conversation
MAX_SESSIONS = 200
MAX_BODY_BYTES = 16 * 1024
MAX_QUESTION_CHARS = 1000

SYSTEM_PROMPT = """You answer questions about the user's own notes.
Answer ONLY from the notes provided below — never from general knowledge.
Answer in two or three sentences, plainly, without markdown.
If the notes do not cover the question, say so plainly (for example: "Your notes don't cover that.") and do not guess.
Earlier turns of this conversation are included so you can resolve follow-up questions.
On the very last line write "SOURCES:" followed by the bracketed ids of the notes you actually used, comma-separated (e.g. "SOURCES: 7, 12"), or "SOURCES: none".

NOTES:
{notes}"""

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

    def score(self, question, previous_question=""):
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
                    s += 3.0 * weight                      # title matches weigh more
                tf = self.body_counts[i].get(term, 0)
                if tf:
                    s += weight * (1.0 + math.log(tf))
            title_phrase = " " + " ".join(re.findall(r"[a-z0-9]+", note["label"].lower())) + " "
            if title_phrase.strip() and title_phrase in q_lower:
                s += 6.0                                   # whole title named in the question
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [i for _, i in scored[:TOP_K]]


def load_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
            f.write("\n")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def key_is_placeholder(key):
    key = (key or "").strip()
    return not key or key == PLACEHOLDER_KEY or "PUT-YOUR" in key.upper()


class ChatError(Exception):
    def __init__(self, status, message, code):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


def call_openai(api_key, model, messages):
    body = json.dumps({"model": model, "messages": messages}).encode("utf-8")
    req = urllib.request.Request(OPENAI_URL, data=body, method="POST", headers={
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            pass
        if e.code == 401:
            raise ChatError(502, "OpenAI rejected the API key in config.json (401). Check it and try again.", "bad_api_key")
        if e.code == 404 or "model" in detail.lower():
            raise ChatError(502, f"OpenAI could not use model '{model}': {detail or e.reason}. "
                                 "Change \"model\" in config.json.", "bad_model")
        if e.code == 429:
            raise ChatError(502, "OpenAI rate limit or quota reached (429). " + detail, "rate_limited")
        raise ChatError(502, f"OpenAI error {e.code}: {detail or e.reason}", "openai_error")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise ChatError(502, f"Could not reach the OpenAI API: {reason}", "network_error")
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise ChatError(502, "OpenAI returned an unexpected response.", "openai_error")


SOURCES_RE = re.compile(r"^\s*\**sources\**\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def split_sources(text, allowed):
    """Pull the SOURCES line off the model's reply -> (answer, [ids])."""
    matches = list(SOURCES_RE.finditer(text))
    if not matches:
        return text.strip(), list(allowed)
    m = matches[-1]
    answer = (text[:m.start()] + text[m.end():]).strip()
    ids = [int(x) for x in re.findall(r"\d+", m.group(1))]
    used = []
    for i in ids:
        if i in allowed and i not in used:
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


# --------------------------------------------------------------------------- HTTP

class Handler(SimpleHTTPRequestHandler):
    server_version = "KnowledgeGalaxy/1.0"
    index = None
    conversations = None

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

    # ---- JSON API
    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/chat":
                self.handle_chat(self.read_json())
            elif path == "/chat/reset":
                data = self.read_json()
                self.conversations.reset(str(data.get("session", ""))[:64])
                self.send_json(200, {"ok": True})
            else:
                self.send_json(404, {"error": "Not found", "code": "not_found"})
        except ChatError as e:
            self.send_json(e.status, {"error": e.message, "code": e.code, "nodes": getattr(e, "nodes", [])})
        except Exception as e:  # never let one bad request take the server down
            self.log_message("chat failed: %r", e)
            self.send_json(500, {"error": "Internal error while answering. See the server log.", "code": "internal"})

    def do_PUT(self):
        self.send_json(405, {"error": "Method not allowed"})
    do_DELETE = do_PATCH = do_PUT

    def handle_chat(self, data):
        question = str(data.get("question", "")).strip()
        sid = re.sub(r"[^A-Za-z0-9_-]", "", str(data.get("session", "default")))[:64] or "default"
        if not question:
            raise ChatError(400, "Ask a question first.", "bad_request")
        if len(question) > MAX_QUESTION_CHARS:
            raise ChatError(400, f"Question is too long (max {MAX_QUESTION_CHARS} characters).", "bad_request")

        history, last_question = self.conversations.get(sid)
        top = self.index.score(question, last_question)

        try:
            cfg = load_config()
        except (OSError, ValueError) as e:
            err = ChatError(500, f"config.json could not be read: {e}", "bad_config")
            err.nodes = top
            raise err
        api_key = str(cfg.get("openai_api_key", ""))
        model = str(cfg.get("model", "")).strip() or DEFAULT_CONFIG["model"]
        if key_is_placeholder(api_key):
            err = ChatError(503, "No OpenAI API key yet. Paste your key into config.json in the project "
                                 "root (replacing PUT-YOUR-KEY-HERE) and ask again. No restart needed.",
                            "missing_api_key")
            err.nodes = top   # the viewer can still light up the notes it would have used
            raise err

        notes = self.index.notes
        context = "\n\n".join(
            f"[{i}] {notes[i]['label']} (folder: {notes[i]['group']})\n{notes[i]['text'][:NOTE_CHARS_FOR_MODEL]}"
            for i in top
        ) or "(no notes matched this question)"
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(notes=context)}]
        messages += history
        messages.append({"role": "user", "content": question})

        try:
            reply = call_openai(api_key, model, messages)
        except ChatError as e:
            e.nodes = top
            raise
        answer, used = split_sources(reply, top)
        if not answer:
            answer = "Your notes don't cover that."
        self.conversations.append(sid, question, answer)
        self.send_json(200, {"answer": answer, "nodes": used})


def main():
    notes, nodes, links = build.build()          # re-index so node ids match what we score
    load_config()                                # creates config.json with the placeholder if missing
    Handler.index = NoteIndex(notes)
    Handler.conversations = Conversations()
    handler = partial(Handler, directory=VIEWER_DIR)
    httpd = ThreadingHTTPServer((HOST, PORT), handler)
    print(f"Knowledge galaxy: http://{HOST if HOST != '0.0.0.0' else 'localhost'}:{PORT}  "
          f"({len(nodes)} notes, {len(links)} links)  Ctrl+C to stop")
    if os.environ.get("GALAXY_NO_BROWSER") != "1":
        url = f"http://127.0.0.1:{PORT}/"
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()   # open the galaxy for you
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
