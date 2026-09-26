"""The galaxy's own AI: an open model running on this computer through Ollama.

Standard library only. Nothing here talks to any online AI service: every
request goes to the Ollama server on this machine (http://127.0.0.1:11434 by
default). The only internet use is Ollama downloading the model once.

BrainManager keeps the brain alive in the background:
  * starts Ollama if it is installed but not running,
  * downloads the model the first time (with progress),
  * loads it into memory so the first question is quick,
and reports all of that through status() for the viewer's status pill.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "qwen2.5:3b"            # ~1.9 GB, good all-rounder that fits most laptops
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
KEEP_ALIVE = "30m"                      # keep the model in memory between questions
DOWNLOAD_PAGE = "https://ollama.com/download"

# Names that belong to online services. If config.json still names one (from the
# old online-AI setup), fall back to the local default instead.
CLOUD_MODEL_RE = re.compile(r"^(gpt-|o\d|chatgpt|claude|gemini|text-|davinci)", re.I)


class BrainError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code, self.message = code, message


def full_name(model):
    return model if ":" in model else model + ":latest"


class Ollama:
    """Tiny client for the parts of the Ollama HTTP API the galaxy uses."""

    def __init__(self, base_url=DEFAULT_OLLAMA_URL):
        self.base = base_url.rstrip("/")

    def _open(self, path, body=None, timeout=10):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data, method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json"})
        # Talk to the local server directly, even if the machine has an HTTP proxy configured.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            return opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read().decode("utf-8", "replace")).get("error", "")
            except Exception:
                pass
            if e.code == 404 and "not found" in detail.lower():
                raise BrainError("model_missing", detail or "The model is not downloaded yet.")
            raise BrainError("ollama_error", f"Ollama said: {detail or e.reason} ({e.code})")
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            raise BrainError("ollama_down", f"Can't reach Ollama at {self.base} ({getattr(e, 'reason', e)}).")

    def is_up(self):
        try:
            with self._open("/api/version", timeout=2) as r:
                r.read()
            return True
        except BrainError:
            return False

    def installed(self):
        with self._open("/api/tags", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
        return {m.get("name", "") for m in data.get("models", [])} | {m.get("model", "") for m in data.get("models", [])}

    def pull(self, model, on_progress):
        """Download a model, calling on_progress(done_bytes, total_bytes, status_text)."""
        layers = {}
        with self._open("/api/pull", {"model": model, "stream": True}, timeout=120) as r:
            for line in r:
                if not line.strip():
                    continue
                ev = json.loads(line.decode("utf-8"))
                if ev.get("error"):
                    raise BrainError("download_failed", ev["error"])
                if ev.get("digest") and ev.get("total"):
                    layers[ev["digest"]] = (ev.get("completed", 0), ev["total"])
                done = sum(c for c, _ in layers.values())
                total = sum(t for _, t in layers.values())
                on_progress(done, total, ev.get("status", ""))
                if ev.get("status") == "success":
                    return
        raise BrainError("download_failed", "The download stopped before it finished.")

    def load(self, model):
        """Load the model into memory (an empty generate request does exactly that)."""
        with self._open("/api/generate", {"model": model, "keep_alive": KEEP_ALIVE}, timeout=600) as r:
            r.read()

    def chat_json(self, model, messages, schema, num_ctx=8192, timeout=300):
        body = {"model": model, "messages": messages, "stream": False, "format": schema,
                "keep_alive": KEEP_ALIVE, "options": {"num_ctx": num_ctx, "temperature": 0.2}}
        with self._open("/api/chat", body, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        if data.get("error"):
            raise BrainError("ollama_error", data["error"])
        content = strip_think((data.get("message") or {}).get("content", ""))
        try:
            return json.loads(content)
        except ValueError:
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                return json.loads(m.group(0))
            raise BrainError("bad_json", "The brain's reply wasn't readable. Try again.")

    def chat_stream(self, model, messages, num_ctx=8192, fmt=None, timeout=600, on_open=None):
        """Yield answer text as it is written. Hidden <think> sections are dropped."""
        body = {"model": model, "messages": messages, "stream": True, "keep_alive": KEEP_ALIVE,
                "options": {"num_ctx": num_ctx}}
        if fmt is not None:
            body["format"] = fmt
        resp = self._open("/api/chat", body, timeout=timeout)
        if on_open:
            on_open(resp)
        filt = ThinkFilter()
        try:
            for line in resp:
                if not line.strip():
                    continue
                ev = json.loads(line.decode("utf-8"))
                if ev.get("error"):
                    raise BrainError("ollama_error", ev["error"])
                piece = filt.feed((ev.get("message") or {}).get("content", ""))
                if piece:
                    yield piece
                if ev.get("done"):
                    break
        finally:
            resp.close()


def strip_think(text):
    return re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S).strip()


class ThinkFilter:
    """Drops <think>...</think> blocks from a token stream (reasoning models emit them)."""

    def __init__(self):
        self.buf, self.inside = "", False

    def feed(self, chunk):
        self.buf += chunk
        out = ""
        while self.buf:
            if self.inside:
                end = self.buf.find("</think>")
                if end < 0:
                    self.buf = self.buf[-8:]
                    return out
                self.buf, self.inside = self.buf[end + 8:], False
            else:
                start = self.buf.find("<think>")
                if start < 0:
                    # keep a possible partial "<think" at the end for the next chunk
                    keep = next((k for k in range(min(7, len(self.buf)), 0, -1) if "<think>".startswith(self.buf[-k:])), 0)
                    out += self.buf[:len(self.buf) - keep]
                    self.buf = self.buf[len(self.buf) - keep:]
                    return out
                out += self.buf[:start]
                self.buf, self.inside = self.buf[start + 7:], True
        return out


# --------------------------------------------------------------------------- lifecycle

def _find_ollama_exe():
    exe = shutil.which("ollama")
    if exe:
        return exe
    candidates = []
    if sys.platform == "win32":
        candidates.append(os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"))
    elif sys.platform == "darwin":
        candidates += ["/Applications/Ollama.app/Contents/Resources/ollama", "/usr/local/bin/ollama", "/opt/homebrew/bin/ollama"]
    else:
        candidates += ["/usr/local/bin/ollama", "/usr/bin/ollama"]
    return next((c for c in candidates if c and os.path.isfile(c)), None)


def try_start_ollama():
    """Start Ollama in the background if it is installed. Returns True if we launched something."""
    if sys.platform == "darwin" and os.path.isdir("/Applications/Ollama.app"):
        subprocess.Popen(["open", "-g", "-a", "Ollama"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    exe = _find_ollama_exe()
    if not exe:
        return False
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([exe, "serve"], **kwargs)
    return True


class BrainManager:
    """Keeps the local AI ready and remembers what it is doing, for the status pill."""

    def __init__(self, config_loader):
        self.config_loader = config_loader
        self.lock = threading.Lock()
        self._status = {"state": "checking", "message": "Checking for the brain…", "model": DEFAULT_MODEL}
        self.wake = threading.Event()
        self.last_start_attempt = 0.0

    # -- config
    def settings(self):
        try:
            cfg = self.config_loader()
        except Exception:
            cfg = {}
        model = str(cfg.get("model") or "").strip()
        if not model or CLOUD_MODEL_RE.match(model):
            model = DEFAULT_MODEL
        url = str(cfg.get("ollama_url") or os.environ.get("GALAXY_OLLAMA_URL") or DEFAULT_OLLAMA_URL)
        return model, url

    def client(self):
        model, url = self.settings()
        return model, Ollama(url)

    # -- status
    def _set(self, **kw):
        with self.lock:
            self._status = kw

    def status(self):
        with self.lock:
            return dict(self._status)

    def ready_model(self):
        """The model name if the brain can answer right now, else raise BrainError with a friendly reason."""
        st = self.status()
        if st["state"] == "ready":
            return st["model"]
        self.wake.set()   # someone wants the brain: re-check now
        raise BrainError("brain_" + st["state"], st["message"])

    def mark_down(self, message):
        self._set(state="no_ollama" if "reach Ollama" in message else "error", message=message,
                  model=self.settings()[0])
        self.wake.set()

    # -- background loop
    def start(self):
        threading.Thread(target=self._run, name="brain-manager", daemon=True).start()

    def _run(self):
        while True:
            try:
                delay = self._step()
            except Exception as e:     # never let the manager thread die
                self._set(state="error", message=f"Brain problem: {e}", model=self.settings()[0])
                delay = 10
            self.wake.wait(delay)
            self.wake.clear()

    def _step(self):
        model, ollama = self.client()
        if not ollama.is_up():
            if time.time() - self.last_start_attempt > 60:
                self.last_start_attempt = time.time()
                if try_start_ollama():
                    self._set(state="starting", message="Starting Ollama…", model=model)
                    for _ in range(30):
                        time.sleep(1)
                        if ollama.is_up():
                            return 0
            if _find_ollama_exe() or (sys.platform == "darwin" and os.path.isdir("/Applications/Ollama.app")):
                msg = "Ollama is installed but not running. Open the Ollama app, and the galaxy will connect by itself."
            else:
                msg = "Install Ollama (free) to give the galaxy its brain. The galaxy will notice when it's installed."
            self._set(state="no_ollama", message=msg, model=model, link=DOWNLOAD_PAGE)
            return 5

        if full_name(model) not in ollama.installed():
            self._set(state="downloading", message=f"Downloading its brain ({model})…", model=model, progress=0)

            def progress(done, total, text):
                pct = (done / total) if total else 0
                gb = f"{done / 1e9:.1f} of {total / 1e9:.1f} GB" if total else text
                self._set(state="downloading", message=f"Downloading its brain: {pct:.0%} ({gb})",
                          model=model, progress=round(pct, 3))
            try:
                ollama.pull(model, progress)
            except BrainError as e:
                self._set(state="error", message=f"Download failed: {e.message}. Retrying in 30 seconds.", model=model)
                return 30
            return 0

        st = self.status()
        if st.get("state") != "ready" or st.get("model") != model:
            self._set(state="waking", message="Waking up…", model=model)
            try:
                ollama.load(model)
            except BrainError as e:
                self._set(state="error", message=f"Couldn't load the brain: {e.message}", model=model)
                return 15
            self._set(state="ready", message=f"Ready · {model} · running on this computer", model=model)
        return 15
