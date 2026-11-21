"""
Antarmuka web untuk memori yang bisa diajari (hanya pustaka standar Python, berjalan offline).

  python tools/memory_web.py                      # http://127.0.0.1:8765
  python tools/memory_web.py --checkpoint <berkas> --db data/memory/memory.db --open

Keamanan: server hanya mendengarkan 127.0.0.1 (komputer ini saja) dan menolak permintaan dengan header Host
selain localhost/127.0.0.1 (menahan DNS rebinding). Tidak ada autentikasi: jangan dibuka ke jaringan.
Memori disimpan lokal di berkas SQLite (--db); tidak ada data yang dikirim ke luar.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from memory_agent import MemoryAgent, MemoryStore, Reader

WEB_DIR = PROJECT_ROOT / "web" / "memory"
MAX_BODY = 20_000
MAX_NOTE = 600
MAX_QUESTION = 300
STRICTNESS = {"longgar": 0.5, "seimbang": 0.67, "ketat": 1.0}


def note_hints(text):
    """Saran penulisan catatan agar mudah dibaca model kecil (bukan larangan)."""
    hints = []
    t = text.strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    if len(t) > 220:
        hints.append("Catatan cukup panjang. Model lebih akurat dengan satu fakta per kalimat; pertimbangkan memecahnya.")
    if len(sentences) >= 3:
        hints.append("Ada beberapa kalimat sekaligus. Simpan satu fakta per catatan agar mudah ditemukan.")
    if re.match(r"^(dia|ia|mereka|beliau|itu|ini|nya)\b", t, re.I):
        hints.append("Mulailah dengan nama lengkap, bukan kata ganti (dia/ia/mereka), agar jelas siapa yang dimaksud.")
    if re.search(r"\b(kemarin|tadi|besok|sekarang|hari ini|minggu depan)\b", t, re.I):
        hints.append("Kata waktu relatif (kemarin/besok/sekarang) akan basi. Tulis tanggal atau hari yang pasti.")
    if len(t) < 12:
        hints.append("Catatan terlalu singkat; tulis kalimat lengkap dengan subjek dan keterangannya.")
    return hints


class State:
    def __init__(self, checkpoint, db, k):
        self.lock = threading.Lock()  # model dan SQLite dipakai satu per satu
        self.store = MemoryStore(db)
        print("[web] memuat model ...", flush=True)
        t0 = time.time()
        self.agent = MemoryAgent(self.store, Reader(checkpoint), k=k)
        self.checkpoint = str(checkpoint)
        print(f"[web] model siap ({time.time() - t0:.0f} dtk).", flush=True)

    def notes(self):
        return [{"id": i, "text": t} for i, t in self.store.list()]


class ExclusiveServer(ThreadingHTTPServer):
    """Di Windows, SO_REUSEADDR mengizinkan dua proses memakai port yang sama tanpa galat (permintaan terbagi acak
    antar-server, termasuk dengan memori berbeda). Maka pemakaian ulang alamat dimatikan di Windows."""
    allow_reuse_address = (os.name != "nt")
    daemon_threads = True


def make_handler(state, port):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", "127.0.0.1", "localhost"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "MemoriWeb/1.0"

        def log_message(self, fmt, *args):
            pass

        def _host_ok(self):
            return (self.headers.get("Host") or "").lower() in allowed_hosts

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if ctype.startswith("text/html"):
                self.send_header("Content-Security-Policy",
                                 "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                                 "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(data)

        def _json_body(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None
            if n <= 0 or n > MAX_BODY:
                return None
            try:
                obj = json.loads(self.rfile.read(n).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return None
            return obj if isinstance(obj, dict) else None

        def do_GET(self):
            if not self._host_ok():
                return self._send(403, {"error": "host tidak diizinkan"})
            if self.path in ("/", "/index.html"):
                page = WEB_DIR / "index.html"
                if not page.exists():
                    return self._send(404, {"error": "web/memory/index.html tidak ditemukan"})
                return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/api/status":
                return self._send(200, {"ready": True, "notes": len(state.store.list()), "checkpoint": Path(state.checkpoint).name,
                                        "strictness": STRICTNESS})
            if self.path == "/api/notes":
                with state.lock:
                    return self._send(200, {"notes": state.notes()})
            return self._send(404, {"error": "tidak ditemukan"})

        def do_POST(self):
            if not self._host_ok():
                return self._send(403, {"error": "host tidak diizinkan"})
            body = self._json_body()
            if body is None:
                return self._send(400, {"error": "isi permintaan tidak valid"})
            if self.path == "/api/notes":
                text = str(body.get("text", "")).strip()
                if not text:
                    return self._send(400, {"error": "Catatan kosong."})
                if len(text) > MAX_NOTE:
                    return self._send(400, {"error": f"Catatan terlalu panjang (maks {MAX_NOTE} karakter)."})
                replace_id = body.get("replace_id")
                with state.lock:
                    nid = state.store.teach(text)
                    if isinstance(replace_id, int):
                        state.store.forget(replace_id)
                    return self._send(200, {"id": nid, "hints": note_hints(text), "notes": state.notes()})
            if self.path == "/api/ask":
                q = str(body.get("question", "")).strip()
                if not q:
                    return self._send(400, {"error": "Pertanyaan kosong."})
                if len(q) > MAX_QUESTION:
                    return self._send(400, {"error": f"Pertanyaan terlalu panjang (maks {MAX_QUESTION} karakter)."})
                level = str(body.get("strictness", "seimbang"))
                with state.lock:
                    state.agent.relevance_threshold = STRICTNESS.get(level, STRICTNESS["seimbang"])
                    t0 = time.time()
                    out = state.agent.ask(q)
                    out["seconds"] = round(time.time() - t0, 1)
                    out["empty"] = not state.store.list()
                    return self._send(200, out)
            return self._send(404, {"error": "tidak ditemukan"})

        def do_DELETE(self):
            if not self._host_ok():
                return self._send(403, {"error": "host tidak diizinkan"})
            m = re.match(r"^/api/notes/(\d+)$", self.path)
            if not m:
                return self._send(404, {"error": "tidak ditemukan"})
            with state.lock:
                ok = state.store.forget(int(m.group(1)))
                return self._send(200 if ok else 404, {"ok": ok, "notes": state.notes()})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "reports/checkpoints_100m_sft_mem2/gpt_sft_best.pt"))
    ap.add_argument("--db", default=str(PROJECT_ROOT / "data/memory/memory.db"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--open", action="store_true", help="Buka peramban otomatis.")
    args = ap.parse_args()

    port_error = (f"[web] Port {args.port} sudah dipakai program lain (mungkin Memori yang sudah berjalan).\n"
                  f"[web] Tutup yang lama, atau jalankan dengan port lain, mis.:  --port {args.port + 35}")
    try:  # periksa port SEBELUM memuat model (hemat waktu) dan agar kegagalan cepat terlihat
        probe = ExclusiveServer(("127.0.0.1", args.port), BaseHTTPRequestHandler)
        probe.server_close()
    except OSError:
        print(port_error, file=sys.stderr)
        sys.exit(1)
    state = State(args.checkpoint, args.db, args.k)
    try:
        server = ExclusiveServer(("127.0.0.1", args.port), make_handler(state, args.port))
    except OSError:
        print(port_error, file=sys.stderr)
        sys.exit(1)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"[web] buka {url}  (Ctrl+C untuk berhenti)", flush=True)
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] berhenti.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
