#!/usr/bin/env python3
"""Record the PhotoSort knowledge-transfer video (about nine minutes, 1440x900, narrated).

Playwright drives the real app in headless Chromium and records each chapter; macOS `say`
narrates; ffmpeg muxes and concatenates. The demo library (testdata/web_photos imported into
library_web) must be served on port 8766; a fresh instance on port 8767 is started for the
live-import chapter and stopped afterwards.

    .venv/bin/python scripts/make_kt_video.py [--out media/PhotoSort-KT.mp4] [--skip-live-chat]

Requires: macOS (for `say`), ffmpeg on PATH (brew install ffmpeg), playwright + chromium in the venv
(pip install playwright && python -m playwright install chromium), testdata/web_photos.
"""
import argparse
import html
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DEMO, FRESH = "http://127.0.0.1:8766", "http://127.0.0.1:8767"
W, H = 1440, 900
VOICE, RATE = "Samantha", 170
REPO = "https://github.com/AmiyTiwari/photosort"
DOC = "https://claude.ai/artifact/DKsVwQx7tsdNLZ9uEvj2PG"

# ---------------------------------------------------------------------------------------------- narration
# Each chapter: title + narration parts. A part after the first is spoken when the recorder calls
# ch.mark() (e.g. once an import has finished), or right after the previous part if no mark comes.
CHAPTERS = [
    ("intro", "What PhotoSort is", [
        "This is PhotoSort, a photo organiser that keeps one copy of every photo and sorts it many ways. "
        "You import folders such as phone backups and WhatsApp exports. It removes duplicates, finds people, "
        "trips, places and content, and asks you only the questions worth asking. Everything runs on your own "
        "machine. An A I assistant is optional and works with any model provider. In the next few minutes we "
        "walk through every screen, the architecture, and the code."]),
    ("architecture", "Architecture", [
        "The browser is a single static page talking JSON to a FastAPI server. FastAPI calls one object, "
        "Library, which owns the import pipeline, the analysis stage, every mutation, and the export. Library "
        "delegates to four groups of modules: per-file hashing and near-duplicate detection; faces and people; "
        "content and search with CLIP; and events, places, rules and the review queue. Underneath sits one "
        "SQLite file plus folders for the stored copies, thumbnails and face crops. The models, a face detector "
        "and recogniser, CLIP, and a GeoNames index, are downloaded once and run locally. On the right is the "
        "assistant: the agent runs a tool loop against the same Library methods, a model adapter speaks to any "
        "provider, and for Claude Code an M C P server exposes the same tools to the command line."]),
    ("import", "Importing a folder", [
        "Let's import a folder into a fresh library. This is the demo backup: a Camera folder, WhatsApp copies, "
        "screenshots and a download. Click Import. Every file is hashed, its EXIF read, two perceptual hashes "
        "computed and a thumbnail made, in a thread pool. A hash seen before only adds a source path; new "
        "photos are copied into the store under their hash. Then the analysis stage runs: faces, CLIP "
        "embeddings, people clustering, place names, events, saved rules and the review queue.",
        "Done. Seventy-four files became seventy-three unique photos: one exact duplicate, four near-duplicate "
        "groups and twenty-two faces. Running the same import again would skip every unchanged file."]),
    ("library", "Library screen", [
        "The Library screen. On the left, every factor with counts: date, people, events, content, places, "
        "camera and type. Clicking a value filters the grid; nothing is copied, these are rows in a mapping "
        "table. Opening a photo shows what the pipeline knows: face boxes, a chip for every factor, the "
        "camera, and every original file that turned out to be this photo. Here the Camera original and the "
        "WhatsApp copy sit under one entry. The search box asks CLIP, so you can type what the picture shows, "
        "such as sunset over water."]),
    ("duplicates", "Duplicates", [
        "Exact copies never reach this screen; they were merged on import. These groups are the same picture "
        "re-saved or resized, or shots taken seconds apart. The starred copy is the one kept visible: most "
        "pixels, then a real camera date, then the sharpest. Clicking another copy keeps that one instead, and "
        "Not a duplicate splits it out. The switch hides everything but the best copy across the whole library."]),
    ("people", "People", [
        "People are face clusters. Faces are found by YuNet and embedded by SFace, two small local models. "
        "A new face joins the closest existing person when it is similar enough; otherwise it starts a new "
        "one. Opening a person shows every face with its match confidence. Clicking a face that is not them "
        "removes it, and it never comes back. Name someone once, and every later import files their photos "
        "under that name."]),
    ("events", "Events and places", [
        "Events are runs of photos close in time, split where the gap passes a setting or the location jumps. "
        "Names come from an offline GeoNames index, country, region and city, plus the date. In Settings the "
        "gap can be changed. Raising it to twenty-four hours merges the two Goa days into one trip, and the "
        "name survives, because events are matched by their photos, not by their dates."]),
    ("content", "Content groups", [
        "Content groups sort by what is in the picture. Each group is a few short phrases; a photo's CLIP "
        "score against them is compared with generic phrases, and the group is assigned above a strictness "
        "threshold. The defaults cover documents, food, pets, beach, nature and buildings. Let's add one: "
        "Sunsets, described by two phrases. It appears in the Library at once, and it keeps learning from your "
        "answers in the review queue."]),
    ("review", "Review queue", [
        "The review queue asks the few questions that sort the most photos, highest impact first: who is this, "
        "is this the same person, what was this event, and which of these belong in a group. For a group "
        "question you click photos to mark them in or out. Every answer becomes a training example, and the "
        "next batch is asked. Skipped questions are never asked again."]),
    ("assistant", "The assistant", [
        "The assistant is a tool-calling agent over the same library operations. It is not tied to a vendor: "
        "one chat interface, and an adapter per provider: OpenAI-compatible A P Is, Anthropic, or Claude Code "
        "on this machine through M C P, which needs no A P I key. Here is an earlier conversation where it "
        "named people, created a Presidents group from a rule, and found dog and sunset photos, showing faces "
        "and pictures inline. Let's ask it something new and wait for the answer.",
        "It answered from the library, with the people and events it knows. The review queue stays yours; it "
        "only decides things when you ask it to."]),
    ("code", "Code tour", [
        "The code is on GitHub. library dot py is the hub: start import runs the pipeline on a background "
        "thread, ingest one stores a single file, and analyse lists the analysis stage in order. agent dot py "
        "holds the system prompt, the tool schemas and the loop; every tool is a plain method next to its "
        "schema. l l m dot py has the provider adapters; the two to-wire functions are the whole difference "
        "between providers. The README has the setup steps, and the onboarding document has the full "
        "architecture and this walkthrough as a script."]),
    ("tests", "Tests", [
        "The test suite: twenty-five tests, all passing in about twelve seconds, with no network and no models, "
        "because every heavy dependency has a deterministic stand-in."]),
    ("closing", "Getting started", [
        "To get started: clone the repository, create the virtual environment, download the models, run the "
        "tests, and start the server. If you remember one thing: the store holds one copy of everything, and "
        "every feature is a mapping onto it that gets better each time you answer a question. Thanks for "
        "watching."]),
]

CURSOR_JS = """(() => {
  const css = `#__cur{position:fixed;left:-50px;top:-50px;width:22px;height:22px;border-radius:50%;
    background:rgba(47,111,219,.35);border:2px solid rgba(47,111,219,.95);pointer-events:none;
    z-index:2147483647;transform:translate(-50%,-50%)}
    #__cur.down{background:rgba(47,111,219,.8);transform:translate(-50%,-50%) scale(.75)}`;
  document.addEventListener('DOMContentLoaded', () => {
    const s = document.createElement('style'); s.textContent = css; document.head.appendChild(s);
    const c = document.createElement('div'); c.id = '__cur'; document.body.appendChild(c);
    document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
    document.addEventListener('mousedown', () => c.classList.add('down'), true);
    document.addEventListener('mouseup', () => c.classList.remove('down'), true);
  });
})();"""

CARD_CSS = """body{margin:0;background:#f6f6f4;color:#1d1d1b;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
display:flex;align-items:center;justify-content:center;height:100vh}.card{max-width:1180px;padding:0 60px}
h1{font-size:64px;margin:0 0 12px;letter-spacing:-.02em}h1 span{color:#2f6fdb}h2{font-size:30px;font-weight:500;color:#6b6b66;margin:0 0 40px}
ul{font-size:27px;line-height:1.6;margin:0;padding-left:28px}li b{color:#2f6fdb}code{font-family:ui-monospace,Menlo,monospace;font-size:24px;
background:#e6eefc;padding:2px 8px;border-radius:6px}.small{font-size:22px;color:#6b6b66;margin-top:36px}
pre{background:#1d1d1b;color:#ecebe7;font:19px/1.5 ui-monospace,Menlo,monospace;padding:28px 34px;border-radius:12px;white-space:pre-wrap}
pre .ok{color:#5cc27f}pre .cmd{color:#9ecbff}img{max-width:1320px;max-height:820px;border-radius:10px;box-shadow:0 8px 40px rgba(0,0,0,.12)}"""


# ---------------------------------------------------------------------------------------------- helpers

def say(text: str, out: Path) -> float:
    subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(out), "--data-format=LEI16@22050", text], check=True)
    with wave.open(str(out)) as w:
        return w.getnframes() / w.getframerate()


def build_audio(parts: list[Path], starts: list[float], total: float, out: Path) -> None:
    """One wav: each part at its start time, silence elsewhere, `total` seconds long."""
    rate = 22050
    buf = bytearray(b"\x00\x00" * int(total * rate))
    for p, start in zip(parts, starts):
        with wave.open(str(p)) as w:
            data = w.readframes(w.getnframes())
        off = int(start * rate) * 2
        buf[off:off + len(data)] = data
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(buf))


def api(url: str, body=None, method=None) -> dict | list | None:
    req = urllib.request.Request(url, method=method or ("POST" if body is not None else "GET"),
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read() or b"null")
    except Exception:
        return None


def wait_up(url: str, seconds: int = 60) -> None:
    for _ in range(seconds):
        if api(url + "/api/stats"):
            return
        time.sleep(1)
    sys.exit(f"{url} is not answering")


class Chapter:
    """Clock, pacing and narration placement for one recorded chapter."""

    def __init__(self, key: str, page, part_durations: list[float]):
        self.key, self.page, self.durs = key, page, part_durations
        self.t0 = time.monotonic()
        self.starts = [0.0]

    def t(self) -> float:
        return time.monotonic() - self.t0

    def until(self, seconds: float) -> None:
        while self.t() < seconds:
            time.sleep(0.1)

    def narration_end(self) -> float:
        return self.starts[-1] + self.durs[len(self.starts) - 1]

    def mark(self) -> None:
        if len(self.starts) < len(self.durs):
            self.starts.append(max(self.narration_end(), self.t()))

    def finish(self) -> float:
        while len(self.starts) < len(self.durs):
            self.starts.append(self.narration_end())
        end = self.narration_end() + 1.0
        self.until(end)
        return end

    # -- actions, paced so the cursor visibly travels
    def click(self, selector: str, nth: int = 0, settle: float = 0.6) -> None:
        loc = self.page.locator(selector).nth(nth)
        loc.scroll_into_view_if_needed()
        box = loc.bounding_box()
        if not box:
            return
        self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=28)
        self.page.mouse.down()
        time.sleep(0.1)
        self.page.mouse.up()
        time.sleep(settle)

    def type(self, selector: str, text: str) -> None:
        self.click(selector, settle=0.3)
        self.page.keyboard.type(text, delay=40)

    def nav(self, view: str) -> None:
        self.click(f"nav button[data-view={view}]", settle=1.0)


# ---------------------------------------------------------------------------------------------- chapters

def rec_intro(ch: Chapter, cards: Path):
    ch.page.goto((cards / "intro.html").as_uri())


def rec_architecture(ch: Chapter, cards: Path):
    ch.page.goto((cards / "architecture.html").as_uri())


def rec_import(ch: Chapter, cards: Path):
    p = ch.page
    p.goto(FRESH + "/")
    p.wait_for_selector("#import-path")
    ch.until(3)
    ch.type("#import-path", str(ROOT / "testdata" / "web_photos"))
    ch.until(9)
    ch.click("#import-start")
    time.sleep(1.2)
    p.locator("#batches").scroll_into_view_if_needed()  # keep the progress card in view
    for _ in range(240):  # the batch card shows the stage while it runs
        time.sleep(1)
        b = (api(FRESH + "/api/imports") or [{}])[0]
        if b.get("status") in ("done", "failed"):
            break
    time.sleep(1.5)
    ch.mark()


def rec_library(ch: Chapter, cards: Path):
    p = ch.page
    p.goto(DEMO + "/")
    p.wait_for_selector("#factors .f-val")
    ch.until(7)
    ch.click('.f-val:has-text("White House visit")')
    ch.until(13)
    ch.click('.f-val:has-text("Goa trip")', nth=1)  # the 20 Dec day: it holds the photo with a WhatsApp copy
    p.wait_for_selector(".tile:has(.tag.left)", timeout=10000)
    ch.until(17)
    ch.click(".tile:has(.tag.left)")  # the photo that also arrived as a WhatsApp copy
    p.wait_for_selector("#detail .src")
    ch.until(24)
    p.locator("#detail .src").first.scroll_into_view_if_needed()
    ch.until(34)
    ch.type("#search-q", "sunset over water")
    p.keyboard.press("Enter")


def rec_duplicates(ch: Chapter, cards: Path):
    p = ch.page
    p.goto(DEMO + "/")
    p.wait_for_selector("#factors .f-val")
    ch.nav("dupes")
    p.wait_for_selector("#groups .group")
    ch.until(10)
    p.locator("#groups .group").nth(1).scroll_into_view_if_needed()
    ch.until(17)
    ch.click("#groups .group .member .tile", nth=1)
    ch.until(24)
    ch.click("#hide-non-best")
    ch.until(31)
    ch.click("#hide-non-best")


def rec_people(ch: Chapter, cards: Path):
    p = ch.page
    p.goto(DEMO + "/")
    p.wait_for_selector("#factors .f-val")
    ch.nav("people")
    p.wait_for_selector("#people-grid .person")
    ch.until(12)
    ch.click('.person:has-text("Kit Harington")')
    p.wait_for_selector("#person-detail .face")
    ch.until(16)
    p.locator("#person-detail").scroll_into_view_if_needed()


def rec_events(ch: Chapter, cards: Path):
    p = ch.page
    p.goto(DEMO + "/")
    p.wait_for_selector("#factors .f-val")
    ch.until(4)
    ch.click('.f-val:has-text("Goa trip")')
    ch.until(12)
    ch.nav("settings")
    p.wait_for_selector("#set-gap")
    p.locator("#set-gap").scroll_into_view_if_needed()
    ch.until(16)
    p.locator("#set-gap").fill("24")
    ch.click("#set-save")
    ch.until(21)
    ch.nav("browse")
    p.wait_for_selector('.f-val:has-text("Goa trip")')
    ch.click('.f-val:has-text("Goa trip")')
    ch.until(29)
    api(DEMO + "/api/settings", {"event_gap_hours": 6})  # restore the demo library quietly


def rec_content(ch: Chapter, cards: Path):
    p = ch.page
    p.goto(DEMO + "/")
    p.wait_for_selector("#factors .f-val")
    ch.nav("settings")
    p.wait_for_selector("#cg-list .custom-item")
    p.locator("#cg-list").scroll_into_view_if_needed()
    ch.until(16)
    p.locator("#cg-name").scroll_into_view_if_needed()
    ch.type("#cg-name", "Sunsets")
    ch.click("#cg-prompts", settle=0.2)
    p.keyboard.type("a photo of a sunset\na photo of the sky at dusk", delay=35)
    ch.click("#cg-save")
    p.wait_for_function("document.querySelector('#cg-result').textContent.startsWith('Saved')")
    ch.until(26)
    ch.nav("browse")
    p.wait_for_selector('.f-val:has-text("Sunsets")')
    ch.click('.f-val:has-text("Sunsets")')


def rec_review(ch: Chapter, cards: Path):
    p = ch.page
    p.goto(DEMO + "/")
    p.wait_for_selector("#factors .f-val")
    ch.nav("review")
    p.wait_for_selector("#questions .q")
    ch.until(12)
    if p.locator("#questions .pick").count() >= 3:
        ch.click("#questions .pick", 0)
        ch.click("#questions .pick", 1)
        ch.click("#questions .pick", 2, settle=0.2)
        ch.click("#questions .pick", 2)
        ch.until(22)
        ch.click('#questions button:has-text("Rest belong")')


def rec_assistant(ch: Chapter, cards: Path, live: bool = True):
    p = ch.page
    p.goto(DEMO + "/")
    p.wait_for_selector("#factors .f-val")
    ch.until(2)
    ch.click("#chat-toggle")
    p.wait_for_selector("#chat-log .msg")
    time.sleep(1)
    p.evaluate("document.querySelector('#chat-log').scrollTop = 0")
    for i in range(1, 9):  # scroll through the earlier conversation
        ch.until(6 + i * 3)
        p.evaluate(f"document.querySelector('#chat-log').scrollTop = {i * 420}")
    ch.until(32)
    if not live:
        return
    before = p.locator("#chat-log .msg.assistant").count()
    ch.type("#chat-input", "Which people appear in the Theatre night event, and how many photos does each have?")
    p.keyboard.press("Enter")
    for _ in range(240):
        time.sleep(1)
        # the "Thinking..." placeholder is an assistant message too; wait for it to be replaced by the reply
        if p.locator("#chat-log .msg.assistant").count() > before and p.locator("#chat-log .typing").count() == 0:
            break
    time.sleep(1)
    ch.mark()


def rec_code(ch: Chapter, cards: Path):
    p = ch.page
    for i, name in enumerate(["library.py", "agent.py", "llm.py"]):
        try:
            p.goto(f"{REPO}/blob/main/photosort/{name}?plain=1", wait_until="domcontentloaded", timeout=25000)
            p.wait_for_selector("textarea#read-only-cursor-text-area, .react-code-lines, table", timeout=15000)
        except Exception:
            p.goto((cards / f"code-{name}.html").as_uri())
        ch.until(2 + i * 13)
        for _ in range(5):
            p.mouse.wheel(0, 360)
            time.sleep(1.4)


def rec_tests(ch: Chapter, cards: Path):
    ch.page.goto((cards / "tests.html").as_uri())


def rec_closing(ch: Chapter, cards: Path):
    ch.page.goto((cards / "closing.html").as_uri())


# ---------------------------------------------------------------------------------------------- cards

def write_cards(cards: Path, pytest_output: str) -> None:
    cards.mkdir(parents=True, exist_ok=True)

    def card(name, body):
        (cards / f"{name}.html").write_text(f"<!doctype html><meta charset='utf-8'><style>{CARD_CSS}</style><div class='card'>{body}</div>")

    card("intro", "<h1><span>PhotoSort</span> — knowledge transfer</h1><h2>One copy of every photo, sorted many ways, on your own machine</h2>"
         "<ul><li><b>Import</b> folders; exact and near-duplicates merged</li><li><b>People, events, places, content</b> found by local models</li>"
         "<li><b>Review queue</b> asks only the questions worth asking</li><li><b>Assistant</b> that works with any model provider</li></ul>"
         f"<div class='small'>{REPO}</div>")
    card("architecture", f"<img src='{(ROOT / 'docs' / 'architecture.png').as_uri()}'>")
    out = html.escape(pytest_output).replace("passed", "<span class='ok'>passed</span>")
    card("tests", f"<pre><span class='cmd'>$ .venv/bin/python -m pytest -q tests</span>\n{out}</pre>")
    card("closing", "<h1>Getting <span>started</span></h1><ul>"
         f"<li><code>git clone {REPO}.git</code></li><li><code>python3.13 -m venv .venv &amp;&amp; .venv/bin/pip install -r requirements.txt</code></li>"
         "<li><code>.venv/bin/python -m photosort models</code> &nbsp;·&nbsp; <code>pytest -q tests</code></li>"
         "<li><code>.venv/bin/python -m photosort serve</code> → http://127.0.0.1:8765</li></ul>"
         f"<div class='small'>README: setup and operations · Onboarding &amp; KT doc: {DOC}</div>")
    for name in ["library.py", "agent.py", "llm.py"]:
        src = (ROOT / "photosort" / name).read_text().splitlines()[:140]
        body = "\n".join(f"{i + 1:4d}  {html.escape(l)}" for i, l in enumerate(src))
        card(f"code-{name}", f"<pre style='font-size:15px;max-height:860px;overflow:hidden'><span class='cmd'>photosort/{name}</span>\n{body}</pre>")


# ---------------------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "media" / "PhotoSort-KT.mp4"))
    ap.add_argument("--skip-live-chat", action="store_true", help="show the earlier conversation only (no model call)")
    ap.add_argument("--only", help="comma-separated chapter keys, for iterating on one chapter")
    args = ap.parse_args()
    ffmpeg = shutil.which("ffmpeg") or sys.exit("ffmpeg not found (brew install ffmpeg)")
    out = Path(args.out).resolve()
    work = out.parent / "kt-work"
    work.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None

    wait_up(DEMO)
    for g in api(DEMO + "/api/content") or []:
        if g["name"] == "Sunsets":  # the content chapter creates it
            api(f"{DEMO}/api/custom/{g['id']}", method="DELETE")
    fresh = None
    if only is None or "import" in only:
        shutil.rmtree(ROOT / "library_kt", ignore_errors=True)
        fresh = subprocess.Popen([sys.executable, "-m", "photosort", "--library", "library_kt", "serve", "--port", "8767"],
                                 cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_up(FRESH)

    print("pytest for the tests card...", flush=True)
    pytest_output = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    write_cards(work / "cards", pytest_output.splitlines()[-1] if pytest_output else "(no output)")

    print("narration...", flush=True)
    durations = {}
    for key, _, parts in CHAPTERS:
        durations[key] = [say(text, work / f"{key}-{i}.wav") for i, text in enumerate(parts)]
    print("  total narration %.0f s" % sum(sum(d) for d in durations.values()), flush=True)

    segments, chapters_txt, cursor = [], [], 0.0
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for key, title, parts in CHAPTERS:
                if only and key not in only:
                    continue
                print(f"recording {key}...", flush=True)
                ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=str(work / "video"),
                                          record_video_size={"width": W, "height": H}, device_scale_factor=1)
                ctx.add_init_script(CURSOR_JS)
                page = ctx.new_page()
                ch = Chapter(key, page, durations[key])
                rec = globals()[f"rec_{key}"]
                try:
                    if key == "assistant":
                        rec(ch, work / "cards", live=not args.skip_live_chat)
                    else:
                        rec(ch, work / "cards")
                except Exception as e:  # keep going: the chapter still gets its narration
                    print(f"  {key}: action failed, continuing ({e})", file=sys.stderr)
                length = ch.finish()
                ctx.close()
                video = Path(page.video.path())
                audio = work / f"{key}.wav"
                build_audio([work / f"{key}-{i}.wav" for i in range(len(parts))], ch.starts, length, audio)
                seg = work / f"{key}.mp4"
                subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-i", str(audio),
                                "-map", "0:v:0", "-map", "1:a:0", "-vf", f"scale={W}:{H}:flags=lanczos,fps=30,format=yuv420p",
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
                                "-af", "apad", "-shortest", "-movflags", "+faststart", str(seg)], check=True)
                segments.append(seg)
                chapters_txt.append(f"{int(cursor // 60):02d}:{int(cursor % 60):02d}  {title}")
                cursor += length
            browser.close()
    finally:
        if fresh:
            fresh.terminate()

    lst = work / "segments.txt"
    lst.write_text("".join(f"file '{s.resolve()}'\n" for s in segments))  # concat resolves relative to the list
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check=True)
    out.with_suffix(".chapters.txt").write_text("\n".join(chapters_txt) + "\n")
    print(f"wrote {out} ({out.stat().st_size // 1_000_000} MB, {int(cursor // 60)} min {int(cursor % 60)} s)")
    print("\n".join(chapters_txt))


if __name__ == "__main__":
    main()
