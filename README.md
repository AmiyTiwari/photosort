# PhotoSort (phases 1–5)

Import a large set of photos, keep **one stored copy** of each unique image, and browse it by
**factors** (date, place, camera, type). A photo can sit under many factors; each one is only a mapping
to the single stored copy. Near-duplicates (resized/re-saved copies, burst shots) are grouped against
the **whole existing library**, so later imports slot into what is already there.

## Setup

Prerequisites: Python 3.10-3.13 (3.13 recommended; 3.14 has no PyTorch wheels yet), git, and about 2.5 GB of disk for the virtualenv and the models. Runs on macOS (Apple Silicon uses the GPU) and Linux; Windows is untested.

```bash
git clone https://github.com/AmiyTiwari/photosort.git && cd photosort
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt        # pinned versions; ~1.5 GB incl. PyTorch
.venv/bin/python -m photosort models             # faces 39 MB + CLIP 577 MB + places 1 MB -> models/ (checksummed)
.venv/bin/python -m pytest -q tests              # 25 tests, ~12 s, offline: confirms the install
.venv/bin/python -m photosort serve              # open http://127.0.0.1:8765
```

Then in the browser: **Import** -> pick a folder of photos -> wait for the batch to finish -> browse. Originals are never modified or deleted.

**Assistant (optional).** Settings -> Assistant model: *Claude Code on this Mac* (needs the `claude` CLI installed and logged in; no key), or a provider with an API key (OpenAI, Gemini, Anthropic, OpenRouter), or a local Ollama / LM Studio model (no key). Click *Test connection*. Keys are stored only in `library/secrets.json` (mode 600) or `PHOTOSORT_LLM_API_KEY`; everything except the chat panel works with no model at all.

**Demo photos.** `.venv/bin/python scripts/make_testdata.py` downloads 70 freely available web images and builds `testdata/web_photos` (74 files: four trips with GPS, duplicates, a burst, screenshots; not committed). Run a second instance on them so your own library stays separate: `.venv/bin/python -m photosort --library library_web serve --port 8766`, then import the folder.

**Another machine.** Clone and repeat the steps above. To carry a library over, stop the server and copy its folder; all paths inside are relative.

## Docs and knowledge transfer

- **Onboarding & KT doc** (architecture with a diagram, data model, pipeline, code map, assistant internals, testing, tuning knobs, decisions, roadmap, and the KT video script): https://claude.ai/artifact/DKsVwQx7tsdNLZ9uEvj2PG
- **Architecture drawing:** `docs/architecture.png`.
- **KT video** (about nine narrated minutes through every screen, the architecture and the code): generated, not committed. Rebuild it with `.venv/bin/python scripts/make_kt_video.py` (needs macOS for the narration, `brew install ffmpeg`, `pip install playwright && python -m playwright install chromium`, the demo library on port 8766); it writes `media/PhotoSort-KT.mp4` and a chapter list next to it.

## Everyday commands

```bash
.venv/bin/python -m photosort serve [--port 8765] [--library DIR]        # web UI + API (localhost only, no auth)
.venv/bin/python -m photosort import ~/Pictures/backup [--mode hardlink] # blocking import from the CLI
.venv/bin/python -m photosort analyze                                    # re-run faces / CLIP / people / places / events / rules / questions
.venv/bin/python -m photosort export date ~/Desktop/views                # hardlink folder tree for one factor
.venv/bin/python -m photosort models [faces|clip|places]                 # (re)download local models
```

The library defaults to `./library`; override with `--library` or `PHOTOSORT_LIBRARY`.

## How it works

| Step | What happens |
|---|---|
| Scan | Recursive walk for jpg/png/heic/webp/tiff/gif/bmp. Skips hidden dirs and `.photoslibrary` bundles. |
| Resume | Files already recorded with the same path + size + mtime are skipped, so re-running an import only processes new or changed files. |
| Analyse | Thread pool: SHA-256, EXIF (date, GPS, camera), pHash and dHash, sharpness score, 320px thumbnail. |
| Exact dedup | Same SHA-256 means the file is not stored again; only a `sources` row is added. |
| Store | `library/store/ab/cd/<sha256>.<ext>`, copied (default) or hardlinked (same disk, no extra space). Originals are never modified or deleted. |
| Factors | Rule-based mapping into `photo_factor`: Date (year › month), Place (0.1° grid), Camera, Type (photo/screenshot). |
| Near-dups | A BK-tree over pHash of the whole library. **variant**: pHash ≤ 4 and dHash ≤ 8. **burst**: pHash ≤ 14 and taken within 10 s. Groups are merged with union-find across imports. |
| Best copy | Most pixels, then a real EXIF date, then sharpest, then largest file. You can override it in the UI, and your choice is kept on later imports. |
| Export | `Views/<Factor>/<value>/…` made of hardlinks to the stored copy (symlinks across disks). |

## Phase 3: people, events, review queue

| Part | How |
|---|---|
| Faces | OpenCV YuNet (detector) + SFace (128-d embedding), run locally on a ≤1280px copy. Faces under 40px are ignored. |
| People | Incremental clustering against **existing** person centroids. Similarity ≥ 0.45 assigns silently; 0.363–0.45 assigns and, if the person is named, asks you to confirm; lower starts a new "Person N". Unnamed clusters ≥ 0.55 alike are merged automatically, and named people are never merged without you. "Not them" is remembered and that face is never put back in that cluster. Naming two clusters the same merges them. |
| Events | Photos with a real capture time (EXIF or filename) are split where the gap is more than `event_gap_hours` (default 6) or the location jumps more than 100 km. On every run, new segments are matched to existing events by overlap, so an event keeps its id and your name when late photos arrive. |
| Dates from filenames | `IMG_20250101_120000`, `PXL_…`, `2026-06-28_194557`, `Screenshot … at 5.50.49 PM`, `WhatsApp Image … at 21.04.33`. |
| Review queue | "Who is this?", "Same person?", "Is this X?", "What was this event?". Ordered by how many photos one answer sorts. Answered or skipped questions are not asked again. |
| Custom groups | A JSON rule (`person`, `any_person`, `event`, `date_from/to`, `months`, `near`, `camera`, `path`, `screenshot`, `min/max_people`, `all/any/not`) compiled to SQL. People and events are saved by id, so renaming doesn't break a rule. Rules are re-applied after every import. Manual collections also work. |

## Phase 4: assistant, model-independent

`photosort/llm.py` defines one interface, `chat(system, messages, tools) -> Reply`, with a neutral message and tool format, plus adapters:

- **openai**: any OpenAI-compatible `/chat/completions` endpoint, which covers OpenAI, Google Gemini (OpenAI endpoint), OpenRouter, Groq, Ollama, LM Studio, vLLM, llama.cpp and others.
- **anthropic**: the Anthropic Messages API. Assistant turns are replayed unchanged (Claude 5 models bind their reasoning to the exact history; if the history must be trimmed, the reasoning blocks are stripped from replayed turns, which is accepted). A `binding_check` diagnostic opts into the API's own check and records `input_transformations`.
- **claude_code** (`photosort/claude_code.py`): no API key. Runs the `claude` CLI installed on the machine in headless mode (`claude -p`) with the app's tools attached through an MCP server (`photosort/mcp_server.py`, stdio JSON-RPC). Claude Code supplies the login, the model and the tool loop; the app supplies the system prompt, the tools and the UI side channel (choices / photos / faces shown). Conversations continue through Claude Code sessions (`--session-id` / `--resume`; "New chat" starts a fresh one). Built-in tools are disabled (`--tools ""`), only the PhotoSort MCP server is loaded (`--strict-mcp-config`), the parent shell's `CLAUDE_EFFORT` is not inherited and the effort comes from Settings.
- **none**: no model. Everything except chat still works, including the review queue.

The MCP server works with any MCP client, not only Claude Code: `python -m photosort.mcp_server --library <dir>`.

To add another provider, write one class with `chat()`. The model must support tool calling. Configure it in **Settings → Assistant model** (provider, base URL, model id, key, "can see images"). The key is kept in `library/secrets.json` (mode 600) or `PHOTOSORT_LLM_API_KEY`, and is never sent back to the browser.

The agent (`photosort/agent.py`) runs a tool loop of up to 10 steps per message. Its tools: `library_overview`, `list_people`, `list_events`, `list_factors`, `find_photos(rule)`, `name_person`, `merge_people`, `rename_event`, `create_custom_value(factor, value, rule|photo_ids)`, `delete_custom_value`, `update_settings`, `list_questions`, `answer_question`, `show` (faces or photos under the reply), `offer_choices` (quick-reply buttons), and `look_at_photos` (only when vision is enabled). On the first chat it interviews you about how you want things sorted. Tool errors go back to the model rather than crashing, and the conversation is kept in `chat_messages`.

## Phase 5: content, learning, places

| Part | How |
|---|---|
| Content understanding | CLIP ViT-B/32 (open_clip, `laion2b_s34b_b79k`) embeds each photo's thumbnail once, locally, on the Apple GPU (MPS) when available. Stored as float16 in `photo_embeddings`. |
| Content groups | Factor **Content**. A group is a few phrases ("a photo of a beach"). Each photo's score against them is compared with generic negative phrases (zero-shot) and assigned above a strictness threshold. Screenshots are left out unless the group is about screenshots. Defaults: Documents & receipts, Food, Pets, Beach & sea, Nature & mountains, which you can edit or delete. |
| Learning from answers | Review shows "Which of these belong in X?" for the photos the group is least sure about. Each ✓/✗ is saved as a training example. Once there are 3 or more of each, a small logistic-regression classifier on the embeddings is blended with the zero-shot score, weighted towards your examples as they grow. Then the next batch is asked. |
| Manual collections | A custom group without a rule learns from the photos you put in it and asks "Add these to X?" for the most similar ones. It suggests but never auto-adds. |
| Plain-language search | Library search box and the assistant's `find_photos(text=…)`. Results score ≥ 0.22 and within 0.06 of the best match, so searching for something absent returns nothing. Can be combined with a rule (person, place, date). |
| Near-duplicates | Besides the hash checks, two photos taken within 60 s whose CLIP embeddings match at ≥ 0.90 join a burst group. This catches different framing or exposure that hashes miss. |
| Place names | Offline GeoNames cities (population 15k+), regions and countries (CC-BY 4.0). Factor **Places** becomes Country › Region › City. The bigger well-known city wins over a closer suburb. Events are labelled like "Calangute · 20–22 Dec 2025". Rule keys `place` and `content` work by name. |

## Live test on Claude Fable 5.1 (25 Nov 2026)

Library: `testdata/web_photos`, built by `scripts/make_testdata.py` (74 web images arranged as a phone backup with simulated EXIF dates, GPS and camera tags; not committed, `SOURCES.md` lists the origins). Pipeline results before any chat: 1 exact duplicate, 4 near-duplicate groups (incl. a portrait at four resolutions and a re-cropped scene), 22 faces in 7 people with no mixing, 6 place-named events, content groups Beach 11 / Nature 7 / Food 3 / Pets 2 / Documents 2 / Buildings 2.

Seven user messages through the `claude_code` adapter (model `claude-fable-5-1`), all correct in effect:

| # | Message | What it did | Tool calls | Time | Cost (list) |
|---|---|---|---|---|---|
| 1 | "Help me organise my photos." | Overview, showed the largest cluster's faces, offered sorting choices | 7 | 16 s | $0.23 |
| 2 | "All of the above. That's Barack Obama." | Named him; looked at trip photos to describe them; asked for the rest | 15 | 70 s | $0.79 |
| 3 | Three more names + event names | 3× name_person, 6× rename_event | 22 | 56 s | $1.32 |
| 4 | "Keep duplicates; make a Presidents group with Obama or Biden" | Rule `{"any_person": [Obama, Biden]}` → 12 photos, showed 6 | 6 | 35 s | $1.51 |
| 5 | "Find my dog photos; anything with a sunset?" | Text search; looked at results and dropped a false positive | 6 | 27 s | $1.68 |
| 6 | Rename event, report the queue | rename_event, several review answers, list_questions | 12 | 65 s | $2.19 |

Cost was dominated by (a) the parent shell's `CLAUDE_EFFORT=max` leaking into the child (fixed: effort is explicit now), (b) the model answering 16 review-queue questions on its own after looking at 80 thumbnails (the prompt now leaves the queue to the user unless asked), and (c) 64-character photo ids in tool output (now 12-character prefixes). Message 7 on the fixed setup ("How many photos of Kit Harington, which events, show them"): correct, 3 tool calls, 15 s, 155k cached tokens read across the 3 calls, 540 output tokens. Claude Code's own `total_cost_usd` for that run ($2.27) does not reconcile with list prices at those token counts (about $0.20), so treat its per-run cost figure as indicative; the token counts in each assistant message's `meta.usage` are the reliable signal.

## UI

- **Library**: factor tree with counts, a search box ("sunset over water"), a thumbnail grid (`⧉n` = near-dup group, `×n` = identical files merged) and a detail panel with factors, EXIF and every original path.
- **Import**: folder picker, copy/hardlink mode, live progress per batch.
- **Duplicates**: group cards with the best copy marked ★. Click to keep a different one, mark "Not a duplicate", or turn on "hide all but best".
- **Export**: create hardlink folder views per factor.
- **People**: face clusters. Name them, merge them, or click a face to remove it from a person.
- **Review**: the question queue, including "which of these belong?" photo pickers.
- **Settings**: AI provider (incl. "Claude Code on this Mac", no key), effort, event settings, re-analyse, content groups, custom groups.
- **Assistant** (button at the top right): chat panel with faces and photos shown inline and quick-reply chips.

## Layout

```
photosort/
  analyze.py   per-file hashing, EXIF, pHash, thumbnail
  similar.py   BK-tree and near-dup rules
  library.py   store, import pipeline, factor mapping, groups, export
  faces.py     YuNet + SFace face engine (optional; app runs without it)
  people.py    incremental face clustering, naming, merging, rejections
  events.py    time/location segmentation with stable event identity
  rules.py     JSON rule language -> SQL, for custom groups and search
  questions.py review queue
  clip.py      CLIP image/text embeddings (optional)
  learning.py  content groups, learning from answers, suggestions, text search, CLIP near-dups
  places.py    offline reverse geocoding (GeoNames)
  llm.py       provider-neutral model interface + adapters (OpenAI-compatible, Anthropic)
  claude_code.py  adapter that drives `claude -p` with the MCP server attached (no API key)
  mcp_server.py   the assistant's tools as an MCP server (stdio)
  agent.py     assistant tools and loop
  db.py        SQLite schema (WAL) + migrations
  api.py       FastAPI endpoints and static UI
  static/      index.html, app.js, style.css (no build step)
tests/         synthetic-photo tests: dedup, groups, resume, incremental, export, API, MCP server,
               Claude Code adapter (against tests/fake_claude.py, a stand-in for the CLI)
scripts/       make_testdata.py (builds testdata/web_photos from the web), make_kt_video.py (records the KT video)
docs/          architecture.png
models/        face models (downloaded)
library/       created at runtime: catalog.db, store/, thumbs/, faces/, secrets.json
```

## Limits and next steps

- Embeddings are held in memory as one matrix: about 1 KB per photo, so 100k photos ≈ 100 MB. For a million or more photos, swap `EmbeddingCache` for a vector index such as FAISS or sqlite-vec.
- Search and content quality are those of CLIP ViT-B/32. A larger open_clip model is a one-line change in `clip.py`, but slower.
- City names cover places with population 15k+. Rural photos fall back to region or country.
