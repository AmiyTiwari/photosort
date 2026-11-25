"use strict";

const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "class") n.className = v;
    else if (v !== false && v != null) n.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null) n.append(k.nodeType ? k : String(k));
  return n;
};
const api = async (path, body, method) => {
  const r = await fetch(path, body === undefined ? { method: method || "GET" } : {
    method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
};
const fmtBytes = (b) => {
  const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return `${b.toFixed(i ? 1 : 0)} ${u[i]}`;
};
const fmtDate = (iso) => iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "";
const thumb = (sha) => `/thumbs/${sha}.jpg`;
const faceImg = (id) => `/faces/${id}.jpg`;

const state = { valueId: null, title: "All photos", offset: 0, showHidden: false, selected: null, query: "" };

// ------------------------------------------------------------------ nav

function show(view) {
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  document.querySelectorAll(".view").forEach((v) => (v.hidden = v.id !== `view-${view}`));
  if (view === "browse") { loadFactors(); loadGrid(true); }
  if (view === "import") { loadBatches(); if (!$("#fs-path").textContent) browse("~"); }
  if (view === "dupes") loadGroups(true);
  if (view === "export") loadExport();
  if (view === "people") loadPeople();
  if (view === "review") loadQuestions();
  if (view === "settings") loadSettings();
}
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));

async function loadStats() {
  const s = await api("/api/stats");
  const saved = s.source_bytes - s.stored_bytes;
  $("#stats").textContent =
    `${s.photos.toLocaleString()} unique photos · ${s.sources.toLocaleString()} files scanned · ` +
    `${fmtBytes(s.stored_bytes)} stored` + (saved > 0 ? ` (${fmtBytes(saved)} saved by dedup)` : "");
  $("#dupe-badge").hidden = !s.similar_groups;
  $("#dupe-badge").textContent = s.similar_groups;
  $("#review-badge").hidden = !s.open_questions;
  $("#review-badge").textContent = s.open_questions;
  $("#hide-non-best").checked = s.dup_policy === "hide_non_best";
  return s;
}

// ------------------------------------------------------------------ library

const collapsed = new Set(JSON.parse(localStorage.getItem("ps-collapsed") || "[]"));

async function loadFactors() {
  const factors = await api(`/api/factors?show_hidden=${state.showHidden}`);
  const box = $("#factors");
  box.replaceChildren(
    el("div", { class: `f-val ${state.valueId === null ? "active" : ""}`, onclick: () => pick(null, "All photos") },
      el("span", {}, "All photos")),
  );
  for (const f of factors) {
    if (!f.values.length) continue;
    const head = el("div", { class: `f-name ${collapsed.has(f.key) ? "collapsed" : ""}` }, f.name);
    const list = el("div", { hidden: collapsed.has(f.key) });
    head.addEventListener("click", () => {
      collapsed.has(f.key) ? collapsed.delete(f.key) : collapsed.add(f.key);
      try { localStorage.setItem("ps-collapsed", JSON.stringify([...collapsed])); } catch {}
      head.classList.toggle("collapsed"); list.hidden = !list.hidden;
    });
    const roots = f.values.filter((v) => !v.parent_id);
    const kids = (id) => f.values.filter((v) => v.parent_id === id);
    const row = (v, child) => el("div", {
      class: `f-val ${child ? "child" : ""} ${state.valueId === v.id ? "active" : ""}`,
      onclick: () => pick(v.id, `${f.name} › ${v.label}`),
    }, el("span", {}, v.label), el("span", { class: "n" }, v.count.toLocaleString()));
    for (const v of roots) {
      list.append(row(v, false));
      for (const c of kids(v.id)) list.append(row(c, true));
    }
    box.append(head, list);
  }
}

function pick(valueId, title) {
  state.valueId = valueId; state.title = title; state.query = ""; $("#search-q").value = "";
  loadFactors(); loadGrid(true);
}

async function loadGrid(reset) {
  if (reset) { state.offset = 0; $("#grid").replaceChildren(); }
  const q = new URLSearchParams({ offset: state.offset, limit: 200, show_hidden: state.showHidden });
  if (state.valueId !== null) q.set("value_id", state.valueId);
  let res;
  if (state.query) {
    try { res = await api(`/api/search?${new URLSearchParams({ q: state.query, offset: state.offset, limit: 200 })}`); }
    catch (e) { $("#grid").replaceChildren(el("div", { class: "empty" }, e.message)); return; }
  } else res = await api(`/api/photos?${q}`);
  const { total, items } = res;
  $("#grid-title").textContent = state.title;
  $("#grid-count").textContent = `${total.toLocaleString()} photos`;
  if (!total) $("#grid").replaceChildren(el("div", { class: "empty" }, "No photos yet — go to Import to add a folder."));
  for (const p of items) {
    $("#grid").append(el("div", {
      class: `tile ${p.hidden ? "hidden-dupe" : ""} ${state.selected === p.sha256 ? "selected" : ""}`,
      "data-sha": p.sha256, title: fmtDate(p.taken_at), onclick: () => openDetail(p.sha256),
    },
      el("img", { src: thumb(p.sha256), loading: "lazy", alt: "" }),
      p.group_size > 1 ? el("span", { class: "tag" }, `⧉ ${p.group_size}`) : null,
      p.n_sources > 1 ? el("span", { class: "tag left", title: "identical files merged" }, `×${p.n_sources}`) : null,
    ));
  }
  state.offset += items.length;
  $("#load-more").hidden = state.offset >= total;
}
$("#load-more").addEventListener("click", () => loadGrid(false));
$("#search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  state.query = $("#search-q").value.trim();
  state.valueId = null;
  state.title = state.query ? `“${state.query}”` : "All photos";
  loadFactors(); loadGrid(true);
});
$("#show-hidden").addEventListener("change", (e) => { state.showHidden = e.target.checked; loadFactors(); loadGrid(true); });

async function openDetail(sha) {
  state.selected = sha;
  document.querySelectorAll(".tile.selected").forEach((t) => t.classList.remove("selected"));
  document.querySelector(`.tile[data-sha="${sha}"]`)?.classList.add("selected");
  const d = await api(`/api/photos/${sha}`);
  const p = d.photo;
  const kv = (k, v) => (v || v === 0 ? [el("dt", {}, k), el("dd", {}, v)] : []);
  $("#detail").hidden = false;
  $("#detail").replaceChildren(
    el("button", { class: "close", onclick: () => { $("#detail").hidden = true; state.selected = null; } }, "×"),
    el("div", { class: "photo-wrap" },
      el("img", { src: thumb(sha), alt: "", onclick: () => openLightbox(sha) }),
      d.faces.map((f) => el("div", { class: "face-box", title: f.label || "",
        style: `left:${f.x * 100}%;top:${f.y * 100}%;width:${f.w * 100}%;height:${f.h * 100}%` }))),
    d.faces.length ? el("h4", {}, "People") : null,
    d.faces.length ? el("div", { class: "faces-row" }, d.faces.map((f) => el("div", { class: "fc" },
      el("img", { class: "face", src: faceImg(f.id), alt: "" }),
      el("div", {}, f.label || "?"),
      el("button", { class: "link", title: "This face is not this person", onclick: async () => {
        await api(`/api/faces/${f.id}/reject`, {}); openDetail(sha); loadFactors(); loadStats(); } }, "Not them")))) : null,
    el("h4", {}, "Sorted under"),
    el("div", { class: "chips" }, d.factors.map((f) =>
      el("span", { class: "chip", title: f.factor, onclick: () => pick(f.value_id, `${f.factor} › ${f.label}`) }, f.label))),
    el("h4", {}, "Details"),
    el("dl", { class: "kv" },
      kv("Taken", `${fmtDate(p.taken_at)}${p.taken_src === "mtime" ? " (file date)" : ""}`),
      kv("Size", `${p.width}×${p.height} · ${fmtBytes(p.bytes)}`),
      kv("Camera", [p.camera_make, p.camera_model].filter(Boolean).join(" ")),
      kv("Location", p.lat != null ? `${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}` : ""),
      kv("Sharpness", p.quality),
      kv("Stored at", p.store_path),
      kv("SHA-256", p.sha256.slice(0, 16) + "…"),
    ),
    d.group ? el("p", { class: "small" }, `In a ${d.group.kind} group of near-duplicates. `,
      el("button", { class: "link", onclick: () => show("dupes") }, "Review")) : null,
    el("h4", {}, `Original files (${d.sources.length})`),
    ...d.sources.map((s) => el("div", { class: "src" }, s.original_path)),  // replaceChildren needs nodes, not an array
    el("h4", {}, "Add to a custom group"),
    el("form", { class: "row", onsubmit: async (e) => {
      e.preventDefault();
      const [factor, value] = [e.target.factor.value.trim(), e.target.value.value.trim()];
      if (!factor || !value) return;
      await api("/api/custom", { factor, value, shas: [sha] });
      openDetail(sha); loadFactors();
    } },
      el("input", { name: "factor", placeholder: "Factor" }), el("input", { name: "value", placeholder: "Group" }),
      el("button", {}, "Add")),
  );
}

function openLightbox(sha) {
  const img = $("#lightbox img");
  img.onerror = () => { img.onerror = null; img.src = thumb(sha); };  // e.g. HEIC in non-Safari browsers
  img.src = `/api/photos/${sha}/file`;
  $("#lightbox").hidden = false;
}
$("#lightbox").addEventListener("click", () => ($("#lightbox").hidden = true));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#lightbox").hidden = true; });

// ------------------------------------------------------------------ import

async function browse(path) {
  try {
    const r = await api(`/api/fs?path=${encodeURIComponent(path)}`);
    $("#fs-path").textContent = r.path;
    $("#fs-up").onclick = () => browse(r.parent);
    $("#fs-dirs").replaceChildren(...r.dirs.map((d) =>
      el("li", { onclick: () => browse(`${r.path}/${d}`) }, d)));
  } catch (e) { showImportError(e.message); }
}
$("#fs-use").addEventListener("click", () => ($("#import-path").value = $("#fs-path").textContent));

function showImportError(msg) {
  $("#import-error").hidden = !msg; $("#import-error").textContent = msg || "";
}

$("#import-start").addEventListener("click", async () => {
  showImportError("");
  const path = $("#import-path").value.trim();
  if (!path) return showImportError("Choose a folder first.");
  const mode = document.querySelector("input[name=mode]:checked").value;
  try { await api("/api/imports", { path, mode }); loadBatches(); }
  catch (e) { showImportError(e.message); }
});

let pollTimer = null;
async function loadBatches() {
  const batches = await api("/api/imports");
  $("#batches").replaceChildren(...(batches.length ? batches.map(batchCard) : [el("div", { class: "muted" }, "No imports yet.")]));
  clearTimeout(pollTimer);
  if (batches.some((b) => b.status === "running")) {
    pollTimer = setTimeout(() => { loadBatches(); loadStats(); }, 1000);
    $("#import-start").disabled = true;
  } else {
    $("#import-start").disabled = false;
    loadStats();
  }
}

function batchCard(b) {
  const pct = b.total ? Math.round((100 * b.processed) / b.total) : 0;
  const n = (label, v) => el("span", {}, `${label} `, el("b", {}, (v || 0).toLocaleString()));
  return el("div", { class: "batch" },
    el("div", { class: "top" },
      el("code", {}, b.root),
      el("span", { class: `status ${b.status}` },
        b.status === "running" ? (b.stage === "importing" || !b.stage ? `${pct}%` : `${b.stage}…`) : b.status)),
    b.status === "running" ? el("div", { class: "bar" }, el("div", { style: `width:${pct}%` })) : null,
    el("div", { class: "nums" },
      n("files", b.total), n("new photos", b.new_photos), n("exact duplicates", b.exact_dupes),
      n("already imported", b.skipped), n("near-dup groups", b.similar_groups), n("faces", b.faces_found),
      n("errors", b.errors),
      el("span", {}, `${b.mode} · ${fmtDate(b.started_at)}`)),
    b.message ? el("div", { class: "error" }, b.message) : null,
  );
}

// ------------------------------------------------------------------ duplicates

let groupOffset = 0;
async function loadGroups(reset) {
  if (reset) { groupOffset = 0; $("#groups").replaceChildren(); }
  const { total, groups } = await api(`/api/similar?offset=${groupOffset}&limit=30`);
  $("#dupe-count").textContent = `${total} groups`;
  if (!total) $("#groups").replaceChildren(el("div", { class: "empty" }, "No near-duplicates found."));
  for (const g of groups) $("#groups").append(groupCard(g));
  groupOffset += groups.length;
  $("#groups-more").hidden = groupOffset >= total;
}
$("#groups-more").addEventListener("click", () => loadGroups(false));

function groupCard(g) {
  const card = el("div", { class: "group" });
  const render = (g) => card.replaceChildren(
    el("div", { class: "head" },
      el("span", { class: `kind ${g.kind}` }, g.kind),
      el("span", { class: "muted" }, `${g.members.length} photos`),
      g.user_picked ? el("span", { class: "muted" }, "· chosen by you") : null),
    el("div", { class: "members" }, g.members.map((m) => {
      const best = m.sha256 === g.best_sha256;
      return el("div", { class: `member ${best ? "best" : ""}` },
        el("div", { class: "tile", title: "Keep this one", onclick: () => act(`/api/similar/${g.id}/best`, m.sha256) },
          el("img", { src: thumb(m.sha256), loading: "lazy", alt: "" }),
          best ? el("span", { class: "star" }, "★") : null),
        el("div", {}, `${m.width}×${m.height} · ${fmtBytes(m.bytes)}`),
        el("div", {}, `sharpness ${Math.round(m.quality)} · ${fmtDate(m.taken_at)}`),
        el("div", { class: "actions" },
          el("button", { class: "link", onclick: () => openLightbox(m.sha256) }, "View"),
          el("button", { class: "link", onclick: () => act(`/api/similar/${g.id}/remove`, m.sha256) }, "Not a duplicate")),
      );
    })),
  );
  const act = async (url, sha) => {
    await api(url, { sha256: sha });
    const fresh = await api(`/api/similar/${g.id}`).catch(() => null);
    fresh && fresh.members.length > 1 ? render(fresh) : card.remove();
    loadStats();
  };
  render(g);
  return card;
}

$("#hide-non-best").addEventListener("change", async (e) => {
  await api("/api/settings", { dup_policy: e.target.checked ? "hide_non_best" : "keep_all" });
  loadStats();
});

// ------------------------------------------------------------------ export

async function loadExport() {
  const [factors, s] = await Promise.all([api("/api/factors"), loadStats()]);
  $("#export-factor").replaceChildren(...factors.map((f) => el("option", { value: f.key }, f.name)));
  if (!$("#export-dest").value) $("#export-dest").value = s.library.replace(/\/library$/, "") + "/views";
}
$("#export-go").addEventListener("click", async () => {
  $("#export-result").textContent = "Working…";
  try {
    const r = await api("/api/export", {
      factor: $("#export-factor").value, dest: $("#export-dest").value.trim(),
      include_hidden: $("#export-hidden").checked,
    });
    $("#export-result").textContent =
      `Created ${r.hardlinks} hardlinks${r.symlinks ? ` and ${r.symlinks} symlinks` : ""} under ${r.path}`;
  } catch (e) { $("#export-result").textContent = e.message; }
});

// ------------------------------------------------------------------ people

let activePerson = null;
async function loadPeople() {
  const [people, s] = await Promise.all([
    api(`/api/people?include_singletons=${$("#people-singletons").checked}`), api("/api/stats")]);
  $("#faces-off").hidden = s.face_detection;
  $("#people-count").textContent = `${people.length} people`;
  $("#people-grid").replaceChildren(...(people.length ? people.map((p) =>
    el("div", { class: `person ${activePerson === p.id ? "active" : ""}`, onclick: () => openPerson(p, people) },
      el("img", { src: p.cover_face ? faceImg(p.cover_face) : "", alt: "" }),
      el("div", { class: `nm ${p.named ? "" : "unnamed"}` }, p.name),
      el("div", { class: "small muted" }, `${p.photos} photos`))) :
    [el("div", { class: "empty" }, "No people found yet.")]));
}
$("#people-singletons").addEventListener("change", loadPeople);

async function openPerson(p, all) {
  activePerson = p.id;
  const faces = await api(`/api/people/${p.id}/faces?limit=60`);
  const others = all.filter((o) => o.id !== p.id);
  const box = $("#person-detail");
  box.hidden = false;
  box.replaceChildren(
    el("form", { class: "row", onsubmit: async (e) => {
      e.preventDefault();
      await api(`/api/people/${p.id}/name`, { name: e.target.n.value });
      loadPeople(); loadStats(); box.hidden = true;
    } },
      el("input", { name: "n", value: p.named ? p.name : "", placeholder: `Who is ${p.name}?` }),
      el("button", { class: "primary" }, "Save name"),
      el("button", { type: "button", onclick: () => { pick(p.id, `People › ${p.name}`); show("browse"); } }, "View photos")),
    el("div", { class: "row small" }, "Same person as ",
      el("select", { id: "merge-into" }, el("option", { value: "" }, "choose…"),
        others.map((o) => el("option", { value: o.id }, `${o.name} (${o.photos})`))),
      el("button", { type: "button", onclick: async () => {
        const keep = Number($("#merge-into").value); if (!keep) return;
        await api("/api/people/merge", { keep, drop: p.id }); box.hidden = true; loadPeople(); loadStats();
      } }, "Merge")),
    el("h4", {}, `${faces.length} faces — click one that is not ${p.name} to remove it`),
    el("div", { class: "faces-row" }, faces.map((f) => el("div", { class: "fc" },
      el("img", { class: "face", src: faceImg(f.id), alt: "", title: `match ${Math.round(f.confidence * 100)}%`,
        onclick: async (e) => { await api(`/api/faces/${f.id}/reject`, {}); e.target.closest(".fc").remove(); } }),
      el("div", { class: "muted" }, f.source === "user" ? "confirmed" : `${Math.round(f.confidence * 100)}%`)))),
  );
  document.querySelectorAll(".person").forEach((n) => n.classList.remove("active"));
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ------------------------------------------------------------------ review queue

async function loadQuestions() {
  const qlist = await api("/api/questions");
  $("#questions").replaceChildren(...(qlist.length ? qlist.map(questionCard) :
    [el("div", { class: "empty" }, "Nothing to review right now.")]));
}

function questionCard(q) {
  const p = q.payload;
  const send = async (answer) => {
    try { await api(`/api/questions/${q.id}/answer`, { answer }); } catch (e) { alert(e.message); return; }
    loadQuestions(); loadStats();
  };
  const nameForm = (placeholder) => el("form", { class: "row", onsubmit: (e) => {
    e.preventDefault(); const name = e.target.n.value.trim(); if (name) send({ name });
  } }, el("input", { name: "n", placeholder }), el("button", { class: "primary" }, "Save"),
    el("button", { type: "button", onclick: () => send({ skip: true }) }, "Skip"));
  const yesNo = (key) => el("div", { class: "row" },
    el("button", { class: "primary", onclick: () => send({ [key]: true }) }, "Yes"),
    el("button", { onclick: () => send({ [key]: false }) }, "No"),
    el("button", { onclick: () => send({ skip: true }) }, "Skip"));
  const faces = (ids) => el("div", { class: "imgs" }, ids.filter(Boolean).map((id) => el("img", { class: "face", src: faceImg(id) })));
  const photos = (shas) => el("div", { class: "imgs" }, shas.map((s) => el("img", { src: thumb(s), onclick: () => openLightbox(s) })));
  let body;
  if (q.kind === "name_person") body = [faces(p.faces), nameForm("Name")];
  else if (q.kind === "merge_people") body = [faces([p.face_a, p.face_b]), yesNo("same")];
  else if (q.kind === "confirm_face") body = [photos([p.sha256]), yesNo("correct")];
  else if (q.kind === "name_event") body = [photos(p.sample), nameForm("e.g. Goa trip, Riya's birthday")];
  else if (q.kind === "confirm_group") body = groupPicker(p, send);
  else body = [];
  return el("div", { class: "q" }, el("div", { class: "qt" }, q.text), body);
}

function groupPicker(p, send) {
  // Click a photo to cycle: unanswered → belongs (✓) → does not belong (✗).
  const marks = new Map();
  const tiles = p.shas.map((s) => {
    const t = el("div", { class: "pick", title: "Click: belongs / doesn't belong" },
      el("img", { src: thumb(s) }), el("span", { class: "mark" }));
    t.addEventListener("click", () => {
      const next = { undefined: "yes", yes: "no", no: undefined }[marks.get(s)];
      next ? marks.set(s, next) : marks.delete(s);
      t.className = `pick ${next || ""}`; t.querySelector(".mark").textContent = next === "yes" ? "✓" : next === "no" ? "✗" : "";
    });
    return t;
  });
  const submit = (fill) => {
    const accept = [], reject = [];
    for (const s of p.shas) {
      const m = marks.get(s) || fill;
      if (m === "yes") accept.push(s); else if (m === "no") reject.push(s);
    }
    send({ accept, reject });
  };
  return [
    el("div", { class: "small muted" }, `${p.in_doubt} photos are uncertain. Click photos to mark ✓ or ✗; the group learns from every answer.`),
    el("div", { class: "imgs" }, tiles),
    el("div", { class: "row" },
      el("button", { class: "primary", onclick: () => submit(undefined) }, "Save"),
      el("button", { onclick: () => submit("yes") }, "Rest belong"),
      el("button", { onclick: () => submit("no") }, "Rest don't belong"),
      el("button", { onclick: () => send({ skip: true }) }, "Skip")),
  ];
}

// ------------------------------------------------------------------ settings

let llmPresets = {};
async function loadSettings() {
  const [l, s, custom, content] = await Promise.all([api("/api/llm"), api("/api/stats"), api("/api/custom"), api("/api/content")]);
  $("#content-state").textContent = (s.content_analysis ? "Content analysis: on." : "Content analysis: off — run `.venv/bin/python -m photosort models clip`, then Re-analyse.")
    + (s.place_names ? " Place names: on." : " Place names: off — run `.venv/bin/python -m photosort models places`.");
  $("#cg-list").replaceChildren(...content.filter((c) => c.prompts).map((c) => el("div", { class: "custom-item" },
    el("span", {}, el("b", {}, c.name), ` · ${c.photos} photos · strictness ${c.threshold} · ${c.examples} answers `,
      el("code", {}, c.prompts.join(" | "))),
    el("span", {},
      el("button", { class: "link", onclick: () => {
        $("#cg-name").value = c.name; $("#cg-prompts").value = c.prompts.join("\n"); $("#cg-threshold").value = c.threshold;
      } }, "Edit"), " ",
      el("button", { class: "link", onclick: async () => {
        if (!confirm(`Delete content group "${c.name}"? Photos are not affected.`)) return;
        await api(`/api/custom/${c.id}`, undefined, "DELETE"); loadSettings();
      } }, "Delete")))));
  llmPresets = l.presets;
  $("#llm-preset").value = l.config.preset || "none";
  $("#llm-base").value = l.config.base_url || "";
  $("#llm-model").value = l.config.model || "";
  $("#llm-vision").checked = !!l.config.vision;
  $("#llm-effort").value = l.config.effort || "";
  $("#llm-key").value = "";
  $("#llm-key-state").textContent = l.key_set ? "(a key is saved)" : "(no key saved)";
  syncPresetUI();
  $("#set-gap").value = s.event_gap_hours;
  $("#set-min").value = s.event_min_photos;
  $("#face-state").textContent = s.face_detection ? "Face detection: on (local models)."
    : "Face detection: off — run `.venv/bin/python -m photosort models` then Re-analyse.";
  $("#cust-list").replaceChildren(...custom.map((c) => el("div", { class: "custom-item" },
    el("span", {}, el("b", {}, `${c.factor} › ${c.value}`), ` · ${c.photos} photos `,
      c.rule ? el("code", {}, JSON.stringify(c.rule)) : el("span", { class: "muted" }, "(manual)")),
    el("button", { class: "link", onclick: async () => {
      if (!confirm(`Delete group "${c.value}"? Photos are not affected.`)) return;
      await api(`/api/custom/${c.id}`, undefined, "DELETE"); loadSettings();
    } }, "Delete"))));
}
function syncPresetUI() {
  const cc = $("#llm-preset").value === "claude_code";
  $("#llm-cc-note").hidden = !cc;
  for (const id of ["#llm-base", "#llm-key"]) { $(id).disabled = cc; $(id).previousElementSibling.style.opacity = cc ? .5 : 1; }
}
$("#llm-preset").addEventListener("change", (e) => {
  const p = llmPresets[e.target.value];
  if (p) $("#llm-base").value = p.base_url;
  syncPresetUI();
});
async function saveLLM(extra = {}) {
  const body = { preset: $("#llm-preset").value, base_url: $("#llm-base").value, model: $("#llm-model").value,
    vision: $("#llm-vision").checked, effort: $("#llm-effort").value || null, ...extra };
  if ($("#llm-key").value && !("api_key" in extra)) body.api_key = $("#llm-key").value;
  await api("/api/llm", body);
  loadSettings();
}
$("#llm-save").addEventListener("click", async () => {
  try { await saveLLM(); $("#llm-result").textContent = "Saved."; } catch (e) { $("#llm-result").textContent = e.message; }
});
$("#llm-clear-key").addEventListener("click", async () => { await saveLLM({ api_key: "" }); $("#llm-result").textContent = "Key removed."; });
$("#llm-test").addEventListener("click", async () => {
  $("#llm-result").textContent = "Testing…";
  try { await saveLLM(); const r = await api("/api/llm/test", {}); $("#llm-result").textContent = `Connected. Model replied: ${r.reply}`; }
  catch (e) { $("#llm-result").textContent = e.message; }
});
$("#set-save").addEventListener("click", async () => {
  try {
    await api("/api/settings", { event_gap_hours: Number($("#set-gap").value), event_min_photos: Number($("#set-min").value) });
    $("#set-result").textContent = "Events rebuilt."; loadStats();
  } catch (e) { $("#set-result").textContent = e.message; }
});
$("#reanalyse").addEventListener("click", async () => {
  try { await api("/api/analyze", {}); $("#set-result").textContent = "Re-analysing… progress is on the Import page."; }
  catch (e) { $("#set-result").textContent = e.message; }
});
$("#cg-save").addEventListener("click", async () => {
  const prompts = $("#cg-prompts").value.split("\n").map((x) => x.trim()).filter(Boolean);
  try {
    const r = await api("/api/content", { name: $("#cg-name").value, prompts, threshold: Number($("#cg-threshold").value) });
    $("#cg-result").textContent = `Saved: ${r.photos} photos.`; loadSettings(); loadStats();
  } catch (e) { $("#cg-result").textContent = e.message; }
});
$("#cust-save").addEventListener("click", async () => {
  let rule;
  try { rule = JSON.parse($("#cust-rule").value); } catch { $("#cust-result").textContent = "Rule is not valid JSON."; return; }
  try {
    const r = await api("/api/custom", { factor: $("#cust-factor").value, value: $("#cust-value").value, rule });
    $("#cust-result").textContent = `Saved: ${r.photos} photos.`; loadSettings();
  } catch (e) { $("#cust-result").textContent = e.message; }
});

// ------------------------------------------------------------------ assistant

function chatMsg(role, text, extra = {}) {
  const m = el("div", { class: `msg ${role} ${extra.error ? "error" : ""}` }, text);
  if (extra.faces?.length) for (const p of extra.faces) {
    m.append(el("div", { class: "who" }, p.name), el("div", { class: "imgs" },
      p.faces.map((id) => el("img", { class: "face", src: faceImg(id) }))));
  }
  if (extra.photos?.length) m.append(el("div", { class: "imgs" },
    extra.photos.map((s) => el("img", { src: thumb(s), onclick: () => openLightbox(s) }))));
  $("#chat-log").append(m);
  $("#chat-log").scrollTop = $("#chat-log").scrollHeight;
  return m;
}

async function openChat() {
  $("#chat").hidden = false;
  if (!$("#chat-log").childElementCount) {
    const hist = await api("/api/chat");
    for (const m of hist) chatMsg(m.role, m.content);
    if (!hist.length) sendChat("Hi! Help me organise my photos.");
  }
  $("#chat-input").focus();
}

async function sendChat(text) {
  $("#chat-choices").replaceChildren();
  chatMsg("user", text);
  const typing = chatMsg("assistant typing", "Thinking…");
  $("#chat-form button").disabled = true;
  try {
    const r = await api("/api/chat", { message: text });
    typing.remove();
    chatMsg("assistant", r.reply || "(no reply)", { ...r, error: !!r.error && r.error !== "not_configured" });
    $("#chat-choices").replaceChildren(...(r.choices || []).map((c) =>
      el("span", { class: "chip", onclick: () => sendChat(c) }, c)));
    if (r.changed) { loadStats(); if (!$("#view-browse").hidden) { loadFactors(); loadGrid(true); }
      if (!$("#view-people").hidden) loadPeople(); if (!$("#view-review").hidden) loadQuestions(); }
  } catch (e) {
    typing.remove(); chatMsg("assistant", e.message, { error: true });
  } finally { $("#chat-form button").disabled = false; }
}

$("#chat-toggle").addEventListener("click", () => ($("#chat").hidden ? openChat() : ($("#chat").hidden = true)));
$("#chat-close").addEventListener("click", () => ($("#chat").hidden = true));
$("#chat-form").addEventListener("submit", (e) => {
  e.preventDefault(); const t = $("#chat-input").value.trim(); if (!t) return;
  $("#chat-input").value = ""; sendChat(t);
});
$("#chat-reset").addEventListener("click", async () => {
  await api("/api/chat", undefined, "DELETE"); $("#chat-log").replaceChildren(); $("#chat-choices").replaceChildren();
  sendChat("Hi! Help me organise my photos.");
});

// ------------------------------------------------------------------ boot

loadStats().then((s) => show(s.photos ? "browse" : "import"));
