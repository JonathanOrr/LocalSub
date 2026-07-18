const BOOL_FIELDS = [
  "no_translate", "no_gpu", "vad", "no_llm_check", "no_llm_vision",
  "no_context_primer", "auto_confirm",
];
const INT_FIELDS = [
  "threads", "max_context", "flag_repeat_count", "vad_min_speech_ms",
  "vad_min_silence_ms", "vad_speech_pad_ms", "vad_segment_gap_ms", "context_primer_frames",
];
const FLOAT_FIELDS = [
  "vad_max_speech_s", "vad_threshold", "entropy_thold", "logprob_thold", "no_speech_thold",
];
const STR_FIELDS = [
  "lang", "target_lang", "engine", "model", "vad_engine", "llm_endpoint", "llm_model",
];

const form = document.getElementById("runForm");
const runBtn = document.getElementById("runBtn");
const resetBtn = document.getElementById("resetBtn");
const logEl = document.getElementById("log");
const confirmArea = document.getElementById("confirmArea");
const confirmTitle = document.getElementById("confirmTitle");
const videoPathInput = document.getElementById("videoPathInput");
const browseModalEl = document.getElementById("browseModal");
const browseModal = new bootstrap.Modal(browseModalEl);
const browsePath = document.getElementById("browsePath");
const browseList = document.getElementById("browseList");

function loadBrowse(path) {
  const url = path ? `/api/browse?path=${encodeURIComponent(path)}` : "/api/browse";
  fetch(url)
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        browsePath.textContent = `Error: ${data.error}`;
        browseList.innerHTML = "";
        return;
      }
      browsePath.textContent = data.path;
      browseList.innerHTML = "";
      if (data.parent) {
        const up = document.createElement("a");
        up.href = "#";
        up.className = "list-group-item list-group-item-action";
        up.textContent = ".. (up)";
        up.onclick = (e) => {
          e.preventDefault();
          loadBrowse(data.parent);
        };
        browseList.appendChild(up);
      }
      data.entries.forEach((entry) => {
        const item = document.createElement("a");
        item.href = "#";
        item.className = "list-group-item list-group-item-action";
        item.textContent = (entry.is_dir ? "📁 " : "🎬 ") + entry.name;
        item.onclick = (e) => {
          e.preventDefault();
          if (entry.is_dir) {
            loadBrowse(entry.path);
          } else {
            videoPathInput.value = entry.path;
            browseModal.hide();
          }
        };
        browseList.appendChild(item);
      });
    });
}

document.getElementById("browseBtn").addEventListener("click", () => {
  loadBrowse(videoPathInput.value || null);
  browseModal.show();
});

resetBtn.addEventListener("click", () => {
  form.reset();
  renderVadViz();
});

// Per-section "Reset section" buttons: form.reset() only resets the whole form, so reset
// just the fields inside the given container back to their defaultValue/defaultChecked/
// defaultSelected - the same server-rendered defaults form.reset() itself relies on.
function resetFieldsIn(container) {
  container.querySelectorAll("input, select").forEach((el) => {
    if (el.type === "checkbox" || el.type === "radio") {
      el.checked = el.defaultChecked;
    } else if (el.tagName === "SELECT") {
      Array.from(el.options).forEach((opt) => {
        opt.selected = opt.defaultSelected;
      });
    } else {
      el.value = el.defaultValue;
    }
  });
}

document.querySelectorAll(".section-reset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = document.getElementById(btn.dataset.resetTarget);
    resetFieldsIn(target);
    if (btn.dataset.resetTarget === "panelVad") renderVadViz();
  });
});
const confirmBody = document.getElementById("confirmBody");
const confirmButtons = document.getElementById("confirmButtons");

let currentJobId = null;

function appendLog(line) {
  logEl.textContent += line + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function buildPayload() {
  const data = new FormData(form);
  const payload = { video_path: data.get("video_path"), workdir: data.get("workdir") || null };
  for (const f of BOOL_FIELDS) payload[f] = form.querySelector(`[name="${f}"]`).checked;
  for (const f of INT_FIELDS) payload[f] = parseInt(data.get(f), 10);
  for (const f of FLOAT_FIELDS) payload[f] = parseFloat(data.get(f));
  for (const f of STR_FIELDS) payload[f] = data.get(f);
  return payload;
}

function hideConfirm() {
  confirmArea.style.display = "none";
  confirmBody.innerHTML = "";
  confirmButtons.innerHTML = "";
}

function submitConfirm(response) {
  fetch(`/api/jobs/${currentJobId}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(response),
  });
  hideConfirm();
}

function showChangesConfirm(event) {
  confirmTitle.textContent = event.description;
  confirmBody.innerHTML = "";
  event.changes.forEach((c, i) => {
    const div = document.createElement("div");
    div.className = "form-check";
    div.innerHTML = `
      <input class="form-check-input" type="checkbox" checked id="chg${i}">
      <label class="form-check-label" for="chg${i}">cues ${c.first_cue}-${c.last_cue}: ${c.summary}</label>
    `;
    confirmBody.appendChild(div);
  });
  confirmButtons.innerHTML = `
    <button class="btn btn-primary btn-sm" id="applySelected">Apply selected</button>
    <button class="btn btn-outline-secondary btn-sm" id="rejectAll">Reject all</button>
  `;
  document.getElementById("applySelected").onclick = () => {
    const selected = event.changes
      .map((_, i) => i)
      .filter((i) => document.getElementById(`chg${i}`).checked);
    submitConfirm({ selected });
  };
  document.getElementById("rejectAll").onclick = () => submitConfirm({ selected: [] });
  confirmArea.style.display = "block";
}

function showPrimerConfirm(event) {
  confirmTitle.textContent = "Context primer";
  confirmBody.innerHTML = `<textarea class="form-control" rows="6" id="primerText">${event.primer}</textarea>`;
  confirmButtons.innerHTML = `
    <button class="btn btn-primary btn-sm" id="usePrimer">Use this primer</button>
    <button class="btn btn-outline-secondary btn-sm" id="skipPrimer">Skip</button>
  `;
  document.getElementById("usePrimer").onclick = () => {
    submitConfirm({ action: "use", text: document.getElementById("primerText").value });
  };
  document.getElementById("skipPrimer").onclick = () => submitConfirm({ action: "skip" });
  confirmArea.style.display = "block";
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  runBtn.disabled = true;
  logEl.textContent = "";
  hideConfirm();

  fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildPayload()),
  })
    .then((r) => r.json())
    .then((data) => {
      currentJobId = data.job_id;
      const es = new EventSource(`/api/jobs/${currentJobId}/events`);
      es.onmessage = (msg) => {
        const event = JSON.parse(msg.data);
        if (event.type === "log") {
          appendLog(event.line);
        } else if (event.type === "confirm_request") {
          if (event.kind === "changes") showChangesConfirm(event);
          else if (event.kind === "primer") showPrimerConfirm(event);
        } else if (event.type === "done") {
          appendLog(`\nDone: ${event.output_path}`);
          appendLog(`  Source subtitles (${event.lang}): ${event.src_srt}`);
          if (event.target_srt) appendLog(`  Target subtitles (${event.target_lang}): ${event.target_srt}`);
          runBtn.disabled = false;
          es.close();
        } else if (event.type === "error") {
          appendLog(`\n[ERROR] ${event.message}`);
          runBtn.disabled = false;
          es.close();
        }
      };
    });
});

// --- VAD settings visualization ---
// A fixed, synthetic "representative dialogue" timeline (seconds) - not real audio, just
// something to run the actual merge/pad/split math against so the diagram reacts to
// whatever the user currently has the VAD fields set to. Mirrors ten_vad_speech_segments()
// in pipeline/vad_ten.py: merge across short silences, drop too-short blips, pad, re-merge
// any overlaps padding creates, then force-split anything too long.
const VAD_RAW_RUNS = [
  [0.5, 0.65], // a short interjection - likely dropped by min-speech-duration
  [1.5, 2.3], // sentence A
  [2.42, 3.1], // sentence B, only 120ms after A - candidate for min-silence merging
  [4.5, 12.5], // one long unbroken monologue - candidate for force-splitting
  [13.5, 13.9], // a trailing short blip
];
const VAD_TIMELINE_END = 15.0;

// Set once "Analyze this video" succeeds: { videoPath, rawRuns, durationS }. While null, the
// diagram runs against the synthetic timeline above instead.
let realVadData = null;

function computeVadSegments(rawRuns, minSpeechMs, minSilenceMs, padMs, maxSpeechS, timelineEnd) {
  let merged = [rawRuns[0].slice()];
  for (let i = 1; i < rawRuns.length; i++) {
    const [start, end] = rawRuns[i];
    const gapMs = (start - merged[merged.length - 1][1]) * 1000;
    if (gapMs <= minSilenceMs) {
      merged[merged.length - 1][1] = end;
    } else {
      merged.push([start, end]);
    }
  }
  const kept = merged.filter(([s, e]) => (e - s) * 1000 >= minSpeechMs);
  const droppedCount = merged.length - kept.length;
  if (kept.length === 0) return { segments: [], droppedCount };

  const padS = padMs / 1000;
  const padded = kept.map(([s, e]) => [Math.max(0, s - padS), Math.min(timelineEnd, e + padS)]);
  let segments = [padded[0].slice()];
  for (let i = 1; i < padded.length; i++) {
    const [s, e] = padded[i];
    if (s <= segments[segments.length - 1][1]) {
      segments[segments.length - 1][1] = Math.max(segments[segments.length - 1][1], e);
    } else {
      segments.push([s, e]);
    }
  }

  let final = [];
  for (const [s, e] of segments) {
    const dur = e - s;
    if (maxSpeechS <= 0 || dur <= maxSpeechS) {
      final.push([s, e]);
      continue;
    }
    const nChunks = Math.ceil(dur / maxSpeechS);
    const chunkLen = dur / nChunks;
    for (let k = 0; k < nChunks; k++) {
      final.push([s + k * chunkLen, Math.min(e, s + (k + 1) * chunkLen)]);
    }
  }
  return { segments: final, droppedCount };
}

function svgTimelineRow(y, label, x0, xw, boxes, color) {
  // label sits a full text-height clear of the box row below it (a font-size-11 glyph's
  // ascender reaches ~8-9 units above its own baseline, so y-8 put the two touching)
  let s = `<text x="0" y="${y - 20}" font-size="11" fill="#666">${label}</text>`;
  s += `<line x1="${x0}" y1="${y}" x2="${x0 + xw}" y2="${y}" stroke="#dee2e6" stroke-width="1"/>`;
  boxes.forEach(({ x, w, title, fill, playStart, playEnd }) => {
    const playable = playStart !== undefined && playEnd !== undefined;
    const cls = playable ? ' class="vad-playable"' : "";
    const dataAttrs = playable ? ` data-start="${playStart}" data-end="${playEnd}"` : "";
    s += `<rect x="${x}" y="${y - 10}" width="${Math.max(2, w)}" height="20" fill="${fill || color}" rx="3"${cls}${dataAttrs}><title>${title}</title></rect>`;
  });
  return s;
}

function renderVadViz() {
  const vizEl = document.getElementById("vadViz");
  const noteEl = document.getElementById("vadVizNote");
  if (!vizEl) return;

  const minSpeechMs = parseFloat(form.querySelector('[name="vad_min_speech_ms"]').value) || 0;
  const minSilenceMs = parseFloat(form.querySelector('[name="vad_min_silence_ms"]').value) || 0;
  const padMs = parseFloat(form.querySelector('[name="vad_speech_pad_ms"]').value) || 0;
  const maxSpeechS = parseFloat(form.querySelector('[name="vad_max_speech_s"]').value) || 0;
  const gapMs = parseFloat(form.querySelector('[name="vad_segment_gap_ms"]').value) || 0;
  const engine = form.querySelector('[name="vad_engine"]').value;

  const usingReal = realVadData !== null;
  const rawRuns = usingReal ? realVadData.rawRuns : VAD_RAW_RUNS;
  const timelineEnd = usingReal ? realVadData.durationS : VAD_TIMELINE_END;

  const { segments, droppedCount } = computeVadSegments(rawRuns, minSpeechMs, minSilenceMs, padMs, maxSpeechS, timelineEnd);

  // A fixed 800-unit-wide diagram squeezes a long video's many segments into slivers too
  // thin to see or click. Instead, give the timeline a minimum pixel budget per second and
  // let the container scroll horizontally (see #vadViz CSS) once that exceeds a normal
  // panel width - short videos still render as one glance-able strip, long ones scroll.
  const PX_PER_SEC = 15;
  const W = Math.max(800, timelineEnd * PX_PER_SEC), x0 = 4, xw = W - 8;
  const scaleX = (t) => x0 + (timelineEnd > 0 ? (t / timelineEnd) * xw : 0);

  const rawBoxes = rawRuns.map(([s, e]) => ({
    x: scaleX(s), w: scaleX(e) - scaleX(s), title: `${s.toFixed(2)}s-${e.toFixed(2)}s`,
    ...(usingReal ? { playStart: s, playEnd: e } : {}),
  }));
  const finalBoxes = segments.map(([s, e]) => ({
    x: scaleX(s), w: scaleX(e) - scaleX(s), title: `${s.toFixed(2)}s-${e.toFixed(2)}s (${(e - s).toFixed(2)}s)`,
    ...(usingReal ? { playStart: s, playEnd: e } : {}),
  }));

  const rawLabel = usingReal ? "Detected speech (raw) - your video" : "Detected speech (raw) - synthetic example";
  const rows = [
    { label: rawLabel, boxes: rawBoxes, color: "#0d6efd" },
    { label: "Kept & padded segments (sent to whisper)", boxes: finalBoxes, color: "#198754" },
  ];
  if (engine === "ten") {
    const gapS = gapMs / 1000;
    const totalDur = segments.reduce((sum, [s, e]) => sum + (e - s), 0) + gapS * Math.max(0, segments.length - 1);
    const scale = totalDur > 0 ? xw / totalDur : 1;
    let cursor = x0;
    const concatBoxes = [];
    segments.forEach(([s, e], i) => {
      const w = (e - s) * scale;
      concatBoxes.push({
        x: cursor, w, title: `${(e - s).toFixed(2)}s`, fill: "#198754",
        ...(usingReal ? { playStart: s, playEnd: e } : {}),
      });
      cursor += w;
      if (i < segments.length - 1) {
        const gw = gapS * scale;
        concatBoxes.push({ x: cursor, w: gw, title: `${gapMs}ms silence gap`, fill: "#adb5bd" });
        cursor += gw;
      }
    });
    rows.push({ label: "Trimmed audio actually sent to whisper (TEN VAD only)", boxes: concatBoxes, color: "#198754" });
  }

  // all rows in a single SVG (not several concatenated <svg> elements) with generous
  // per-row spacing, so labels/boxes never crowd into the row above or below
  const ROW_H = 55;
  const H = ROW_H * rows.length + 15;
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`;
  rows.forEach((row, i) => {
    const y = ROW_H * i + 40;
    svg += svgTimelineRow(y, row.label, x0, xw, row.boxes, row.color);
  });
  svg += `</svg>`;
  vizEl.innerHTML = svg;

  let note = usingReal
    ? `${rawRuns.length} real speech blips -> ${segments.length} final segment(s) sent to whisper (click a box to hear it)`
    : `${rawRuns.length} raw speech blips -> ${segments.length} final segment(s) sent to whisper`;
  if (droppedCount > 0) note += `, ${droppedCount} short blip(s) dropped entirely (below min speech duration)`;
  noteEl.textContent = note;
}

["vad_min_speech_ms", "vad_min_silence_ms", "vad_speech_pad_ms", "vad_max_speech_s", "vad_segment_gap_ms", "vad_engine"]
  .forEach((name) => {
    const el = form.querySelector(`[name="${name}"]`);
    el.addEventListener("input", renderVadViz);
    el.addEventListener("change", renderVadViz);
  });
renderVadViz();

// "Analyze this video" - fetches real raw VAD runs once (expensive: extracts audio + runs
// VAD inference), then renderVadViz()'s existing math takes it from there for every other
// slider. Only vad_threshold changes what counts as "speech" at the frame level, so it's
// the one field that also needs a fresh analyze - every other VAD field is cheap
// post-processing math already computed instantly client-side.
const vadAnalyzeBtn = document.getElementById("vadAnalyzeBtn");
const vadAnalyzeStatus = document.getElementById("vadAnalyzeStatus");

function runVadAnalyze() {
  const videoPath = videoPathInput.value.trim();
  if (!videoPath) {
    vadAnalyzeStatus.textContent = "Enter a video path above first.";
    return;
  }
  const threshold = parseFloat(form.querySelector('[name="vad_threshold"]').value) || 0.5;
  vadAnalyzeBtn.disabled = true;
  vadAnalyzeStatus.textContent = "Analyzing (extracting audio + running VAD)...";
  fetch("/api/vad_analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_path: videoPath, vad_threshold: threshold }),
  })
    .then((r) => r.json())
    .then((data) => {
      vadAnalyzeBtn.disabled = false;
      if (data.error) {
        vadAnalyzeStatus.textContent = `Error: ${data.error}`;
        return;
      }
      realVadData = { videoPath, rawRuns: data.raw_runs, durationS: data.duration_s };
      vadAnalyzeStatus.textContent = `Analyzed ${videoPath.split("/").pop()} (${data.duration_s.toFixed(1)}s)`;
      renderVadViz();
    })
    .catch((err) => {
      vadAnalyzeBtn.disabled = false;
      vadAnalyzeStatus.textContent = `Error: ${err}`;
    });
}
vadAnalyzeBtn.addEventListener("click", runVadAnalyze);

// changing vad_threshold invalidates the analyzed data (it affects what counts as "speech"
// in the first place), and changing the video path entirely obviously does too - both just
// fall back to the synthetic timeline until re-analyzed, rather than silently showing stale
// results for a different video/threshold.
form.querySelector('[name="vad_threshold"]').addEventListener("change", () => {
  if (realVadData) {
    realVadData = null;
    vadAnalyzeStatus.textContent = "Threshold changed - click Analyze again to refresh.";
    renderVadViz();
  }
});
videoPathInput.addEventListener("input", () => {
  if (realVadData && realVadData.videoPath !== videoPathInput.value.trim()) {
    realVadData = null;
    vadAnalyzeStatus.textContent = "Video path changed - click Analyze again to refresh.";
    renderVadViz();
  }
});

// click-to-play: boxes are only playable when using real data (a delegated listener since
// the SVG is regenerated via innerHTML on every render, so per-element listeners would be
// lost each time anyway)
document.getElementById("vadViz").addEventListener("click", (e) => {
  const rect = e.target.closest("rect.vad-playable");
  if (!rect || !realVadData) return;
  const start = rect.dataset.start, end = rect.dataset.end;
  const url = `/api/audio_clip?path=${encodeURIComponent(realVadData.videoPath)}&start=${start}&end=${end}`;
  new Audio(url).play().catch((err) => {
    vadAnalyzeStatus.textContent = `Playback error: ${err}`;
  });
});
