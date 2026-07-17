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
