// Sections in this file, in order (see CONTRIBUTING.md for the bigger picture):
//   1. Form / payload            - reading the form into a PipelineConfig-shaped JSON body
//   2. Pipeline stage progress   - the checklist mirroring orchestrate.py's stage_fn(...) calls
//   3. Post-run preview          - in-browser <video> playback with generated subtitle tracks
//   4. Recent jobs               - the server-side job list, for reconnecting from any tab
//   5. Confirm dialogs           - the 4 confirm_request renderers (changes/primer/transcript/primer_frames)
//   6. Job connection (SSE)      - EventSource wiring, reconnect-on-load, form submit
//   7. VAD visualization         - the settings-preview diagram + real-video analyze/click-to-play
//   8. Reference-frame picker    - pinning labeled character-identity frames

// ===== SECTION: Form / payload =====
const BOOL_FIELDS = [
  "no_translate", "no_gpu", "vad", "no_llm_check", "no_llm_vision",
  "no_context_primer", "no_transcript_review", "auto_confirm",
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
// Mirrors pipeline/orchestrate.py's AUDIO_EXTENSIONS - keep in sync by hand (same caveat
// as PIPELINE_STAGES below, see CONTRIBUTING.md).
const AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus", ".wma"];

const form = document.getElementById("runForm");
const runBtn = document.getElementById("runBtn");
const cancelBtn = document.getElementById("cancelBtn");
const resetBtn = document.getElementById("resetBtn");
const logEl = document.getElementById("log");
const confirmArea = document.getElementById("confirmArea");
const confirmTitle = document.getElementById("confirmTitle");
const videoPathInput = document.getElementById("videoPathInput");
const audioOnlyNote = document.getElementById("audioOnlyNote");
const noLlmVisionCheckbox = document.getElementById("no_llm_vision");
const refFrameAddControls = document.getElementById("refFrameAddControls");
const refFrameAudioNote = document.getElementById("refFrameAudioNote");
const browseModalEl = document.getElementById("browseModal");
const browseModal = new bootstrap.Modal(browseModalEl);
const browsePath = document.getElementById("browsePath");
const browseList = document.getElementById("browseList");
const stageListEl = document.getElementById("stageList");
const recentJobsList = document.getElementById("recentJobsList");
const refreshJobsBtn = document.getElementById("refreshJobsBtn");
const outputPreviewArea = document.getElementById("outputPreviewArea");
const outputPreviewDesc = document.getElementById("outputPreviewDesc");
const outputPreviewVideo = document.getElementById("outputPreviewVideo");
const outputPreviewSrcTrack = document.getElementById("outputPreviewSrcTrack");
const outputPreviewTgtTrack = document.getElementById("outputPreviewTgtTrack");

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
        item.textContent = (entry.is_dir ? "📁 " : entry.is_audio ? "🎵 " : "🎬 ") + entry.name;
        item.onclick = (e) => {
          e.preventDefault();
          if (entry.is_dir) {
            loadBrowse(entry.path);
          } else {
            videoPathInput.value = entry.path;
            updateAudioOnlyNote();
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
  clearRefFramePicker();
  renderVadViz();
  updateAudioOnlyNote();
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
    if (btn.dataset.resetTarget === "panelLlm") {
      clearRefFramePicker();
      updateAudioOnlyNote(); // re-force "Disable vision" if audio-only input is still active
    }
  });
});
const confirmBody = document.getElementById("confirmBody");
const confirmButtons = document.getElementById("confirmButtons");

let currentJobId = null;
let currentEventSource = null;
const JOB_STORAGE_KEY = "localsub_current_job_id";

function appendLog(line) {
  const div = document.createElement("div");
  if (/\[WARNING\]/.test(line)) div.className = "log-warning";
  else if (/\[ERROR\]/i.test(line)) div.className = "log-error";
  div.textContent = line;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
}

function isAudioPath(path) {
  const lower = path.trim().toLowerCase();
  return AUDIO_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

// Remembers whatever the user had "Disable vision" set to before audio-only input forced
// it on, so switching back to a video path restores their actual preference instead of
// silently leaving it checked.
let noLlmVisionUserValue = null;

// Reflects pipeline/orchestrate.py's is_audio_only-forces-no_llm_vision behavior (and the
// fact that audio input skips the final mux) directly in the form, rather than letting
// someone discover it only after a run - see webapp/runner.py's start_job for the
// matching server-side config_flags override that also keeps the stage checklist honest.
// The checkbox itself gets forced checked + disabled too - there's no real choice to make
// once there's no video to pull frames from, so let the control say that outright.
function updateAudioOnlyNote() {
  const isAudio = isAudioPath(videoPathInput.value);
  audioOnlyNote.style.display = isAudio ? "block" : "none";
  if (isAudio) {
    if (!noLlmVisionCheckbox.disabled) noLlmVisionUserValue = noLlmVisionCheckbox.checked;
    noLlmVisionCheckbox.checked = true;
    noLlmVisionCheckbox.disabled = true;
    noLlmVisionCheckbox.title = "Forced on for audio-only input - there's no video to pull frames from.";
  } else if (noLlmVisionCheckbox.disabled) {
    noLlmVisionCheckbox.checked = noLlmVisionUserValue === null ? false : noLlmVisionUserValue;
    noLlmVisionCheckbox.disabled = false;
    noLlmVisionCheckbox.title = "";
  }
  // Pinning reference frames has no effect once vision is off - grey out the "add a new
  // one" controls rather than leaving them fully interactive for something that gets
  // silently discarded server-side. Already-pinned frames (refFrameList, not part of this
  // wrapper) stay removable either way - clearing stale ones is still a reasonable action.
  refFrameAddControls.classList.toggle("section-disabled", isAudio);
  refFrameAudioNote.style.display = isAudio ? "block" : "none";
}

function buildPayload() {
  const data = new FormData(form);
  const payload = { video_path: data.get("video_path"), workdir: data.get("workdir") || null };
  for (const f of BOOL_FIELDS) payload[f] = form.querySelector(`[name="${f}"]`).checked;
  for (const f of INT_FIELDS) payload[f] = parseInt(data.get(f), 10);
  for (const f of FLOAT_FIELDS) payload[f] = parseFloat(data.get(f));
  for (const f of STR_FIELDS) payload[f] = data.get(f);
  payload.reference_frames = referenceFrames;
  return payload;
}

// ===== SECTION: Pipeline stage progress =====
// --- Pipeline stage progress indicator ---
// Mirrors the stage ids pipeline/orchestrate.py's run_pipeline calls stage_fn(...) with.
// skipIf reads the same config flags buildPayload()/job_status's config_flags produce, so
// the checklist can mark stages this particular run's config will never reach as "skipped"
// up front instead of leaving them looking merely "not started yet" forever.
const PIPELINE_STAGES = [
  { id: "audio", label: "Extract audio" },
  { id: "transcribe", label: "Transcribe" },
  { id: "repeats", label: "Repeat-loop resolution", skipIf: (c) => c.no_llm_check },
  { id: "primer_frames", label: "Primer-frame review", skipIf: (c) => c.no_llm_check || c.no_context_primer },
  { id: "primer", label: "Context primer", skipIf: (c) => c.no_llm_check || c.no_context_primer },
  { id: "rationality", label: "Rationality check", skipIf: (c) => c.no_llm_check },
  { id: "vision", label: "Vision follow-up", skipIf: (c) => c.no_llm_check || c.no_llm_vision },
  { id: "transcript_review", label: "Transcript review", skipIf: (c) => c.no_transcript_review },
  { id: "translate", label: "Translate", skipIf: (c) => c.no_translate },
  { id: "mux", label: "Mux output", skipIf: (c) => c.is_audio_only },
];
let stageStates = {};

function initStages(config) {
  stageStates = {};
  PIPELINE_STAGES.forEach((s) => {
    stageStates[s.id] = s.skipIf && s.skipIf(config) ? "skipped" : "pending";
  });
  renderStages();
}

function setStage(name) {
  let reachedActive = false;
  for (const s of PIPELINE_STAGES) {
    if (stageStates[s.id] === "skipped") continue;
    if (s.id === name) {
      stageStates[s.id] = "active";
      reachedActive = true;
    } else if (!reachedActive) {
      stageStates[s.id] = "done";
    }
  }
  renderStages();
}

function finishStages() {
  Object.keys(stageStates).forEach((id) => {
    if (stageStates[id] === "active") stageStates[id] = "done";
  });
  renderStages();
}

function errorStages() {
  Object.keys(stageStates).forEach((id) => {
    if (stageStates[id] === "active") stageStates[id] = "error";
  });
  renderStages();
}

// Distinct from errorStages() - cancellation isn't a failure, so the active stage gets its
// own neutral styling (see .stage-cancelled) rather than the red used for a real error.
function cancelStages() {
  Object.keys(stageStates).forEach((id) => {
    if (stageStates[id] === "active") stageStates[id] = "cancelled";
  });
  renderStages();
}

const STAGE_ICON = { pending: "○", active: "◐", done: "●", skipped: "—", error: "✕", cancelled: "⊘" };

function renderStages() {
  stageListEl.innerHTML = PIPELINE_STAGES.map((s) => {
    const state = stageStates[s.id] || "pending";
    return `<span class="stage-badge stage-${state}">${STAGE_ICON[state]} ${s.label}</span>`;
  }).join("");
}

// ===== SECTION: Post-run preview =====
// --- Post-run in-browser preview ---
function showOutputPreview(result) {
  if (!result || !result.video_path) return;
  outputPreviewVideo.src = `/api/video?path=${encodeURIComponent(result.video_path)}`;
  outputPreviewDesc.textContent = isAudioPath(result.video_path)
    ? "Plays the source audio with the generated subtitles as selectable tracks (use the "
      + "player's CC/subtitles button) - there's no picture, just the player controls and captions."
    : "Plays the source video with the generated subtitles as selectable tracks (use the "
      + "player's CC/subtitles button) - not every source format is browser-playable, the "
      + "same caveat as the reference-frame picker above.";
  if (result.src_srt) {
    outputPreviewSrcTrack.src = `/api/subtitle_vtt?path=${encodeURIComponent(result.src_srt)}`;
    outputPreviewSrcTrack.label = `Source (${result.lang})`;
    outputPreviewSrcTrack.default = !result.target_srt;
  } else {
    outputPreviewSrcTrack.removeAttribute("src");
  }
  if (result.target_srt) {
    outputPreviewTgtTrack.src = `/api/subtitle_vtt?path=${encodeURIComponent(result.target_srt)}`;
    outputPreviewTgtTrack.label = `Translated (${result.target_lang})`;
    outputPreviewTgtTrack.default = true;
  } else {
    outputPreviewTgtTrack.removeAttribute("src");
  }
  outputPreviewArea.style.display = "block";
}

function hideOutputPreview() {
  outputPreviewArea.style.display = "none";
  outputPreviewVideo.removeAttribute("src");
  outputPreviewSrcTrack.removeAttribute("src");
  outputPreviewTgtTrack.removeAttribute("src");
  outputPreviewVideo.load();
}

// ===== SECTION: Recent jobs =====
// --- Recent jobs (server-side job list - works for reconnecting from any tab/browser,
// not just the one localStorage remembers) ---
const STATUS_BADGE = {
  running: { cls: "bg-primary", text: "Running" },
  waiting_confirm: { cls: "bg-warning text-dark", text: "Waiting for input" },
  done: { cls: "bg-success", text: "Done" },
  error: { cls: "bg-danger", text: "Error" },
  cancelled: { cls: "bg-secondary", text: "Cancelled" },
};

// Shared by every place a job transitions into or out of "actively running" - keeps
// runBtn and cancelBtn in sync without duplicating the same two lines at each call site.
function setJobActive(active) {
  runBtn.disabled = active;
  cancelBtn.style.display = active ? "inline-block" : "none";
}

function switchToJob(jobId) {
  fetch(`/api/jobs/${jobId}`)
    .then((r) => r.json())
    .then((data) => {
      if (data.error) return;
      hideConfirm();
      hideOutputPreview();
      logEl.replaceChildren();
      appendLog(`[Connected to job ${jobId} - status: ${data.status}]`);
      connectToJob(jobId, data.config_flags);
      if (data.status === "done") showOutputPreview(data.result);
      setJobActive(data.status === "running" || data.status === "waiting_confirm");
    });
}

function loadRecentJobs() {
  fetch("/api/jobs")
    .then((r) => r.json())
    .then((jobs) => {
      recentJobsList.innerHTML = "";
      if (!jobs.length) {
        recentJobsList.innerHTML = '<div class="text-muted small">No jobs yet.</div>';
        return;
      }
      jobs.forEach((j) => {
        const a = document.createElement("a");
        a.href = "#";
        a.className = "list-group-item list-group-item-action d-flex justify-content-between align-items-center";
        const name = j.video_path ? j.video_path.split("/").pop() : j.job_id;
        const badge = STATUS_BADGE[j.status] || { cls: "bg-secondary", text: j.status };
        a.innerHTML = `<span>${escapeHtml(name)}</span><span class="badge ${badge.cls}">${badge.text}</span>`;
        a.onclick = (e) => {
          e.preventDefault();
          switchToJob(j.job_id);
        };
        recentJobsList.appendChild(a);
      });
    });
}

refreshJobsBtn.addEventListener("click", loadRecentJobs);

// ===== SECTION: Confirm dialogs =====
// Renders the 4 kinds of confirm_request event a job can emit (see
// pipeline/orchestrate.py's confirm_*_fn parameters and webapp/runner.py's
// make_web_confirm_* factories) and posts the browser's decision back via submitConfirm().
function hideConfirm() {
  confirmArea.style.display = "none";
  confirmBody.innerHTML = "";
  confirmButtons.innerHTML = "";
  document.title = "LocalSub";
}

// A job can pause on a confirm step for a while if you've tabbed away - scroll it into
// view, flash the title, and (if permission was granted) fire a real OS notification so
// it doesn't just sit there silently waiting on you.
function notifyActionNeeded(message) {
  confirmArea.style.display = "block";
  confirmArea.scrollIntoView({ behavior: "smooth", block: "center" });
  if (document.hidden) {
    document.title = "⚠ Action needed - LocalSub";
    if (typeof Notification !== "undefined" && Notification.permission === "granted") {
      new Notification("LocalSub needs your input", { body: message });
    }
  }
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) document.title = "LocalSub";
});

function submitConfirm(response) {
  fetch(`/api/jobs/${currentJobId}/confirm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(response),
  });
  hideConfirm();
}

// No optimistic UI update here - wait for the "cancelled" SSE event (handleJobEvent) to
// actually update the checklist/buttons, consistent with how confirm submission works.
// Cancellation is cooperative (see CONTRIBUTING.md / pipeline/errors.py): usually near-
// instant, but if an LLM call is actively in flight it takes effect once that call returns.
cancelBtn.addEventListener("click", () => {
  if (!currentJobId) return;
  fetch(`/api/jobs/${currentJobId}/cancel`, { method: "POST" });
});

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
  notifyActionNeeded(event.description);
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
  notifyActionNeeded("Review the generated context primer.");
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function showTranscriptConfirm(event) {
  confirmTitle.textContent = "Review transcript before translation";
  confirmBody.innerHTML = `
    <p class="text-muted small">Fix any transcription mistakes here before it's sent for translation.</p>
    <textarea class="form-control" rows="16" style="font-family: monospace" id="transcriptText">${escapeHtml(event.transcript)}</textarea>
  `;
  confirmButtons.innerHTML = `<button class="btn btn-primary btn-sm" id="continueTranscript">Continue</button>`;
  document.getElementById("continueTranscript").onclick = () => {
    submitConfirm({ text: document.getElementById("transcriptText").value });
  };
  notifyActionNeeded("Review the transcript before translation.");
}

function showPrimerFramesConfirm(event) {
  confirmTitle.textContent = "Review frames before the context primer";
  const working = event.frames.map((f) => ({ t: f.t, label: f.label }));
  let captured = null; // { t } once "Capture"/"Preview" has loaded a pending new frame

  function renderList(listEl) {
    listEl.innerHTML = "";
    if (working.length === 0) {
      const none = document.createElement("p");
      none.className = "text-muted small fst-italic";
      none.textContent = "No automatically-sampled frames left here - any reference frames you already pinned are still included regardless.";
      listEl.appendChild(none);
      return;
    }
    working.forEach((f, i) => {
      const row = document.createElement("div");
      row.className = "d-flex align-items-center gap-2 mb-2";

      const img = document.createElement("img");
      img.className = "rounded border";
      img.style.height = "70px";
      img.src = `/api/frame_preview?path=${encodeURIComponent(event.video_path)}&t=${f.t}`;

      const tInput = document.createElement("input");
      tInput.type = "number";
      tInput.step = "0.1";
      tInput.min = "0";
      tInput.className = "form-control form-control-sm";
      tInput.style.width = "6rem";
      tInput.value = f.t;
      tInput.onchange = () => {
        f.t = parseFloat(tInput.value) || 0;
        img.src = `/api/frame_preview?path=${encodeURIComponent(event.video_path)}&t=${f.t}`;
      };

      const labelInput = document.createElement("input");
      labelInput.type = "text";
      labelInput.className = "form-control form-control-sm";
      labelInput.placeholder = "Label (optional)";
      labelInput.value = f.label;
      labelInput.onchange = () => {
        f.label = labelInput.value;
      };

      const rm = document.createElement("button");
      rm.type = "button";
      rm.className = "btn btn-sm btn-outline-danger";
      rm.textContent = "Remove";
      rm.onclick = () => {
        working.splice(i, 1);
        renderList(listEl);
      };

      row.append(img, tInput, labelInput, rm);
      listEl.appendChild(row);
    });
  }

  confirmBody.innerHTML = `
    <p class="text-muted small">
      These are the frames automatically sampled for the context primer (any reference
      frames you already pinned are sent too, but aren't re-listed here - you already
      curated those). Adjust, remove, or add frames below before continuing. A labeled
      frame also gets reused as a reference image in every vision follow-up call later
      in the run.
    </p>
    <div id="primerFramesList"></div>
    <hr>
    <p class="text-muted small mb-1">Add another frame - scrub the video below, or enter a timestamp by hand:</p>
    <video id="primerFramesVideo" controls style="display:block; max-width:100%; max-height:200px; background:#000;"></video>
    <div class="d-flex align-items-end gap-2 mt-2 flex-wrap">
      <button type="button" class="btn btn-sm btn-outline-primary" id="primerFramesCaptureBtn">Capture frame at current position</button>
      <div>
        <label class="form-label small mb-0">or Timestamp (s)</label>
        <input type="number" step="0.1" min="0" class="form-control form-control-sm" id="primerFramesManualTime" style="width: 8rem">
      </div>
      <button type="button" class="btn btn-sm btn-outline-secondary" id="primerFramesManualPreviewBtn">Preview</button>
    </div>
    <div class="d-flex align-items-center gap-2 mt-2">
      <img id="primerFramesCaptureImg" style="display:none; max-height: 70px;" class="rounded border">
      <input type="text" class="form-control form-control-sm" id="primerFramesCaptureLabel" placeholder="Label (optional)" style="width: 12rem; display:none" >
      <button type="button" class="btn btn-sm btn-primary" id="primerFramesAddBtn" style="display:none">Add</button>
      <span id="primerFramesAddStatus" class="text-muted small"></span>
    </div>
  `;
  const listEl = document.getElementById("primerFramesList");
  renderList(listEl);

  const video = document.getElementById("primerFramesVideo");
  const captureBtn = document.getElementById("primerFramesCaptureBtn");
  const manualTime = document.getElementById("primerFramesManualTime");
  const manualPreviewBtn = document.getElementById("primerFramesManualPreviewBtn");
  const captureImg = document.getElementById("primerFramesCaptureImg");
  const captureLabel = document.getElementById("primerFramesCaptureLabel");
  const addBtn = document.getElementById("primerFramesAddBtn");
  const addStatus = document.getElementById("primerFramesAddStatus");

  video.onerror = () => {
    addStatus.textContent = "Couldn't play this video in the browser - use the timestamp field instead.";
  };
  video.src = `/api/video?path=${encodeURIComponent(event.video_path)}`;

  function previewCapture(t) {
    if (isNaN(t) || t < 0) {
      addStatus.textContent = "Enter a valid timestamp.";
      return;
    }
    captured = null;
    addBtn.style.display = "none";
    addStatus.textContent = "Loading preview...";
    captureImg.onload = () => {
      addStatus.textContent = "";
      captured = { t };
      captureLabel.style.display = "inline-block";
      addBtn.style.display = "inline-block";
    };
    captureImg.onerror = () => {
      addStatus.textContent = "Couldn't extract a frame at that timestamp.";
      captureImg.style.display = "none";
    };
    captureImg.style.display = "inline-block";
    captureImg.src = `/api/frame_preview?path=${encodeURIComponent(event.video_path)}&t=${t}`;
  }

  captureBtn.addEventListener("click", () => previewCapture(video.currentTime));
  manualPreviewBtn.addEventListener("click", () => previewCapture(parseFloat(manualTime.value)));

  addBtn.addEventListener("click", () => {
    if (!captured) return;
    working.push({ t: captured.t, label: captureLabel.value.trim() });
    renderList(listEl);
    captureLabel.value = "";
    captureImg.style.display = "none";
    captureLabel.style.display = "none";
    addBtn.style.display = "none";
    captured = null;
  });

  confirmButtons.innerHTML = `<button class="btn btn-primary btn-sm" id="continuePrimerFrames">Continue</button>`;
  document.getElementById("continuePrimerFrames").onclick = () => {
    submitConfirm({ frames: working.map((f) => ({ t: f.t, label: f.label })) });
  };
  notifyActionNeeded("Review the frames sampled for the context primer.");
}

// ===== SECTION: Job connection (SSE) =====
// Shared by a fresh submit, a page-load reconnect, and switching in from the recent-jobs
// list - all three just need to point the same event handling at a (possibly already
// in-progress) job id. configFlags drives the stage checklist; pass what job_status
// returned when reconnecting/switching, or the just-built payload for a fresh submit.
function connectToJob(jobId, configFlags) {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
  currentJobId = jobId;
  localStorage.setItem(JOB_STORAGE_KEY, jobId);
  if (configFlags) initStages(configFlags);
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  es.onmessage = handleJobEvent;
  currentEventSource = es;
}

function handleJobEvent(msg) {
  const event = JSON.parse(msg.data);
  if (event.type === "log") {
    appendLog(event.line);
  } else if (event.type === "stage") {
    setStage(event.name);
  } else if (event.type === "confirm_request") {
    if (event.kind === "changes") showChangesConfirm(event);
    else if (event.kind === "primer") showPrimerConfirm(event);
    else if (event.kind === "transcript") showTranscriptConfirm(event);
    else if (event.kind === "primer_frames") showPrimerFramesConfirm(event);
  } else if (event.type === "done") {
    finishStages();
    if (event.output_path) {
      appendLog(`\nDone: ${event.output_path}`);
    } else {
      appendLog(`\nDone (audio-only input - no muxed file, subtitle files below):`);
    }
    appendLog(`  Source subtitles (${event.lang}): ${event.src_srt}`);
    if (event.target_srt) appendLog(`  Target subtitles (${event.target_lang}): ${event.target_srt}`);
    showOutputPreview(event);
    setJobActive(false);
    if (currentEventSource) {
      currentEventSource.close();
      currentEventSource = null;
    }
  } else if (event.type === "error") {
    errorStages();
    appendLog(`\n[ERROR] ${event.message}`);
    setJobActive(false);
    if (currentEventSource) {
      currentEventSource.close();
      currentEventSource = null;
    }
  } else if (event.type === "cancelled") {
    cancelStages();
    appendLog("\nCancelled by user.");
    setJobActive(false);
    if (currentEventSource) {
      currentEventSource.close();
      currentEventSource = null;
    }
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  setJobActive(true);
  logEl.replaceChildren();
  hideConfirm();
  hideOutputPreview();
  if (typeof Notification !== "undefined" && Notification.permission === "default") {
    Notification.requestPermission();
  }
  const payload = buildPayload();
  // Optimistic first paint from the submitted form values, so the checklist isn't blank
  // while the request is in flight - connectToJob() below re-initializes it from the
  // server's response a moment later, which is the authoritative one (e.g. audio-only
  // input forces vision off server-side regardless of what the checkbox said - see
  // webapp/runner.py's start_job - so the client's own guess can be briefly wrong here).
  initStages(payload);

  fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then((r) => r.json())
    .then((data) => {
      connectToJob(data.job_id, data.config_flags);
      loadRecentJobs();
    });
});

// Reconnect to whatever job this browser last touched, so refreshing the page (or the tab
// crashing) mid-run doesn't strand you with no visibility into a job that's still going
// server-side - the SSE stream replays its full log history plus current stage/confirm/
// done/error state on (re)connect, see Job.subscribe() in webapp/runner.py.
window.addEventListener("DOMContentLoaded", () => {
  updateAudioOnlyNote(); // in case the browser restored a previously-typed value on load
  loadRecentJobs();
  const savedJobId = localStorage.getItem(JOB_STORAGE_KEY);
  if (!savedJobId) return;
  fetch(`/api/jobs/${savedJobId}`)
    .then((r) => r.json())
    .then((data) => {
      if (data.error) {
        localStorage.removeItem(JOB_STORAGE_KEY);
        return;
      }
      appendLog(`[Reconnected to job ${savedJobId} - status: ${data.status}]`);
      connectToJob(savedJobId, data.config_flags);
      if (data.status === "done") showOutputPreview(data.result);
      setJobActive(data.status === "running" || data.status === "waiting_confirm");
    });
});

// ===== SECTION: VAD visualization =====
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

  const rawLabel = usingReal ? "Detected speech (raw) - your file" : "Detected speech (raw) - synthetic example";
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
  updateAudioOnlyNote();
  if (realVadData && realVadData.videoPath !== videoPathInput.value.trim()) {
    realVadData = null;
    vadAnalyzeStatus.textContent = "File path changed - click Analyze again to refresh.";
    renderVadViz();
  }
  if (refFrameLoadedPath !== null && refFrameLoadedPath !== videoPathInput.value.trim()) {
    refFrameLoadedPath = null;
    refFrameVideo.style.display = "none";
    refFrameCaptureRow.style.display = "none";
    refFrameVideoStatus.textContent = "File path changed - click Load video again to refresh.";
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

// ===== SECTION: Reference-frame picker =====
// --- Reference frames: pin labeled character-identity frames (e.g. a clear intro shot),
// sent to both the context primer and every vision follow-up call (see
// PipelineConfig.reference_frames / pipeline.video_frames.reference_frame_content_blocks) so
// the LLM has an actual face to match against instead of only a prose guess at who's who.
let referenceFrames = [];
let refFrameLoadedPath = null;

const refFrameTime = document.getElementById("refFrameTime");
const refFrameLabel = document.getElementById("refFrameLabel");
const refFramePreviewBtn = document.getElementById("refFramePreviewBtn");
const refFrameAddBtn = document.getElementById("refFrameAddBtn");
const refFrameStatus = document.getElementById("refFrameStatus");
const refFramePreviewImg = document.getElementById("refFramePreviewImg");
const refFrameList = document.getElementById("refFrameList");
const refFrameLoadVideoBtn = document.getElementById("refFrameLoadVideoBtn");
const refFrameVideoStatus = document.getElementById("refFrameVideoStatus");
const refFrameVideo = document.getElementById("refFrameVideo");
const refFrameCaptureRow = document.getElementById("refFrameCaptureRow");
const refFrameCaptureBtn = document.getElementById("refFrameCaptureBtn");

function clearRefFramePicker() {
  referenceFrames = [];
  renderRefFrameList();
  refFrameLoadedPath = null;
  refFrameVideo.removeAttribute("src");
  refFrameVideo.style.display = "none";
  refFrameCaptureRow.style.display = "none";
  refFrameVideoStatus.textContent = "";
  refFramePreviewImg.style.display = "none";
  refFrameStatus.textContent = "";
  refFrameAddBtn.disabled = true;
}

function renderRefFrameList() {
  refFrameList.innerHTML = "";
  referenceFrames.forEach((rf, i) => {
    const badge = document.createElement("span");
    badge.className = "badge text-bg-secondary d-flex align-items-center gap-2";
    badge.style.fontSize = "0.85rem";
    badge.style.cursor = "pointer";
    badge.title = "Click to edit";
    const label = document.createElement("span");
    label.textContent = `${rf.t.toFixed(1)}s: ${rf.label}`;
    badge.appendChild(label);
    badge.addEventListener("click", () => editRefFrame(i));
    const rm = document.createElement("button");
    rm.type = "button";
    rm.className = "btn-close btn-close-white";
    rm.style.fontSize = "0.55rem";
    rm.setAttribute("aria-label", "Remove");
    rm.onclick = (e) => {
      e.stopPropagation();
      referenceFrames.splice(i, 1);
      renderRefFrameList();
    };
    badge.appendChild(rm);
    refFrameList.appendChild(badge);
  });
}

// Editing a pinned reference frame just pulls it back out of the list and back into the
// picker fields (pre-loading its preview) - reuses the exact same preview/Add flow as
// adding a new one, rather than a separate edit-in-place state machine.
function editRefFrame(i) {
  const rf = referenceFrames[i];
  referenceFrames.splice(i, 1);
  renderRefFrameList();
  refFrameLabel.value = rf.label;
  previewRefFrameAt(rf.t);
}

function previewRefFrameAt(t) {
  const videoPath = videoPathInput.value.trim();
  refFrameAddBtn.disabled = true;
  if (!videoPath) {
    refFrameStatus.textContent = "Enter a video path above first.";
    return;
  }
  if (isNaN(t) || t < 0) {
    refFrameStatus.textContent = "Enter a valid timestamp.";
    return;
  }
  refFrameTime.value = t.toFixed(2);
  refFrameStatus.textContent = "Loading preview...";
  const url = `/api/frame_preview?path=${encodeURIComponent(videoPath)}&t=${t}`;
  refFramePreviewImg.onload = () => {
    refFrameStatus.textContent = "";
    refFrameAddBtn.disabled = false;
  };
  refFramePreviewImg.onerror = () => {
    refFrameStatus.textContent = "Couldn't extract a frame at that timestamp.";
    refFramePreviewImg.style.display = "none";
  };
  refFramePreviewImg.style.display = "inline-block";
  refFramePreviewImg.src = url;
}

refFramePreviewBtn.addEventListener("click", () => {
  previewRefFrameAt(parseFloat(refFrameTime.value));
});

// Real <video> element with native seeking, so you can scrub to the moment you want instead
// of guessing a timestamp - "Capture" just reads currentTime and reuses the same
// server-side (ffmpeg) preview/extraction path as manual entry, so what gets pinned is
// exactly what the pipeline will later extract, not a browser-decoded approximation.
refFrameLoadVideoBtn.addEventListener("click", () => {
  const videoPath = videoPathInput.value.trim();
  if (!videoPath) {
    refFrameVideoStatus.textContent = "Enter a video path above first.";
    return;
  }
  refFrameVideoStatus.textContent = "Loading...";
  refFrameVideo.onloadedmetadata = () => {
    refFrameVideoStatus.textContent = "";
    refFrameVideo.style.display = "block";
    refFrameCaptureRow.style.display = "block";
    refFrameLoadedPath = videoPath;
  };
  refFrameVideo.onerror = () => {
    refFrameVideoStatus.textContent = "Couldn't play this video in the browser - use the manual timestamp field below instead.";
    refFrameVideo.style.display = "none";
    refFrameCaptureRow.style.display = "none";
    refFrameLoadedPath = null;
  };
  refFrameVideo.src = `/api/video?path=${encodeURIComponent(videoPath)}`;
});

refFrameCaptureBtn.addEventListener("click", () => {
  previewRefFrameAt(refFrameVideo.currentTime);
});

refFrameAddBtn.addEventListener("click", () => {
  const t = parseFloat(refFrameTime.value);
  const label = refFrameLabel.value.trim();
  if (isNaN(t) || t < 0 || !label) {
    refFrameStatus.textContent = "Need both a previewed timestamp and a label.";
    return;
  }
  referenceFrames.push({ t, label });
  renderRefFrameList();
  refFrameLabel.value = "";
  refFramePreviewImg.style.display = "none";
  refFrameAddBtn.disabled = true;
  refFrameStatus.textContent = "";
});
