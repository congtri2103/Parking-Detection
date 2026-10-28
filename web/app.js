const state = {
  config: null,
  source: "demo",
  points: [],
  jobId: null,
  pollTimer: null,
  duration: 20,
};

const $ = (selector) => document.querySelector(selector);
const sourceVideo = $("#sourceVideo");
const canvas = $("#roiCanvas");
const context = canvas.getContext("2d");
const roiText = $("#roiText");
const runButton = $("#runButton");

function parsePoints(text) {
  return text.split(";").map((part) => part.split(",").map(Number)).filter((point) => point.length === 2 && point.every(Number.isFinite));
}

function syncText() {
  roiText.value = state.points.map(([x, y]) => `${Math.round(x)},${Math.round(y)}`).join(";");
}

function resizeCanvas() {
  if (!sourceVideo.videoWidth) return;
  canvas.width = sourceVideo.videoWidth;
  canvas.height = sourceVideo.videoHeight;
  drawRoi();
}

function drawRoi() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.points.length) return;
  context.save();
  context.beginPath();
  state.points.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
  if (state.points.length > 2) context.closePath();
  context.fillStyle = "rgba(255, 174, 66, .16)";
  context.strokeStyle = "#ffbd5c";
  context.lineWidth = Math.max(3, canvas.width / 500);
  context.setLineDash([12, 7]);
  if (state.points.length > 2) context.fill();
  context.stroke();
  context.setLineDash([]);
  state.points.forEach(([x, y], index) => {
    context.beginPath();
    context.arc(x, y, Math.max(6, canvas.width / 190), 0, Math.PI * 2);
    context.fillStyle = "#07100f";
    context.fill();
    context.strokeStyle = "#ffbd5c";
    context.stroke();
    context.fillStyle = "#fff";
    context.font = `600 ${Math.max(11, canvas.width / 100)}px ui-monospace`;
    context.fillText(String(index + 1), x + 10, y - 10);
  });
  context.restore();
}

function setSource(source) {
  state.source = source;
  document.querySelectorAll(".source-button").forEach((button) => button.classList.toggle("active", button.dataset.source === source));
  $("#uploadBox").classList.toggle("hidden", source !== "upload");
  if (source === "demo" && state.config?.default_video) {
    sourceVideo.src = state.config.default_video;
    sourceVideo.load();
  } else if (source === "upload") {
    const file = $("#videoFile").files[0];
    sourceVideo.removeAttribute("src");
    if (file) sourceVideo.src = URL.createObjectURL(file);
    sourceVideo.load();
  }
}

async function loadConfig() {
  const response = await fetch("/api/config");
  state.config = await response.json();
  $("#deviceBadge").textContent = state.config.device.toUpperCase();
  state.points = parsePoints(state.config.default_roi);
  syncText();
  setSource("demo");
}

canvas.addEventListener("click", (event) => {
  if (!canvas.width || !canvas.height) return;
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * canvas.width / rect.width;
  const y = (event.clientY - rect.top) * canvas.height / rect.height;
  state.points.push([x, y]);
  syncText();
  drawRoi();
});

sourceVideo.addEventListener("loadedmetadata", () => {
  state.duration = sourceVideo.duration || 20;
  $("#videoStage").style.aspectRatio = `${sourceVideo.videoWidth} / ${sourceVideo.videoHeight}`;
  $("#videoMeta").textContent = `${sourceVideo.videoWidth}×${sourceVideo.videoHeight} · ${state.duration.toFixed(1)}s`;
  resizeCanvas();
});
window.addEventListener("resize", drawRoi);

document.querySelectorAll(".source-button").forEach((button) => button.addEventListener("click", () => setSource(button.dataset.source)));
$("#videoFile").addEventListener("change", (event) => {
  const file = event.target.files[0];
  $("#uploadName").textContent = file ? file.name : "Chọn video CCTV";
  setSource("upload");
});
$("#confidence").addEventListener("input", (event) => $("#confidenceValue").textContent = Number(event.target.value).toFixed(2));
roiText.addEventListener("change", () => { state.points = parsePoints(roiText.value); drawRoi(); });
$("#undoPoint").addEventListener("click", () => { state.points.pop(); syncText(); drawRoi(); });
$("#clearRoi").addEventListener("click", () => { state.points = []; syncText(); drawRoi(); });
$("#resetRoi").addEventListener("click", () => { state.points = parsePoints(state.config.default_roi); syncText(); drawRoi(); });

async function runAnalysis() {
  const error = $("#formError");
  error.textContent = "";
  if (state.points.length < 3) { error.textContent = "Hãy chọn ít nhất ba điểm cho vùng cấm đỗ."; return; }
  const file = $("#videoFile").files[0];
  if (state.source === "upload" && !file) { error.textContent = "Hãy chọn một video để tải lên."; return; }
  const form = new FormData();
  if (state.source === "upload") form.append("video", file);
  form.append("roi", roiText.value);
  form.append("confidence", $("#confidence").value);
  form.append("parking_seconds", $("#parkingSeconds").value);
  form.append("motion_threshold", $("#motionThreshold").value);
  runButton.disabled = true;
  runButton.querySelector("span").textContent = "Đang gửi yêu cầu…";
  try {
    const response = await fetch("/api/run", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Không thể chạy pipeline");
    state.jobId = payload.job_id;
    $("#progressPanel").classList.remove("hidden");
    $("#results").classList.add("hidden");
    state.pollTimer = setInterval(pollStatus, 1000);
    await pollStatus();
  } catch (exception) {
    error.textContent = exception.message;
    resetRunButton();
  }
}

function resetRunButton() {
  runButton.disabled = false;
  runButton.querySelector("span").textContent = "Chạy phân tích video";
}

async function pollStatus() {
  if (!state.jobId) return;
  try {
    const response = await fetch(`/api/status?id=${encodeURIComponent(state.jobId)}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error);
    $("#progressMessage").textContent = job.message;
    $("#progressTime").textContent = `${Number(job.processed_seconds || 0).toFixed(1)}s`;
    const progress = Math.min(98, Math.max(3, (Number(job.processed_seconds || 0) / Math.max(state.duration, 1)) * 100));
    $("#progressBar").style.width = `${job.status === "complete" ? 100 : progress}%`;
    $("#liveLog").textContent = (job.logs || []).slice(-18).join("\n");
    $("#liveLog").scrollTop = $("#liveLog").scrollHeight;
    if (job.status === "complete") {
      clearInterval(state.pollTimer);
      renderResults(job);
      resetRunButton();
    } else if (job.status === "failed") {
      clearInterval(state.pollTimer);
      $("#progressTitle").textContent = "Xử lý thất bại";
      $("#formError").textContent = job.message;
      resetRunButton();
    }
  } catch (exception) {
    clearInterval(state.pollTimer);
    $("#formError").textContent = exception.message;
    resetRunButton();
  }
}

function renderResults(job) {
  $("#progressTitle").textContent = "Hoàn tất phân tích";
  const summary = job.summary || {};
  $("#trackedMetric").textContent = summary.tracked_vehicles ?? 0;
  $("#violationMetric").textContent = summary.violating_vehicles ?? 0;
  $("#stoppedMetric").textContent = `${Number(summary.max_stopped_seconds || 0).toFixed(1)}s`;
  $("#observationMetric").textContent = summary.logged_observations ?? 0;
  const resultVideo = $("#resultVideo");
  resultVideo.src = `${job.result_video}?v=${Date.now()}`;
  resultVideo.load();
  $("#downloadVideo").href = job.result_video;
  $("#downloadCsv").href = job.csv_url;
  const gallery = $("#evidenceGallery");
  gallery.innerHTML = "";
  if (!job.evidence?.length) {
    gallery.innerHTML = '<div class="empty-evidence">Không có ảnh vi phạm.<br>Hãy kiểm tra ROI, confidence hoặc thời gian đỗ.</div>';
  } else {
    job.evidence.forEach((url, index) => {
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.target = "_blank";
      const image = document.createElement("img");
      image.src = url;
      image.alt = `Bằng chứng vi phạm ${index + 1}`;
      anchor.appendChild(image);
      gallery.appendChild(anchor);
    });
  }
  $("#results").classList.remove("hidden");
  $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

runButton.addEventListener("click", runAnalysis);
loadConfig().catch((error) => { $("#formError").textContent = `Không tải được cấu hình: ${error.message}`; });
