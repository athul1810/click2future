"use strict";
const $ = id => document.getElementById(id);
let state, scene, image, selected, force = [0, 0], drag = false, jobId = null;
let playback = null, frame = 0, playing = false, previous = 0, strips = [], referenceStrip;
let operation = 0, busy = false;
let pinned = null, pinnedStrips = [], referenceOperation = 0;
const emptyState = $("results").firstElementChild.cloneNode(true);

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: {"Content-Type": "application/json", "X-Demo-Token": state.token},
    body: JSON.stringify(body)
  };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "request failed");
  return data;
}
function message(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}
function loadImage(source) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("could not load video frames"));
    img.src = source;
  });
}
function setBusy(value) {
  busy = value;
  for (const id of ["generate", "scene", "model", "seed", "reverse", "zero", "open-image", "save-image", "upload-image"]) {
    $(id).disabled = value;
  }
  document.querySelectorAll("#objects button").forEach(button => {button.disabled = value;});
  $("input").setAttribute("aria-disabled", String(value));
  $("cancel").hidden = !value;
  $("pin").disabled = value || !playback;
  $("reference-world").disabled = value || !playback?.reference;
}
function changed() {
  if (!playback) return;
  setPlaying(false);
  $("mode").textContent = "previous request · generate to update";
  message("the videos show the previous request. generate to apply these changes.");
}
function drawInput() {
  if (!image) return;
  const canvas = $("input"), ctx = canvas.getContext("2d"), scale = canvas.width / scene.size[1];
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  const obj = scene.objects.find(item => item.id === selected);
  if (!obj) return;
  const [x, y] = obj.center.map(value => value * scale);
  ctx.strokeStyle = "#faffec";
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.arc(x, y, Math.max(16, scale * 6), 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  const magnitude = Math.hypot(...force), length = canvas.width * .36 * magnitude / state.force_limit;
  if (magnitude > 0) {
    const angle = Math.atan2(force[1], force[0]);
    const endX = x + Math.cos(angle) * length, endY = y + Math.sin(angle) * length;
    ctx.strokeStyle = "#1b3113";
    ctx.fillStyle = "#1b3113";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(endX, endY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(endX, endY);
    ctx.lineTo(endX - 13 * Math.cos(angle - .45), endY - 13 * Math.sin(angle - .45));
    ctx.lineTo(endX - 13 * Math.cos(angle + .45), endY - 13 * Math.sin(angle + .45));
    ctx.closePath();
    ctx.fill();
  }
  $("strength").textContent = Math.round(100 * magnitude / state.force_limit) + "%";
  $("selection").textContent = "object " + (selected + 1) + " selected · drag from it to set the push";
  document.querySelectorAll("#objects button").forEach(button => {
    const active = Number(button.dataset.object) === selected;
    button.classList.toggle("selected", active);
    button.setAttribute("aria-pressed", String(active));
  });
}
function clampForce(x, y) {
  const magnitude = Math.hypot(x, y);
  const scale = magnitude > state.force_limit ? state.force_limit / magnitude : 1;
  force = [x * scale, y * scale];
  changed();
  drawInput();
}
function pointer(event) {
  const bounds = $("input").getBoundingClientRect();
  return [(event.clientX - bounds.left) * scene.size[1] / bounds.width,
    (event.clientY - bounds.top) * scene.size[0] / bounds.height];
}
function pushTo(event) {
  const [x, y] = pointer(event), obj = scene.objects.find(item => item.id === selected);
  clampForce((x - obj.center[0]) * state.force_limit / (scene.size[1] * .36),
    (y - obj.center[1]) * state.force_limit / (scene.size[0] * .36));
}
async function chooseScene() {
  const current = ++operation;
  scene = state.scenes.find(item => item.id === $("scene").value);
  selected = scene.default_object;
  setBusy(true);
  $("cancel").hidden = true;
  try {
    const loaded = await loadImage(scene.image);
    if (current !== operation) return;
    image = loaded;
    const magnitude = Math.hypot(...scene.default_force);
    force = magnitude > state.force_limit ? scene.default_force.map(value => value * state.force_limit / magnitude) : scene.default_force.slice();
    $("objects").replaceChildren();
    scene.objects.forEach(obj => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.object = obj.id;
      button.textContent = "object " + (obj.id + 1);
      button.onclick = () => {selected = obj.id; changed(); drawInput();};
      $("objects").append(button);
    });
    playback = null;
    referenceStrip = null;
    clearPin();
    ++referenceOperation;
    setPlaying(false);
    strips = [];
    $("results").replaceChildren(emptyState.cloneNode(true));
    $("play").disabled = true;
    $("timeline").disabled = true;
    $("timeline").value = 0;
    $("time").textContent = "0.00s";
    $("mode").textContent = "awaiting a push";
    $("details").textContent = "generate a future to see its record.";
    $("reference-details").hidden = true;
    $("reference-properties").textContent = "";
    message(scene.split + " · " + state.notice);
    drawInput();
  } finally {
    if (current === operation) setBusy(false);
  }
}
async function showResult(result, current) {
  const loaded = await Promise.all(result.models.map(model => loadImage(model.strip)));
  const reference = result.reference ? await loadImage(result.reference.strip) : null;
  if (current !== operation) return;
  referenceStrip = reference;
  playback = result;
  strips = [];
  frame = 0;
  previous = 0;
  $("results").replaceChildren();
  buildCards(result, loaded, $("results"), strips);
  $("reference-world").replaceChildren();
  scene.reference_worlds.forEach((world, index) => {
    const option = document.createElement("option");
    option.value = world;
    option.textContent = "world " + (index + 1);
    $("reference-world").append(option);
  });
  if (result.reference) {
    $("reference-world").value = result.reference.world;
    referenceProperties(result.reference);
  }
  $("pin").disabled = false;
  updatePairing();
  $("timeline").max = result.frames - 1;
  $("timeline").value = 0;
  $("timeline").disabled = false;
  $("play").disabled = false;
  setPlaying(!window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  $("reference-details").hidden = !result.reference;
  const cached = result.models.filter(model => model.source === "cached replay").length;
  $("mode").textContent = cached === result.models.length ? "cached replay" : cached ? "live + cached outputs" : "live model output";
  $("details").textContent = JSON.stringify({scene: result.scene, split: result.split, input_source: result.input_source,
    selected_object: result.object, force: result.force, seeds: result.seeds,
    models: result.models.map(model => ({name: model.name, source: model.source, ...model.details,
      pixel_tracker: model.pixel_tracks?.extractor_version,
      tracker_status: model.pixel_tracks?.status,
      tracker_adoption: model.pixel_tracks?.adoption,
      track_failures: model.pixel_tracks?.samples?.map(sample => sample.reasons)})),
    note: result.notice}, null, 2);
  drawFrame();
}
function buildCards(result, loaded, target, destination) {
  result.models.forEach((model, index) => {
    const section = document.createElement("section");
    section.className = "model-result";
    const heading = document.createElement("div");
    heading.className = "model-heading";
    const title = document.createElement("h3"), source = document.createElement("span");
    title.textContent = model.name;
    source.textContent = model.source + " · " + model.seconds.toFixed(2) + "s";
    heading.append(title, source);
    section.append(heading);
    const grid = document.createElement("div");
    grid.className = "future-grid";
    result.seeds.forEach((seed, row) => {
      const card = document.createElement("div"), canvas = document.createElement("canvas");
      card.className = "future-card";
      canvas.width = 256;
      canvas.height = 256;
      canvas.setAttribute("aria-label", model.name + " future " + (row + 1));
      const caption = document.createElement("p"), label = document.createElement("span"), value = document.createElement("span");
      label.textContent = "future " + (row + 1);
      value.textContent = "seed " + seed;
      caption.append(label, value);
      card.append(canvas, caption);
      const tracks = model.pixel_tracks;
      const count = document.createElement("p");
      count.className = "track-count";
      count.textContent = tracks?.status === "available"
        ? tracks.samples[row].future_tracked + "/" + tracks.samples[row].future_expected + " future object observations tracked"
        : "pixel tracking unavailable";
      card.append(count);
      const response = document.createElement("p");
      response.className = "motion-response";
      card.append(response);
      grid.append(card);
      destination.push({canvas, image: loaded[index], row, tracks, result, response});
    });
    section.append(grid);
    target.append(section);
  });
}
function drawFrame() {
  if (!playback) return;
  const [height, width] = playback.size;
  [...strips, ...pinnedStrips].forEach(({canvas, image, row, tracks, result, response}) => {
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(image, frame * width, row * height, width, height, 0, 0, canvas.width, canvas.height);
    if ($("tracks").checked && tracks?.status === "available") {
      drawTracks(ctx, canvas, tracks.samples[row], tracks.object_ids, result.object);
    }
    response.textContent = motionResponse(tracks?.samples[row], result.object, result.force, result.size, frame);
  });
  if (referenceStrip) {
    const canvas = $("reference"), ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(referenceStrip, frame * width, 0, width, height, 0, 0, canvas.width, canvas.height);
  }
  $("timeline").value = frame;
  $("time").textContent = (frame / playback.fps).toFixed(2) + "s";
}
function motionResponse(sample, object, push, size, at) {
  if (!sample?.tracked[0]?.[object] || !sample.tracked[at]?.[object]) return "selected motion: untracked";
  const start = sample.positions[0][object], end = sample.positions[at][object];
  const dx = (end[0] - start[0]) * size[1], dy = (end[1] - start[1]) * size[0];
  const magnitude = Math.hypot(...push);
  if (!magnitude) return "selected motion: " + Math.hypot(dx, dy).toFixed(2) + " px from start";
  const along = (dx * push[0] + dy * push[1]) / magnitude;
  return "selected motion: " + (along >= 0 ? "+" : "") + along.toFixed(2) + " px along push";
}
function drawTracks(ctx, canvas, sample, ids, selectedObject) {
  for (const object of ids) {
    ctx.strokeStyle = object === selectedObject ? "#ffffff" : "#0c719c";
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let connected = false;
    for (let index = 0; index <= frame; index++) {
      if (!sample.tracked[index][object]) {connected = false; continue;}
      const [x, y] = sample.positions[index][object];
      if (connected) ctx.lineTo(x * canvas.width, y * canvas.height);
      else ctx.moveTo(x * canvas.width, y * canvas.height);
      connected = true;
    }
    ctx.stroke();
    if (sample.tracked[frame][object]) {
      const [x, y] = sample.positions[frame][object];
      ctx.beginPath();
      ctx.arc(x * canvas.width, y * canvas.height, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
function clearPin() {
  pinned = null;
  pinnedStrips = [];
  $("pinned-results").replaceChildren();
  $("pinned").hidden = true;
  $("clear-pin").hidden = true;
  updatePairing();
}
function updatePairing() {
  const paired = pinned && playback && JSON.stringify(pinned.seeds) === JSON.stringify(playback.seeds);
  $("pairing").textContent = pinned
    ? paired ? "same four seeds; world-conditioned models reuse their sampled prior draws" : "different seeds; comparison is unpaired"
    : "four samples; no seed selection";
}
function pinAction() {
  if (!playback || busy) return;
  clearPin();
  pinned = playback;
  const images = playback.models.map((model, index) => strips[index * playback.seeds.length].image);
  buildCards(pinned, images, $("pinned-results"), pinnedStrips);
  $("pinned-description").textContent = "object " + (pinned.object + 1) + "; force [" + pinned.force.map(v => v.toFixed(3)).join(", ") + "]; seeds " + pinned.seeds.join(", ");
  $("pinned").hidden = false;
  $("clear-pin").hidden = false;
  updatePairing();
  drawFrame();
}
function referenceProperties(reference) {
  $("reference-properties").textContent = reference.physical_properties.map((p, i) =>
    "object " + (i + 1) + ": mass " + p.mass.toFixed(2) + ", friction " + p.friction.toFixed(2) + ", restitution " + p.restitution.toFixed(2)).join("\n");
}
async function changeReference() {
  if (!playback?.reference || busy) return;
  const current = ++referenceOperation, result = playback;
  $("reference-world").disabled = true;
  try {
    const reference = await api("/api/reference", {scene: result.scene, object: result.object,
      force: result.force, world: $("reference-world").value});
    const loaded = await loadImage(reference.strip);
    if (current !== referenceOperation || result !== playback) return;
    referenceStrip = loaded;
    referenceProperties(reference);
    $("reference-note").textContent = reference.notice;
    drawFrame();
  } catch (error) {
    if (current === referenceOperation) message(error.message, true);
  } finally {
    if (current === referenceOperation) $("reference-world").disabled = busy;
  }
}
async function generate() {
  if (busy) return;
  const seed = Number($("seed").value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 2147483644) {
    message("choose a whole-number seed between 0 and 2147483644", true);
    return;
  }
  const current = ++operation;
  ++referenceOperation;
  setBusy(true);
  drag = false;
  message("queuing your futures…");
  try {
    const names = $("model").value === "__all" ? state.models.map(model => model.id) : [$("model").value];
    const request = await api("/api/generate", {scene: scene.id, object: selected,
      force: force.slice(), seeds: [seed, seed + 1, seed + 2, seed + 3], models: names});
    if (current !== operation) {
      await api("/api/jobs/" + request.id + "/cancel", {});
      return;
    }
    jobId = request.id;
    while (current === operation) {
      const job = await api("/api/jobs/" + request.id);
      if (current !== operation) break;
      message(job.progress);
      if (job.status === "complete") {
        await showResult(job.result, current);
        if (current === operation) message("ready. change the push or object and keep the same seeds to compare.");
        break;
      }
      if (job.status === "error") throw new Error(job.progress);
      if (job.status === "cancelled") break;
      await new Promise(resolve => setTimeout(resolve, 300));
    }
  } catch (error) {
    if (current === operation) message(error.message, true);
  } finally {
    if (current === operation) {jobId = null; setBusy(false);}
  }
}
async function cancel() {
  ++operation;
  const current = jobId;
  jobId = null;
  // The request that is still being accepted cancels itself when its ID arrives.
  if (current) {
    try {await api("/api/jobs/" + current + "/cancel", {});}
    catch (error) {message(error.message, true); setBusy(false); return;}
  }
  setBusy(false);
  message("generation cancelled");
}
$("input").addEventListener("pointerdown", event => {
  if (busy || !scene) return;
  const [x, y] = pointer(event), hit = scene.hit_map[Math.floor(y)]?.[Math.floor(x)];
  if (hit === undefined || hit < 0) {message("click a coloured object first"); return;}
  selected = hit;
  drag = true;
  $("input").setPointerCapture(event.pointerId);
  changed();
  drawInput();
});
$("input").addEventListener("pointermove", event => {if (drag && !busy) pushTo(event);});
$("input").addEventListener("pointerup", () => {drag = false;});
$("input").addEventListener("pointercancel", () => {drag = false;});
$("input").addEventListener("keydown", event => {
  if (busy || !scene) return;
  const changes = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]};
  if (changes[event.key]) {
    event.preventDefault();
    const delta = state.force_limit * (event.shiftKey ? .01 : .05);
    clampForce(force[0] + changes[event.key][0] * delta, force[1] + changes[event.key][1] * delta);
  }
});
$("reverse").onclick = () => clampForce(-force[0], -force[1]);
$("zero").onclick = () => clampForce(0, 0);
$("open-image").onclick = () => $("upload-image").click();
$("save-image").onclick = () => {
  if (!scene || busy) return;
  const link = document.createElement("a");
  link.href = scene.image;
  link.download = "click2future-input.png";
  link.click();
};
$("upload-image").onchange = async () => {
  const file = $("upload-image").files[0];
  if (!file || busy) return;
  const current = ++operation;
  setBusy(true);
  $("cancel").hidden = true;
  try {
    if (file.size > 65536) throw new Error("use a 64 x 64 tabletop png, at most 64 KiB");
    const pixels = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("could not read the image"));
      reader.readAsDataURL(file);
    });
    const opened = await api("/api/images", {image: pixels});
    if (current !== operation) return;
    if (!state.scenes.some(item => item.id === opened.scene.id)) {
      state.scenes.push(opened.scene);
      const option = document.createElement("option");
      option.value = opened.scene.id;
      option.textContent = "uploaded image " + state.scenes.filter(item => item.id.startsWith("upload-")).length;
      $("scene").append(option);
    }
    $("scene").value = opened.scene.id;
    await chooseScene();
    message("image opened. click an object and drag a push; no simulator reference is available.");
  } catch (error) {
    message(error.message, true);
  } finally {
    $("upload-image").value = "";
    if (!jobId) setBusy(false);
  }
};
$("scene").onchange = () => chooseScene().catch(error => message(error.message, true));
$("model").onchange = changed;
$("seed").oninput = changed;
$("generate").onclick = generate;
$("cancel").onclick = cancel;
function setPlaying(value) {
  playing = value;
  previous = performance.now();
  $("play").textContent = playing ? "pause" : "play";
  $("play").setAttribute("aria-label", playing ? "pause futures" : "play futures");
}
$("play").onclick = () => setPlaying(!playing);
$("timeline").oninput = () => {
  frame = Number($("timeline").value);
  setPlaying(false);
  drawFrame();
};
$("tracks").onchange = drawFrame;
$("pin").onclick = pinAction;
$("clear-pin").onclick = clearPin;
$("reference-world").onchange = changeReference;
function tick(now) {
  if (playing && playback && now - previous >= 1000 / playback.fps) {
    frame = (frame + 1) % playback.frames;
    previous = now;
    drawFrame();
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
async function init() {
  try {
    state = await api("/api/state");
    state.scenes.forEach((item, index) => {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent = "scene " + String(index + 1).padStart(2, "0");
      $("scene").append(option);
    });
    if (state.models.length > 1) {
      const all = document.createElement("option");
      all.value = "__all";
      all.textContent = "all loaded models";
      $("model").append(all);
    }
    state.models.forEach(model => {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.id;
      $("model").append(option);
    });
    $("model").value = state.models[0].id;
    await chooseScene();
  } catch (error) {message("could not start the demo: " + error.message, true);}
}
init();
