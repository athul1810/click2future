"use strict";
const $ = id => document.getElementById(id);
let state, scene, image, selected, force = [0, 0], drag = false, jobId = null;
let playback = null, frame = 0, playing = false, previous = 0, strips = [], referenceStrip;
let operation = 0, busy = false;
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
  for (const id of ["generate", "scene", "model", "seed", "reverse", "zero"]) {
    $(id).disabled = value;
  }
  document.querySelectorAll("#objects button").forEach(button => {button.disabled = value;});
  $("input").setAttribute("aria-disabled", String(value));
  $("cancel").hidden = !value;
}
function changed() {
  if (!playback) return;
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
    playing = false;
    strips = [];
    $("results").replaceChildren(emptyState.cloneNode(true));
    $("play").disabled = true;
    $("play").textContent = "▶";
    $("timeline").disabled = true;
    $("timeline").value = 0;
    $("time").textContent = "0.00s";
    $("mode").textContent = "awaiting a push";
    $("details").textContent = "generate a future to see its record.";
    $("reference-details").hidden = true;
    message(scene.split + " · " + state.notice);
    drawInput();
  } finally {
    if (current === operation) setBusy(false);
  }
}
async function showResult(result, current) {
  const loaded = await Promise.all(result.models.map(model => loadImage(model.strip)));
  const reference = await loadImage(result.reference.strip);
  if (current !== operation) return;
  referenceStrip = reference;
  playback = result;
  strips = [];
  frame = 0;
  previous = 0;
  $("results").replaceChildren();
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
      grid.append(card);
      strips.push({canvas, image: loaded[index], row});
    });
    section.append(grid);
    $("results").append(section);
  });
  $("timeline").max = result.frames - 1;
  $("timeline").value = 0;
  $("timeline").disabled = false;
  $("play").disabled = false;
  $("reference-details").hidden = false;
  $("mode").textContent = result.models.every(model => model.source === "cached replay") ? "cached replay" : "live model output";
  $("details").textContent = JSON.stringify({scene: result.scene, split: result.split,
    selected_object: result.object, force: result.force, seeds: result.seeds,
    models: result.models.map(model => ({name: model.name, source: model.source, ...model.details})),
    note: result.notice}, null, 2);
  playing = true;
  $("play").textContent = "Ⅱ";
  drawFrame();
}
function drawFrame() {
  if (!playback) return;
  const [height, width] = playback.size;
  strips.forEach(({canvas, image, row}) => {
    const ctx = canvas.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(image, frame * width, row * height, width, height, 0, 0, canvas.width, canvas.height);
  });
  const canvas = $("reference"), ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(referenceStrip, frame * width, 0, width, height, 0, 0, canvas.width, canvas.height);
  $("timeline").value = frame;
  $("time").textContent = (frame / playback.fps).toFixed(2) + "s";
}
async function generate() {
  if (busy) return;
  const seed = Number($("seed").value);
  if (!Number.isInteger(seed) || seed < 0 || seed > 2147483644) {
    message("choose a whole-number seed between 0 and 2147483644", true);
    return;
  }
  const current = ++operation;
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
$("scene").onchange = () => chooseScene().catch(error => message(error.message, true));
$("model").onchange = changed;
$("seed").oninput = changed;
$("generate").onclick = generate;
$("cancel").onclick = cancel;
$("play").onclick = () => {playing = !playing; previous = 0; $("play").textContent = playing ? "Ⅱ" : "▶";};
$("timeline").oninput = () => {
  frame = Number($("timeline").value);
  playing = false;
  $("play").textContent = "▶";
  drawFrame();
};
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
    await chooseScene();
  } catch (error) {message("could not start the demo: " + error.message, true);}
}
init();
