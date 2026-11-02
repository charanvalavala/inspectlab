const $ = (selector) => document.querySelector(selector);
const canvas = $("#image");
const ctx = canvas.getContext("2d");
let samples = [],
  original = null,
  current = null,
  result = null;
let overlay = true,
  busy = false,
  generation = 0,
  uploaded = false,
  imageReady = false;
let enginePromise = null;
const banks = new Map();
const label = (value) => value.replaceAll("_", " ");
const status = (value) => {
  $("#engine-state").textContent = value;
};

async function readJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}`);
  return response.json();
}

async function loadImage(url) {
  const image = new Image();
  image.src = url;
  await image.decode();
  return image;
}

function invalidate() {
  generation += 1;
  imageReady = false;
  $("#inspect").disabled = true;
  result = null;
  $("#download").disabled = true;
  $("#score").textContent = "—";
  $("#threshold").textContent = "—";
  $("#decision").textContent = "Ready to inspect.";
  $("#decision").style.color = "";
  $("#finding").textContent =
    "Run inspection to compare this image with the selected normal reference bank.";
  $("#sample-truth").textContent = "";
  $("#timing").textContent = "";
  $("#score-bar").style.width = "0";
  $("#reference").getContext("2d").clearRect(0, 0, 224, 224);
  $("#error").textContent = "";
}

function makeCurrent() {
  if (!original) return;
  const source = document.createElement("canvas");
  source.width = source.height = 224;
  const context = source.getContext("2d");
  context.filter =
    $("#condition").value === "blur"
      ? "blur(2px)"
      : $("#condition").value === "dark"
        ? "brightness(65%)"
        : "none";
  context.drawImage(original, 0, 0, 224, 224);
  current = source;
}

async function applyCondition() {
  invalidate();
  const token = generation;
  makeCurrent();
  if (current && $("#condition").value === "jpeg") {
    const compressed = await loadImage(current.toDataURL("image/jpeg", 0.25));
    if (token !== generation) return;
    current.getContext("2d").drawImage(compressed, 0, 0);
  }
  imageReady = Boolean(current);
  $("#inspect").disabled = busy || !imageReady;
  render();
}

function render() {
  ctx.clearRect(0, 0, 448, 448);
  if (!current) return;
  $("#empty-overlay").hidden = true;
  ctx.drawImage(current, 0, 0, 448, 448);
  if (result && overlay) {
    const map = document.createElement("canvas");
    map.width = map.height = 16;
    const context = map.getContext("2d");
    const pixels = context.createImageData(16, 16);
    const opacity = Number($("#opacity").value);
    result.patch_distances.forEach((distance, index) => {
      const heat = Math.max(0, Math.min(1, distance / result.map_scale));
      const offset = index * 4;
      pixels.data[offset] = Math.round(50 + 205 * heat);
      pixels.data[offset + 1] = Math.round(75 + 130 * heat * heat);
      pixels.data[offset + 2] = Math.round(150 - 100 * heat);
      pixels.data[offset + 3] = Math.round(255 * opacity * heat);
    });
    context.putImageData(pixels, 0, 0);
    ctx.save();
    ctx.filter = "blur(5px)";
    ctx.drawImage(map, 0, 0, 448, 448);
    ctx.restore();
    const patch = result.query_patch;
    ctx.strokeStyle = "#fff3bd";
    ctx.lineWidth = 2;
    ctx.strokeRect(
      (patch % 16) * 28 + 1,
      Math.floor(patch / 16) * 28 + 1,
      26,
      26,
    );
  }
}

async function engine() {
  if (!enginePromise) {
    enginePromise = (async () => {
      if (!window.ort)
        throw new Error(
          "Inference runtime did not load. Check your connection and retry.",
        );
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths =
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
      status("Downloading encoder · 25 MB");
      const response = await fetch("artifacts/encoder.int8.onnx");
      if (!response.ok) throw new Error("Encoder download failed");
      const reader = response.body.getReader();
      const chunks = [];
      let loaded = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        status(`Downloading encoder · ${(loaded / 1e6).toFixed(1)} MB`);
      }
      const bytes = new Uint8Array(loaded);
      let offset = 0;
      chunks.forEach((chunk) => {
        bytes.set(chunk, offset);
        offset += chunk.length;
      });
      status("Preparing inference engine");
      const encoder = await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
      });
      const scorer = await ort.InferenceSession.create(
        "artifacts/scorer.onnx",
        { executionProviders: ["wasm"] },
      );
      return { encoder, scorer };
    })().catch((error) => {
      enginePromise = null;
      throw error;
    });
  }
  return enginePromise;
}

async function referenceBank(category) {
  if (!banks.has(category)) {
    const metadata = await readJSON(`data/${category}.json`);
    const response = await fetch(`data/${category}.f32`);
    if (!response.ok) throw new Error("Reference download failed");
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength !== metadata.shape[0] * metadata.shape[1] * 4)
      throw new Error("Reference bank has invalid dimensions");
    banks.set(category, {
      metadata,
      tensor: new ort.Tensor(
        "float32",
        new Float32Array(buffer),
        metadata.shape,
      ),
    });
  }
  return banks.get(category);
}

function tensorInput() {
  const pixels = current.getContext("2d").getImageData(0, 0, 224, 224).data;
  const plane = 224 * 224,
    array = new Float32Array(3 * plane);
  const mean = [0.485, 0.456, 0.406],
    deviation = [0.229, 0.224, 0.225];
  for (let index = 0; index < plane; index += 1) {
    for (let channel = 0; channel < 3; channel += 1)
      array[channel * plane + index] =
        (pixels[index * 4 + channel] / 255 - mean[channel]) /
        deviation[channel];
  }
  return new ort.Tensor("float32", array, [1, 3, 224, 224]);
}

async function inspect() {
  if (busy || !current || !imageReady) return;
  busy = true;
  const token = generation,
    category = $("#category").value;
  $("#inspect").disabled = true;
  $("#error").textContent = "";
  try {
    const { encoder, scorer } = await engine();
    status("Loading normal references");
    const { metadata, tensor } = await referenceBank(category);
    if (token !== generation) return;
    status("Comparing image patches");
    const started = performance.now();
    const encoded = await encoder.run({ pixels: tensorInput() });
    const matched = await scorer.run({
      patches: encoded.patches,
      bank: tensor,
    });
    if (token !== generation) return;
    const distances = [...matched.distances.data];
    const top = [...distances].sort((a, b) => b - a).slice(0, 3);
    const score = top.reduce((sum, value) => sum + value, 0) / top.length;
    const worst = distances.indexOf(Math.max(...distances));
    const nearest = Number(matched.nearest.data[worst]);
    result = {
      category,
      condition: $("#condition").value,
      source: uploaded ? "user upload" : $("#sample").value,
      score,
      threshold: metadata.threshold,
      flagged: score > metadata.threshold,
      score_is_probability: false,
      patch_distances: distances,
      query_patch: worst,
      nearest_reference: Math.floor(nearest / 256),
      reference_patch: nearest % 256,
      map_scale: metadata.map_scale,
      encoder_sha256: metadata.encoder_sha256,
      inference_ms: Math.round(performance.now() - started),
    };
    $("#decision").textContent = result.flagged
      ? "Worth a closer look."
      : "Within normal range.";
    $("#decision").style.color = result.flagged
      ? "var(--accent)"
      : "var(--blue)";
    $("#finding").textContent = result.flagged
      ? "Some patches differ from the normal reference bank. Review the highlighted area; the score alone does not identify a defect type."
      : "The score is below the held-out-normal threshold. That does not guarantee the object is defect-free.";
    $("#score").textContent = score.toFixed(3);
    $("#threshold").textContent = metadata.threshold.toFixed(3);
    const axis = Math.max(score * 1.2, metadata.threshold * 1.5, 0.001);
    $("#score-bar").style.width = `${Math.min(100, (score / axis) * 100)}%`;
    $("#threshold-marker").style.left = `${(metadata.threshold / axis) * 100}%`;
    $("#match-note").textContent =
      `Input patch ${worst + 1} → normal reference ${result.nearest_reference + 1}, patch ${result.reference_patch + 1}.`;
    const reference = await loadImage(
      `references/${category}-${result.nearest_reference}.jpg`,
    );
    if (token !== generation) return;
    const context = $("#reference").getContext("2d");
    context.drawImage(reference, 0, 0, 224, 224);
    context.strokeStyle = "#ffe5a0";
    context.lineWidth = 3;
    context.strokeRect(
      (result.reference_patch % 16) * 14,
      Math.floor(result.reference_patch / 16) * 14,
      14,
      14,
    );
    const sample = samples.find((item) => item.id === $("#sample").value);
    $("#sample-truth").textContent = uploaded
      ? "User upload · no ground-truth label available"
      : `Dataset label: ${label(sample.truth)}${result.condition === "clean" ? "" : " · modified image"}`;
    $("#timing").textContent =
      `${result.inference_ms.toLocaleString()} ms in this browser · model/reference download excluded`;
    $("#download").disabled = false;
    render();
    status("Browser inference complete");
  } catch (error) {
    $("#error").textContent =
      error.message || "Inspection failed. Reload and try again.";
    status("Inspection unavailable");
  } finally {
    busy = false;
    $("#inspect").disabled = !imageReady;
    if (token !== generation) status("Image changed · run inspection again");
  }
}

async function chooseSample() {
  invalidate();
  const token = generation;
  const image = await loadImage(`samples/${$("#sample").value}.png`);
  if (token !== generation) return;
  original = image;
  uploaded = false;
  $("#upload").value = "";
  await applyCondition();
}

async function chooseCategory() {
  const category = $("#category").value;
  $("#sample").replaceChildren(
    ...samples
      .filter((sample) => sample.category === category)
      .map((sample) => {
        const option = document.createElement("option");
        option.value = sample.id;
        option.textContent = `Example ${sample.id.split("-").at(-1)}`;
        return option;
      }),
  );
  await chooseSample();
}

$("#inspect").addEventListener("click", inspect);
$("#category").addEventListener("change", () =>
  chooseCategory().catch(showError),
);
$("#sample").addEventListener("change", () => chooseSample().catch(showError));
$("#condition").addEventListener("change", () =>
  applyCondition().catch(showError),
);
$("#opacity").addEventListener("input", render);
$("#view-toggle").addEventListener("click", () => {
  overlay = !overlay;
  $("#view-toggle").setAttribute("aria-pressed", String(overlay));
  $("#view-toggle").textContent = overlay ? "Heatmap on" : "Heatmap off";
  render();
});
function showError(error) {
  $("#error").textContent = error.message;
}
$("#upload").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  invalidate();
  if (
    file.size > 4_000_000 ||
    !["image/png", "image/jpeg", "image/webp"].includes(file.type)
  ) {
    showError(new Error("Use PNG, JPEG or WebP up to 4 MB."));
    return;
  }
  const url = URL.createObjectURL(file);
  try {
    const image = await loadImage(url);
    if (image.width * image.height > 16_000_000)
      throw new Error("Use an image up to 16 megapixels.");
    original = image;
    uploaded = true;
    await applyCondition();
  } catch (error) {
    showError(error);
  } finally {
    URL.revokeObjectURL(url);
  }
});
$("#download").addEventListener("click", () => {
  if (!result) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = "inspectlab-inspection.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

async function init() {
  samples = await readJSON("samples.json");
  await chooseCategory();
  const report = await readJSON("results.json");
  $("#test-count").textContent = Object.values(report.inventory).reduce(
    (sum, value) => sum + value.test_normal + value.test_anomalous,
    0,
  );
  for (const method of ["dino", "rgb"]) {
    const row = report.summary.find(
      (item) => item.method === method && item.shots === 16,
    );
    $(`#${method}-metric`).textContent = row.macro_image_auroc.toFixed(3);
  }
  $("#results").innerHTML = report.summary
    .map(
      (row) =>
        `<tr><td>${row.method === "dino" ? "DINOv2 · INT8" : "RGB patches"}</td><td>${row.shots}</td><td>${row.macro_image_auroc.toFixed(3)} ± ${row.std_across_seed_macro.toFixed(3)}</td><td>${row.macro_pixel_auroc_56.toFixed(3)} ± ${row.pixel_std.toFixed(3)}</td></tr>`,
    )
    .join("");
  $("#robustness").innerHTML = report.robustness
    .map(
      (row) =>
        `<tr><td>${label(row.category)}</td><td>${row.condition}</td><td>${row.image_auroc.toFixed(3)}</td><td>${(row.anomaly_recall_at_threshold * 100).toFixed(1)}%</td><td>${(row.normal_false_positive_rate * 100).toFixed(1)}%</td></tr>`,
    )
    .join("");
}
init().catch(showError);
