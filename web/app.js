// Digit Recognizer - browser version.
// Runs the SVM trained by train.py (exported by export_web_model.py) directly
// in the page: an RBF-kernel SVM with 45 one-vs-one classifiers that vote.

const M = DIGITS_MODEL;

// Where each class's support vectors start in the support_vectors list.
const SV_START = M.n_support.reduce((acc, n) => [...acc, acc[acc.length - 1] + n], [0]);

// Predict one 8x8 image (64 values, 0-16). Returns the digit and each class's votes.
function predict(pixels) {
  const kernel = M.support_vectors.map((sv) => {
    let d2 = 0;
    for (let k = 0; k < 64; k++) {
      const d = sv[k] - pixels[k];
      d2 += d * d;
    }
    return Math.exp(-M.gamma * d2); // RBF kernel
  });

  const n = M.classes.length;
  const votes = new Array(n).fill(0);
  let pair = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      // libsvm's one-vs-one decision for classes i and j
      let sum = M.intercept[pair++];
      for (let k = SV_START[i]; k < SV_START[i + 1]; k++) sum += M.dual_coef[j - 1][k] * kernel[k];
      for (let k = SV_START[j]; k < SV_START[j + 1]; k++) sum += M.dual_coef[i][k] * kernel[k];
      votes[sum > 0 ? i : j]++;
    }
  }
  let best = 0;
  for (let c = 1; c < n; c++) if (votes[c] > votes[best]) best = c;
  return { digit: M.classes[best], votes };
}

// Turn pen strokes into an 8x8 image the way the training data was made:
// the digit is scaled to fill a 32x32 box, then each 4x4 block's "on" pixels
// are counted, giving values 0-16.
function strokesToPixels(strokes, doc = document) {
  const points = strokes.flat();
  if (!points.length) return new Array(64).fill(0);
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const size = Math.max(maxX - minX, maxY - minY, 1);

  const pen = 4.6;              // stroke width in the 32x32 box
  const room = 32 - pen - 1;    // keep the whole stroke inside the box
  const scale = room / Math.max(size, 40);
  const offX = 16 - ((minX + maxX) / 2) * scale;
  const offY = 16 - ((minY + maxY) / 2) * scale;

  const canvas = doc.createElement("canvas");
  canvas.width = canvas.height = 32;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.lineWidth = pen;
  ctx.strokeStyle = ctx.fillStyle = "#000";
  for (const stroke of strokes) {
    ctx.beginPath();
    stroke.forEach(([x, y], i) => {
      const px = x * scale + offX, py = y * scale + offY;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    if (stroke.length === 1) {
      ctx.arc(stroke[0][0] * scale + offX, stroke[0][1] * scale + offY, pen / 2, 0, Math.PI * 2);
      ctx.fill();
    } else ctx.stroke();
  }

  const alpha = ctx.getImageData(0, 0, 32, 32).data;
  const pixels = new Array(64).fill(0);
  for (let y = 0; y < 32; y++) {
    for (let x = 0; x < 32; x++) {
      if (alpha[(y * 32 + x) * 4 + 3] >= 128) pixels[(y >> 2) * 8 + (x >> 2)]++;
    }
  }
  return pixels;
}

// ---- Page wiring (browser only) ----
if (typeof document !== "undefined") {
  const $ = (id) => document.getElementById(id);
  const pad = $("pad");
  const ctx = pad.getContext("2d");
  let strokes = [];
  let drawing = false;
  let queued = false;

  function sizePad() {
    const ratio = window.devicePixelRatio || 1;
    const css = pad.clientWidth;
    pad.width = pad.height = Math.round(css * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    redraw();
  }

  function redraw() {
    const css = pad.clientWidth;
    ctx.clearRect(0, 0, css, css);
    ctx.lineCap = ctx.lineJoin = "round";
    ctx.lineWidth = css * 0.075;
    ctx.strokeStyle = "#e6edf3";
    ctx.shadowColor = "rgba(34, 211, 238, 0.65)";
    ctx.shadowBlur = 14;
    for (const stroke of strokes) {
      ctx.beginPath();
      stroke.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      if (stroke.length === 1) ctx.lineTo(stroke[0][0] + 0.1, stroke[0][1]);
      ctx.stroke();
    }
  }

  function point(e) {
    const r = pad.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  pad.addEventListener("pointerdown", (e) => {
    pad.setPointerCapture(e.pointerId);
    drawing = true;
    strokes.push([point(e)]);
    $("pad-hint").hidden = true;
    $("truth").hidden = true;
    schedule();
  });
  pad.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    strokes[strokes.length - 1].push(point(e));
    schedule();
  });
  const stop = () => { drawing = false; };
  pad.addEventListener("pointerup", stop);
  pad.addEventListener("pointercancel", stop);

  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      redraw();
      // Scale the strokes from on-screen size before converting.
      show(strokesToPixels(strokes));
    });
  }

  function drawGrid(canvas, pixels, withNumbers) {
    const c = canvas.getContext("2d");
    const s = canvas.width / 8;
    c.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < 64; i++) {
      const v = pixels[i] / 16;
      c.fillStyle = `rgb(${Math.round(11 + v * 219)}, ${Math.round(15 + v * 196)}, ${Math.round(26 + v * 212)})`;
      c.fillRect((i % 8) * s, Math.floor(i / 8) * s, s - 1, s - 1);
      if (withNumbers && pixels[i]) {
        c.fillStyle = v > 0.55 ? "#0b0f1a" : "#8b949e";
        c.font = `${Math.round(s * 0.36)}px ui-monospace, Consolas, monospace`;
        c.textAlign = "center";
        c.textBaseline = "middle";
        c.fillText(pixels[i], (i % 8) * s + s / 2, Math.floor(i / 8) * s + s / 2);
      }
    }
  }

  function show(pixels) {
    drawGrid($("grid"), pixels, true);
    const empty = pixels.every((v) => v === 0);
    const result = empty ? null : predict(pixels);
    $("prediction").textContent = result ? result.digit : "?";
    const bars = $("votes");
    bars.innerHTML = "";
    for (let c = 0; c < 10; c++) {
      const v = result ? result.votes[c] : 0;
      const row = document.createElement("div");
      row.className = "vote" + (result && c === result.digit ? " win" : "");
      row.innerHTML = `<span class="d">${c}</span><span class="bar"><span style="width:${(v / 9) * 100}%"></span></span><span class="v">${v}</span>`;
      bars.appendChild(row);
    }
    $("vote-note").textContent = result
      ? `Digit ${result.digit} won ${result.votes[result.digit]} of its 9 head-to-head matchups.`
      : "Draw a digit, or pick a real test image below.";
  }

  $("clear").addEventListener("click", () => {
    strokes = [];
    redraw();
    $("pad-hint").hidden = false;
    $("truth").hidden = true;
    show(new Array(64).fill(0));
  });

  // Gallery of real test images.
  const gallery = $("gallery");
  M.samples.forEach((sample) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "thumb";
    b.setAttribute("aria-label", `Test image of a ${sample.label}`);
    const cv = document.createElement("canvas");
    cv.width = cv.height = 48;
    drawGrid(cv, sample.pixels, false);
    b.appendChild(cv);
    b.addEventListener("click", () => {
      strokes = [];
      redraw();
      $("pad-hint").hidden = false;
      show(sample.pixels);
      const r = predict(sample.pixels);
      const t = $("truth");
      t.hidden = false;
      t.textContent = r.digit === sample.label ? `✓ correct — the true label is ${sample.label}` : `✗ wrong — the true label is ${sample.label}`;
      t.className = "chip " + (r.digit === sample.label ? "good" : "bad");
      document.querySelector(".lab").scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    gallery.appendChild(b);
  });

  // Model report: accuracies + confusion matrix.
  const m = M.metrics;
  $("svm-acc").textContent = (m.svm_accuracy * 100).toFixed(2) + "%";
  $("rf-acc").textContent = (m.forest_accuracy * 100).toFixed(2) + "%";
  $("svm-bar").style.width = m.svm_accuracy * 100 + "%";
  $("rf-bar").style.width = m.forest_accuracy * 100 + "%";
  $("split").textContent = `${m.train_images} training images · ${m.test_images} test images · ${M.support_vectors.length} support vectors`;
  const cm = $("cm");
  const max = Math.max(...m.confusion_matrix.flat());
  let html = '<tr><th></th>' + M.classes.map((c) => `<th scope="col">${c}</th>`).join("") + "</tr>";
  m.confusion_matrix.forEach((row, i) => {
    html += `<tr><th scope="row">${i}</th>` + row.map((v, j) => {
      const a = v ? 0.12 + (0.88 * v) / max : 0;
      const color = i === j ? `rgba(34,211,238,${a})` : `rgba(244,114,182,${v ? 0.35 + a : 0})`;
      return `<td style="background:${color}" title="true ${i}, predicted ${j}: ${v}">${v || ""}</td>`;
    }).join("") + "</tr>";
  });
  cm.innerHTML = html;

  window.addEventListener("resize", sizePad);
  sizePad();
  show(new Array(64).fill(0));
}

if (typeof module !== "undefined") module.exports = { predict, strokesToPixels };
