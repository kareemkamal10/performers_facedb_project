// ============================================================
// embed_batch.js — Batch face-embedding worker, invoked once per pipeline
// batch by pipeline.py.
//
// Launches ONE headless Chromium (via Puppeteer) and processes every image
// path in the manifest through the EXACT same detection/alignment/embedding
// code as the real browser search page (build_page.html), using several
// tabs in parallel (manifest.concurrency) - so results match the browser
// search exactly, which is the whole point of running this in a real
// browser instead of a Python re-implementation.
//
// Usage: node embed_batch.js <manifest.json> <output.json>
//
// manifest.json shape:
//   {
//     "baseDir": "/path/to/face_embed",   // serves build_page.html + models
//     "concurrency": 4,                    // parallel browser tabs
//     "items": [ {"id": "...", "paths": ["/abs/path/0000.jpg", ...]}, ... ]
//   }
//
// output.json shape:
//   {
//     "<id>": {"ok": true, "embedding": [512 floats], "num_images_used": n},
//     "<id2>": {"ok": false}
//   }
// ============================================================

const fs = require("fs");
const path = require("path");
const express = require("express");
const puppeteer = require("puppeteer");

const [, , MANIFEST_PATH, OUTPUT_PATH] = process.argv;
if (!MANIFEST_PATH || !OUTPUT_PATH) {
  console.error("Usage: node embed_batch.js <manifest.json> <output.json>");
  process.exit(1);
}

const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf-8"));
const BASE_DIR = manifest.baseDir;
const CONCURRENCY = Math.max(1, manifest.concurrency || 4);
const items = manifest.items || [];
const RECOVERY_MODE = !!manifest.recoveryMode;
const MIN_FACE_CONFIDENCE = manifest.minFaceDetectionConfidence || 0.3;

function mimeFor(fname) {
  const ext = path.extname(fname).toLowerCase();
  if (ext === ".png") return "image/png";
  if (ext === ".webp") return "image/webp";
  return "image/jpeg";
}

// ----- float32 -> float16 -> float32 round-trip -----
// We don't need a packed binary format here (the output is JSON); the point
// is that every stored number is snapped to the nearest value a real
// float16 could represent, matching config.EMBEDDING_DTYPE.
function float32ToFloat16Bits(val) {
  const floatView = new Float32Array(1);
  const int32View = new Int32Array(floatView.buffer);
  floatView[0] = val;
  const x = int32View[0];

  let bits = (x >> 16) & 0x8000; // sign
  let m = (x >> 12) & 0x07ff; // mantissa
  const e = (x >> 23) & 0xff; // exponent

  if (e < 103) return bits;
  if (e > 142) {
    bits |= 0x7c00;
    bits |= (e === 255 ? 0 : 1) && (x & 0x007fffff ? 1 : 0);
    return bits;
  }
  if (e < 113) {
    m |= 0x0800;
    bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
    return bits;
  }
  bits |= ((e - 112) << 10) | (m >> 1);
  bits += m & 1;
  return bits;
}

function float16BitsToFloat32(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;

  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  } else if (e === 0x1f) {
    return f ? NaN : (s ? -1 : 1) * Infinity;
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function roundToFloat16(x) {
  return float16BitsToFloat32(float32ToFloat16Bits(x));
}

async function main() {
  const app = express();
  app.use(express.static(BASE_DIR));
  const server = app.listen(0);
  await new Promise((resolve) => server.once("listening", resolve));
  const port = server.address().port;

  const browser = await puppeteer.launch({
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  // One tab per concurrency slot; models are loaded ONCE per tab and reused
  // for every image that tab ends up processing.
  const pages = [];
  for (let i = 0; i < CONCURRENCY; i++) {
    const page = await browser.newPage();
    page.on("pageerror", (err) => console.error("[browser error]", err.message));
    await page.goto(`http://localhost:${port}/build_page.html`, { waitUntil: "load" });
    await page.evaluate(
      (opts) => window.initModels(opts),
      RECOVERY_MODE ? { minFaceDetectionConfidence: MIN_FACE_CONFIDENCE } : undefined
    );
    pages.push(page);
  }

  // Flatten to a single work queue of {id, filePath}.
  const queue = [];
  for (const item of items) {
    for (const p of item.paths) queue.push({ id: item.id, filePath: p });
  }

  const embeddingsById = {}; // id -> [ [512 floats], ... ] (one per successful image)
  let cursor = 0;

  async function worker(page) {
    while (cursor < queue.length) {
      const { id, filePath } = queue[cursor++];
      try {
        const buf = fs.readFileSync(filePath);
        const dataUrl = `data:${mimeFor(filePath)};base64,${buf.toString("base64")}`;
        const fnName = RECOVERY_MODE ? "processImageRecovery" : "processImage";
        const result = await page.evaluate(
          (fn, url) => window[fn](url),
          fnName,
          dataUrl
        );
        if (result && result.ok) {
          if (!embeddingsById[id]) embeddingsById[id] = [];
          embeddingsById[id].push(result.embedding);
        }
      } catch (err) {
        // Treat any read/decode/inference error the same as "no face found" -
        // this one image just doesn't contribute, the element isn't failed
        // outright unless NONE of its images succeed.
      }
    }
  }

  await Promise.all(pages.map((p) => worker(p)));

  await browser.close();
  server.close();

  // Combine per-element: average multiple successful embeddings (unit
  // vectors), re-normalize, then snap every component to float16 precision.
  const output = {};
  for (const item of items) {
    const embs = embeddingsById[item.id];
    if (!embs || embs.length === 0) {
      output[item.id] = { ok: false };
      continue;
    }

    const dim = embs[0].length;
    const avg = new Array(dim).fill(0);
    for (const e of embs) {
      for (let i = 0; i < dim; i++) avg[i] += e[i];
    }
    for (let i = 0; i < dim; i++) avg[i] /= embs.length;

    let norm = 0;
    for (let i = 0; i < dim; i++) norm += avg[i] * avg[i];
    norm = Math.sqrt(norm) || 1;

    const embedding = new Array(dim);
    for (let i = 0; i < dim; i++) embedding[i] = roundToFloat16(avg[i] / norm);

    output[item.id] = { ok: true, embedding, num_images_used: embs.length };
  }

  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(output));
}

main().catch((err) => {
  console.error("embed_batch.js fatal error:", err);
  process.exit(1);
});
