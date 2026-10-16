// Model loading, preprocessing and classification, mirroring the Python package.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

/**
 * Fetch the classifier settings written by scripts/build_site.py from ClassifierConfig.
 * @param {string} url
 * @returns {Promise<{modelUrl: string, threshold: number, hotdogClassIndex: number,
 *   imageSize: number, mean: number[], std: number[]}>}
 */
export async function loadConfig(url = "config.json") {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Cannot load ${url}: HTTP ${response.status}`);
  }
  return response.json();
}

/**
 * Download the ONNX model, reporting progress between 0 and 1 when the size is known.
 * @param {string} url
 * @param {(fraction: number) => void} onProgress
 * @returns {Promise<Uint8Array>}
 */
export async function downloadModel(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Cannot download ${url}: HTTP ${response.status}`);
  }
  const total = Number(response.headers.get("Content-Length")) || 0;
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    // Content-Length is the compressed size if the server gzips the file: clamp to 1.
    if (total) onProgress(Math.min(received / total, 1));
  }
  const model = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    model.set(chunk, offset);
    offset += chunk.length;
  }
  return model;
}

/**
 * Create a WebAssembly inference session for the model bytes.
 * @param {Uint8Array} model
 * @returns {Promise<ort.InferenceSession>}
 */
export function createSession(model) {
  return ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
}

/**
 * Center-crop the image to a square, scale it to the model input and normalize it (NCHW).
 * Cropping the shorter side and scaling in one drawImage call matches resize-then-crop.
 * @param {ImageBitmap} bitmap
 * @param {{imageSize: number, mean: number[], std: number[]}} config
 * @returns {ort.Tensor}
 */
export function preprocess(bitmap, config) {
  const size = config.imageSize;
  const side = Math.min(bitmap.width, bitmap.height);
  const canvas = new OffscreenCanvas(size, size);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingQuality = "high";
  context.drawImage(
    bitmap,
    (bitmap.width - side) / 2,
    (bitmap.height - side) / 2,
    side,
    side,
    0,
    0,
    size,
    size,
  );
  const pixels = context.getImageData(0, 0, size, size).data;
  const area = size * size;
  const tensor = new Float32Array(3 * area);
  for (let pixel = 0; pixel < area; pixel++) {
    for (let channel = 0; channel < 3; channel++) {
      const value = pixels[pixel * 4 + channel] / 255;
      tensor[channel * area + pixel] = (value - config.mean[channel]) / config.std[channel];
    }
  }
  return new ort.Tensor("float32", tensor, [1, 3, size, size]);
}

/**
 * Classify an image: hotdog if the probability of the ImageNet hotdog class reaches the threshold.
 * @param {ort.InferenceSession} session
 * @param {ImageBitmap} bitmap
 * @param {{threshold: number, hotdogClassIndex: number, imageSize: number,
 *   mean: number[], std: number[]}} config
 * @returns {Promise<{isHotdog: boolean, hotdogProbability: number}>}
 */
export async function classify(session, bitmap, config) {
  const feeds = { [session.inputNames[0]]: preprocess(bitmap, config) };
  const outputs = await session.run(feeds);
  const probabilities = outputs[session.outputNames[0]].data;
  const hotdogProbability = probabilities[config.hotdogClassIndex];
  return { isHotdog: hotdogProbability >= config.threshold, hotdogProbability };
}
