// SeeFood page: loads the model, takes a photo and shows the verdict.
import { classify, createSession, downloadModel, loadConfig } from "./seefood.js";

const MAX_FILE_BYTES = 20 * 1024 * 1024;

const elements = {
  viewfinder: document.getElementById("viewfinder"),
  photo: document.getElementById("photo"),
  placeholder: document.getElementById("placeholder"),
  banner: document.getElementById("banner"),
  verdict: document.getElementById("verdict"),
  badge: document.getElementById("badge"),
  probability: document.getElementById("probability"),
  fileInput: document.getElementById("file-input"),
  camera: document.getElementById("camera"),
  shutter: document.getElementById("shutter"),
  cameraToggle: document.getElementById("camera-toggle"),
  status: document.getElementById("status"),
  progress: document.getElementById("progress"),
};

const CAMERA_ERRORS = {
  NotAllowedError: "Camera access was blocked. Allow it in your browser settings, or choose a photo.",
  NotFoundError: "No camera found. Choose a photo instead.",
  NotReadableError: "The camera is in use by another app. Close it, or choose a photo.",
};

const CAMERA_PROMPT = "Point the camera at your food and take a photo.";

const ready = loadModel();
let cameraStream = null;

async function loadModel() {
  try {
    const config = await loadConfig();
    const model = await downloadModel(config.modelUrl, (fraction) => {
      elements.progress.value = fraction;
      elements.status.textContent = `Downloading the model… ${Math.round(fraction * 100)}%`;
    });
    elements.status.textContent = "Starting the model…";
    const session = await createSession(model);
    elements.progress.hidden = true;
    elements.status.textContent =
      elements.viewfinder.dataset.state === "camera"
        ? CAMERA_PROMPT
        : "Ready. Choose or take a photo of your food.";
    return { config, session };
  } catch (error) {
    console.error(error);
    elements.progress.hidden = true;
    elements.status.textContent =
      "The model couldn't be loaded. Check your connection and reload the page.";
    throw error;
  }
}

function showPhoto(file) {
  if (elements.photo.src) URL.revokeObjectURL(elements.photo.src);
  elements.photo.src = URL.createObjectURL(file);
  elements.photo.hidden = false;
  elements.placeholder.hidden = true;
}

function hideVerdict() {
  elements.banner.hidden = true;
  elements.badge.hidden = true;
  elements.probability.hidden = true;
}

function showVerdict({ isHotdog, hotdogProbability }) {
  const verdict = isHotdog ? "hotdog" : "not-hotdog";
  elements.viewfinder.dataset.verdict = verdict;
  elements.verdict.textContent = isHotdog ? "Hotdog!" : "Not hotdog!";
  elements.probability.textContent = `Hotdog probability: ${(hotdogProbability * 100).toFixed(1)}%`;
  elements.banner.hidden = false;
  elements.badge.hidden = false;
  elements.probability.hidden = false;
}

function rejectFile(message) {
  elements.viewfinder.dataset.state = "empty";
  elements.status.textContent = message;
}

async function startCamera() {
  try {
    // The rear camera on phones; desktops fall back to their only camera.
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
  } catch (error) {
    console.error(error);
    elements.status.textContent =
      CAMERA_ERRORS[error.name] ?? "The camera couldn't be started. Choose a photo instead.";
    return;
  }
  hideVerdict();
  elements.camera.srcObject = cameraStream;
  await elements.camera.play();
  elements.photo.hidden = true;
  elements.placeholder.hidden = true;
  elements.camera.hidden = false;
  elements.shutter.hidden = false;
  elements.viewfinder.dataset.state = "camera";
  elements.cameraToggle.textContent = "Close camera";
  elements.status.textContent = CAMERA_PROMPT;
}

function stopCamera() {
  if (!cameraStream) return;
  for (const track of cameraStream.getTracks()) track.stop();
  cameraStream = null;
  elements.camera.srcObject = null;
  elements.camera.hidden = true;
  elements.shutter.hidden = true;
  elements.cameraToggle.textContent = "Use camera";
  if (elements.photo.hidden) {
    elements.placeholder.hidden = false;
    elements.viewfinder.dataset.state = "empty";
  }
}

function takePhoto() {
  const { videoWidth, videoHeight } = elements.camera;
  if (!videoWidth) return;
  const canvas = document.createElement("canvas");
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  canvas.getContext("2d").drawImage(elements.camera, 0, 0, videoWidth, videoHeight);
  stopCamera();
  canvas.toBlob(
    (blob) => handleFile(new File([blob], "camera.jpg", { type: "image/jpeg" })),
    "image/jpeg",
    0.92,
  );
}

async function handleFile(file) {
  if (!file) return;
  stopCamera();
  if (!file.type.startsWith("image/")) {
    rejectFile("That file isn't an image. Choose a JPEG or PNG photo.");
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    rejectFile("That photo is over 20 MB. Choose a smaller one.");
    return;
  }
  hideVerdict();
  showPhoto(file);
  elements.viewfinder.dataset.state = "evaluating";
  elements.status.textContent = "Evaluating…";
  try {
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const { config, session } = await ready;
    const result = await classify(session, bitmap, config);
    bitmap.close();
    showVerdict(result);
    elements.viewfinder.dataset.state = "result";
    elements.status.textContent = "Choose or take another photo to try again.";
  } catch (error) {
    console.error(error);
    rejectFile("This photo couldn't be read. Try a JPEG or PNG.");
  }
}

elements.fileInput.addEventListener("change", () => {
  handleFile(elements.fileInput.files[0]);
  elements.fileInput.value = "";
});

if (navigator.mediaDevices?.getUserMedia) {
  elements.cameraToggle.hidden = false;
  elements.cameraToggle.addEventListener("click", () => {
    if (cameraStream) stopCamera();
    else startCamera();
  });
  elements.shutter.addEventListener("click", takePhoto);
}

document.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.viewfinder.classList.add("is-dragging");
});
document.addEventListener("dragleave", (event) => {
  if (event.relatedTarget === null) elements.viewfinder.classList.remove("is-dragging");
});
document.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.viewfinder.classList.remove("is-dragging");
  handleFile(event.dataTransfer.files[0]);
});
document.addEventListener("paste", (event) => {
  const item = [...event.clipboardData.items].find((entry) => entry.type.startsWith("image/"));
  if (item) handleFile(item.getAsFile());
});

ready.catch(() => {});
