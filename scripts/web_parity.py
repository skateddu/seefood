"""Compare the web demo's predictions with the Python package on dataset images.

Builds the site in a temporary directory, adds a parity page that classifies sample images
with web/seefood.js in a headless Chromium-based browser, and compares the probabilities
with the Python classifier. The browser's canvas resampling is not identical to Pillow's
bicubic resize, so small differences are expected; verdicts should agree.
"""

import argparse
import functools
import http.server
import json
import logging
import queue
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from build_site import SiteHandler, build_site

from seefood.classifier import HotdogClassifier
from seefood.config import ClassifierConfig

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_DIR = Path("data/raw/hotdog-or-not-a-hotdog/data/test")
DEFAULT_BROWSER = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
LABEL_DIRS = ("hotdog", "others")
PARITY_PAGE = """<!doctype html>
<meta charset="utf-8">
<script type="module">
const post = (payload) => fetch("results", { method: "POST", body: JSON.stringify(payload) });
try {
  const { classify, createSession, downloadModel, loadConfig } = await import("./seefood.js");
  const images = %s;
  const config = await loadConfig();
  const session = await createSession(await downloadModel(config.modelUrl, () => {}));
  const results = {};
  for (const image of images) {
    const bitmap = await createImageBitmap(await (await fetch(image)).blob());
    results[image] = (await classify(session, bitmap, config)).hotdogProbability;
  }
  await post(results);
} catch (error) {
  await post({ error: `${error}\\n${error.stack}` });
}
</script>
"""


class ParityHandler(SiteHandler):
    """Serves the site and receives the browser's results on ``POST /results``."""

    def __init__(self, *args: Any, results: "queue.Queue[dict[str, float]]", **kwargs: Any):
        self.results = results
        super().__init__(*args, **kwargs)

    def do_POST(self) -> None:  # noqa: N802 - name required by http.server
        """Store the posted JSON results."""
        length = int(self.headers["Content-Length"])
        self.results.put(json.loads(self.rfile.read(length)))
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Silence per-request logging."""


def sample_images(split_dir: Path, per_label: int) -> list[Path]:
    """Pick the first images of each label, sorted by name.

    Parameters
    ----------
    split_dir : Path
        Dataset split with one sub-directory per label.
    per_label : int
        Number of images per label.

    Returns
    -------
    list[Path]
        Selected image files.
    """
    return [
        path
        for label_dir in LABEL_DIRS
        for path in sorted((split_dir / label_dir).glob(pattern="*.jpg"))[:per_label]
    ]


def prepare_site(site: Path, images: list[Path], config: ClassifierConfig) -> list[str]:
    """Build the site with the sample images and the parity page.

    Parameters
    ----------
    site : Path
        Destination directory.
    images : list[Path]
        Images to classify in the browser.
    config : ClassifierConfig
        Configuration of the classifier.

    Returns
    -------
    list[str]
        Site-relative URLs of the images, aligned with ``images``.
    """
    build_site(site, config)
    (site / "samples").mkdir()
    urls = []
    for index, image in enumerate(images):
        url = f"samples/{index}{image.suffix}"
        shutil.copy2(image, site / url)
        urls.append(url)
    (site / "parity.html").write_text(PARITY_PAGE % json.dumps(urls), encoding="utf-8")
    return urls


def run_browser(site: Path, browser: Path, timeout: float) -> dict[str, float]:
    """Open the parity page in a headless browser and wait for its results.

    Parameters
    ----------
    site : Path
        Built site containing ``parity.html``.
    browser : Path
        Chromium-based browser executable.
    timeout : float
        Seconds to wait for the results.

    Returns
    -------
    dict[str, float]
        Hotdog probability per image URL.
    """
    results: queue.Queue[dict[str, float]] = queue.Queue()
    handler = functools.partial(ParityHandler, directory=str(site), results=results)
    with (
        http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server,
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile,
    ):
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/parity.html"
        command = [str(browser), "--headless=new", f"--user-data-dir={profile}", url]
        with subprocess.Popen(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ) as process:
            try:
                payload = results.get(timeout=timeout)
            finally:
                process.terminate()
                server.shutdown()
    if "error" in payload:
        raise RuntimeError(f"Parity page failed: {payload['error']}")
    return payload


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--per-label", type=int, default=10)
    parser.add_argument("--browser", type=Path, default=DEFAULT_BROWSER)
    parser.add_argument("--timeout", type=float, default=300)
    return parser.parse_args()


def main() -> None:
    """Classify the sample images in Python and in the browser, then print the comparison."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    args = parse_args()
    config = ClassifierConfig()
    images = sample_images(args.split_dir, args.per_label)
    with tempfile.TemporaryDirectory() as site:
        urls = prepare_site(Path(site) / "site", images, config)
        browser_results = run_browser(Path(site) / "site", args.browser, args.timeout)

    classifier = HotdogClassifier.from_config(config)
    print("| Image | Python | Browser | Difference | Same verdict |")
    print("| --- | --- | --- | --- | --- |")
    agreements = 0
    for image, url in zip(images, urls, strict=True):
        python_probability = classifier.predict(image).hotdog_probability
        browser_probability = browser_results[url]
        same_verdict = (python_probability >= config.threshold) == (
            browser_probability >= config.threshold
        )
        agreements += same_verdict
        print(
            f"| {image.parent.name}/{image.name} | {python_probability:.4f} "
            f"| {browser_probability:.4f} | {browser_probability - python_probability:+.4f} "
            f"| {'yes' if same_verdict else 'NO'} |"
        )
    logger.info("Same verdict on %d of %d images", agreements, len(images))


if __name__ == "__main__":
    main()
