"""Assemble the static web demo: the web/ files, the ONNX model and a config.json.

The config is written from ClassifierConfig, so the browser uses the same threshold,
input size and normalization as the Python package.
"""

import argparse
import functools
import http.server
import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path

from seefood.config import ClassifierConfig

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DEFAULT_OUTPUT = Path("build/site")
MODEL_DIR = "models"


class SiteHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler that keeps connections open after each response.

    With the default HTTP/1.0, the server closes the socket right after writing, and on
    Windows that can cut large files (the 37 MB model) short with a connection reset.
    """

    protocol_version = "HTTP/1.1"


def site_config(config: ClassifierConfig, model_url: str) -> dict[str, object]:
    """Return the browser-side settings derived from the Python configuration.

    Parameters
    ----------
    config : ClassifierConfig
        Configuration of the Python classifier.
    model_url : str
        URL of the ONNX model, relative to the page.

    Returns
    -------
    dict[str, object]
        Settings serialized into ``config.json``, with camelCase keys for JavaScript.
    """
    settings = asdict(config)
    return {
        "modelUrl": model_url,
        "threshold": settings["threshold"],
        "hotdogClassIndex": settings["hotdog_class_index"],
        "imageSize": settings["image_size"],
        "mean": list(settings["mean"]),
        "std": list(settings["std"]),
    }


def build_site(output: Path, config: ClassifierConfig) -> None:
    """Copy the web files and the model into ``output`` and write ``config.json``.

    Parameters
    ----------
    output : Path
        Destination directory, replaced if it already exists.
    config : ClassifierConfig
        Configuration providing the model and the classifier settings.
    """
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(WEB_DIR, output)
    model_url = f"{MODEL_DIR}/{config.model_path.name}"
    (output / MODEL_DIR).mkdir()
    shutil.copy2(config.model_path, output / model_url)
    config_json = json.dumps(site_config(config, model_url), indent=2)
    (output / "config.json").write_text(config_json + "\n", encoding="utf-8")
    logger.info("Built site in %s", output)


def serve(directory: Path, port: int) -> None:
    """Serve ``directory`` over HTTP on localhost until interrupted.

    Parameters
    ----------
    directory : Path
        Directory to serve.
    port : int
        TCP port to listen on.
    """
    handler = functools.partial(SiteHandler, directory=str(directory))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        logger.info("Serving on http://127.0.0.1:%d (Ctrl+C to stop)", port)
        server.serve_forever()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="site directory")
    parser.add_argument("--serve", action="store_true", help="serve the site after building it")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    """Build the site, then optionally serve it."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)
    args = parse_args()
    build_site(args.output, ClassifierConfig())
    if args.serve:
        serve(directory=args.output, port=args.port)


if __name__ == "__main__":
    main()
