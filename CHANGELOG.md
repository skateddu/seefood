# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-10-17

### Added

- EfficientNet-B2 (ImageNet) exported to ONNX with softmax in the graph (`scripts/export.py`).
- Hotdog classifier on ImageNet class 934 with NumPy/Pillow preprocessing matching torchvision,
  including EXIF orientation for phone photos.
- Threshold calibration on the "Hotdog or not a hotdog" dataset (`scripts/calibrate.py`):
  threshold 0.025, 98.0% test accuracy.
- `seefood` command (and `python -m seefood`): text or JSON Lines output, `--threshold`,
  `--model`, `--verbose`; unreadable images are reported and skipped, with exit code 1.
- Web demo (`web/`): the SeeFood screen running the model in the browser with onnxruntime-web,
  built by `scripts/build_site.py` with settings taken from `ClassifierConfig`. Photos come from
  a file, drag and drop, paste, or the device camera with a live viewfinder.
- `scripts/web_parity.py`: compares browser and Python predictions in headless Edge.
- GitHub Actions: CI (Ruff, pytest on Python 3.10–3.13, integration tests) and GitHub Pages deployment.

[Unreleased]: https://github.com/skateddu/seefood/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/skateddu/seefood/releases/tag/v1.0.0
