## Contents

- `image_translator.py` — main single-file pipeline. Useful as reference implementation.
- `fonts/` — local fonts directory. The pipeline will prefer downloaded fonts and fall back to fonts placed here.
- `process-images/` — working directory where masks, inpainted images and final images are written.
- `Dockerfile` — containerize the pipeline for local dev/testing.
- `requirements.txt` — Python dependencies used when building the Docker image or creating a venv.

## Quick start (local, without Docker)

1. Create and activate a virtual environment (PowerShell):

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

2. Install dependencies:

    pip install -r requirements.txt

3. Set required environment variables (example using PowerShell):

    $env:AZURE_API_KEY = "<your-azure-key>"
    $env:AZURE_ENDPOINT = "https://<your-azure-endpoint>"
    $env:REPLICATE_API_KEY = "<your-replicate-key>"
    $env:WHATFONTIS_API_KEY = "<your-whatfontis-key>"
    $env:SEGMENT_MASK_DEBUG = "1"
    $env:SEGMENT_MASK_DILATE = "20"

4. Run the pipeline on the example image in `test-images/` (or point to your image):

    python image_translator.py

Outputs will be written to `process-images/` (mask, inpainted, and final image).

## Docker (recommended for reproducible runs)

Build the image:

    docker build -t expense-trend:latest .

Run the container (mount `process-images` and `fonts` so outputs persist locally):

    docker run --rm \
      -e AZURE_API_KEY="<your-azure-key>" \
      -e AZURE_ENDPOINT="https://<your-azure-endpoint>" \
      -e REPLICATE_API_KEY="<your-replicate-key>" \
      -e WHATFONTIS_API_KEY="<your-whatfontis-key>" \
      -v ${PWD}/process-images:/app/process-images \
      -v ${PWD}/fonts:/app/fonts \
      expense-trend:latest

## Environment variables and adjustment variables

- `AZURE_API_KEY` — Azure Cognitive Services key used for OCR and brand detection.
- `AZURE_ENDPOINT` — Your Azure endpoint, defaults are present in the script but you should set your own.
- `REPLICATE_API_KEY` — Replicate API key used for Lang‑SAM segmentation and LaMa inpainting.
- `WHATFONTIS_API_KEY` — Optional key for WhatFontIs font identification.
- `SEGMENT_MASK_DILATE` — How many pixels to expand segmentation masks. Larger values are more aggressive.
- `SEGMENT_MASK_EXPAND_MODE` — `dt` (distance transform, smoother) or `morph` (morphological dilation).
- `SEGMENT_MASK_CLOSE_KERNEL` — Kernel size for morphological close smoothing.
- `SEGMENT_MASK_DEBUG` — If `1`, saves debug masks to `process-images/debug_masks/`.
WHATFONTIS_API_KEY=
OPENAI_API_KEY =
AZURE_API_KEY=
REPLICATE_API_KEY=
