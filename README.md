## Contents

- `image_translator.py` — main single-file pipeline. Useful as reference implementation.
- `fonts/` — local fonts directory. The pipeline will prefer downloaded fonts and fall back to fonts placed here.
- `process-images/` — working directory where masks, inpainted images, and final images are written.
- `Dockerfile` — containerize the pipeline for local dev/testing.
- `requirements.txt` — Python dependencies used when building the Docker image or creating a venv.

---

## Quick Start (Local, Without Docker)

1. **Create and activate a virtual environment** (PowerShell example):

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

2. **Install dependencies:**

    ```powershell
    pip install -r requirements.txt
    ```

3. **Create a `.env` file** in the project root with the following content:

    ```env
    AZURE_API_KEY=<your-azure-key>
    AZURE_ENDPOINT=https://<your-azure-endpoint>
    REPLICATE_API_KEY=<your-replicate-key>
    WHATFONTIS_API_KEY=<your-whatfontis-key>
    SEGMENT_MASK_DEBUG=1
    SEGMENT_MASK_DILATE=20
    ```

4. **Run the pipeline** on the example image in `test-images/` (or point to your image):

    ```powershell
    python image_translator.py
    ```

Outputs will be written to `process-images/` (mask, inpainted, and final image).

---

## Docker (Recommended for Reproducible Runs)

1. **Build the image:**

    ```bash
    docker build -t expense-trend:latest .
    ```

2. **Run the container** (mount `process-images` and `fonts` so outputs persist locally):

    ```bash
    docker run --rm \
      --env-file .env \
      -v ${PWD}/process-images:/app/process-images \
      -v ${PWD}/fonts:/app/fonts \
      expense-trend:latest
    ```
