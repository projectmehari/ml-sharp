"""
SharpView · Modal Backend
Deploy:  modal deploy modal_app.py
"""

import modal
from fastapi import Request
from fastapi.responses import Response, JSONResponse

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.13",
    )
    .run_commands("apt-get update -qq && apt-get install -y git wget")
    .pip_install("uv")
    .run_commands("uv pip install --system git+https://github.com/apple/ml-sharp.git")
    .run_commands(
        "mkdir -p /root/.cache/torch/hub/checkpoints && "
        "wget -q -O /root/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt "
        "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
    )
    .pip_install("pillow", "python-multipart", "fastapi")
)

app = modal.App("sharpview", image=image)

CHECKPOINT = "/root/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt"

# Declare the FastAPI app at module level
web_app = modal.asgi_app()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.get("/health")
def health():
    return {"status": "ok", "model": "apple/ml-sharp"}

@fastapi_app.post("/predict")
async def predict(request: Request):
    import tempfile, subprocess, os, pathlib

    content_type = request.headers.get("content-type", "")
    ext = ".jpg"

    try:
        if "multipart" in content_type:
            form = await request.form()
            image_file = form.get("image") or next(iter(form.values()), None)
            if image_file is None:
                return JSONResponse({"error": "no image field"}, status_code=400)
            image_bytes = await image_file.read()
            fname = getattr(image_file, "filename", "input.jpg") or "input.jpg"
            ext = pathlib.Path(fname).suffix.lower() or ".jpg"
        else:
            image_bytes = await request.body()
            if image_bytes[:4] == b'\x89PNG':
                ext = ".png"
            elif image_bytes[:4] == b'RIFF':
                ext = ".webp"
            else:
                ext = ".jpg"
    except Exception as e:
        return JSONResponse({"error": f"parse error: {e}"}, status_code=400)

    if not image_bytes:
        return JSONResponse({"error": "empty body"}, status_code=400)

    print(f"Image received: {len(image_bytes)} bytes, ext={ext}")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "in")
        output_dir = os.path.join(tmpdir, "out")
        os.makedirs(input_dir)
        os.makedirs(output_dir)

        with open(os.path.join(input_dir, f"image{ext}"), "wb") as f:
            f.write(image_bytes)

        cmd = ["sharp", "predict", "-i", input_dir, "-o", output_dir, "-c", CHECKPOINT]
        print(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        print("stdout:", result.stdout[:400])
        print("stderr:", result.stderr[:400])

        if result.returncode != 0:
            return JSONResponse({
                "error": "SHARP failed",
                "stderr": result.stderr[-500:],
            }, status_code=500)

        ply_files = list(pathlib.Path(output_dir).glob("**/*.ply"))
        if not ply_files:
            all_files = [str(f) for f in pathlib.Path(output_dir).rglob("*")]
            return JSONResponse({"error": "no PLY found", "files": all_files}, status_code=500)

        ply_bytes = ply_files[0].read_bytes()
        print(f"✓ PLY: {len(ply_bytes)//1024}kb")

        return Response(
            content=ply_bytes,
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="output.ply"'},
        )


@app.function(
    image=image,
    gpu="A10G",
    timeout=120,
    container_idle_timeout=60,
)
@modal.asgi_app()
def serve():
    return fastapi_app
